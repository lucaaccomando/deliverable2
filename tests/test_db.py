import sqlite3

import pytest

from src import db


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


def make_run(conn, sha="abc"):
    return db.get_or_create_run(conn, "file.csv", sha, 100, 95.0)


def test_same_file_maps_to_same_run(conn):
    assert make_run(conn) == make_run(conn)
    assert make_run(conn, "different") != make_run(conn)


def test_label_roundtrip_and_overwrite(conn):
    run = make_run(conn)
    db.save_label(conn, run, 7, 98.2, True, "src_bytes +5.0", {"service": "http"}, "confirmed")
    assert db.get_labels(conn, run) == {7: "confirmed"}
    db.save_label(conn, run, 7, 98.2, True, "src_bytes +5.0", {"service": "http"}, "false_positive")
    assert db.get_labels(conn, run) == {7: "false_positive"}          # overwritten, not duplicated
    db.clear_label(conn, run, 7)
    assert db.get_labels(conn, run) == {}


def test_labels_survive_restart(tmp_path):
    path = tmp_path / "persist.db"
    first = db.connect(path)
    run = make_run(first)
    db.save_label(first, run, 3, 91.0, True, "", {}, "confirmed")
    first.close()
    second = db.connect(path)                        # simulates restarting the app
    assert db.get_labels(second, make_run(second)) == {3: "confirmed"}
    second.close()


def test_invalid_label_rejected_by_code_and_by_database(conn):
    run = make_run(conn)
    with pytest.raises(ValueError):
        db.save_label(conn, run, 1, 90.0, True, "", {}, "maybe")
    with pytest.raises(sqlite3.IntegrityError):      # the CHECK constraint is a second line of defence
        conn.execute("INSERT INTO alerts (run_id, record_index, score, flagged, top_features, record_json,"
                     " analyst_label, labeled_at) VALUES (?, 1, 1, 1, '', '{}', 'maybe', 'now')", (run,))


def test_alert_requires_existing_run(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.save_label(conn, 999, 1, 90.0, True, "", {}, "confirmed")


def test_hostile_strings_are_stored_as_data_not_sql(conn):
    evil = "x'); DROP TABLE alerts; --"
    run = db.get_or_create_run(conn, evil, "sha-evil", 1, 95.0)
    db.save_label(conn, run, 0, 99.0, True, evil, {"service": evil}, "confirmed")
    history = db.get_labeled_alerts(conn, run)
    assert history.loc[0, "filename"] == evil and history.loc[0, "rec_service"] == evil


def test_history_flattens_record_and_filters_by_run(conn):
    a, b = make_run(conn, "a"), make_run(conn, "b")
    db.save_label(conn, a, 1, 90.0, True, "r", {"service": "http"}, "confirmed")
    db.save_label(conn, b, 2, 80.0, False, "r", {"service": "ftp"}, "false_positive")
    assert list(db.get_labeled_alerts(conn, a)["rec_service"]) == ["http"]
    assert len(db.get_labeled_alerts(conn)) == 2
    assert db.get_labeled_alerts(conn, 12345).empty
