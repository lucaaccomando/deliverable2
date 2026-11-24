# Technology guide: what each piece is, why it is here, and where to find it

Written so every technology in ThreatLens can be explained to a user or an instructor in plain terms.

| Technology | What it is | Why ThreatLens uses it | Where in the code |
|---|---|---|---|
| **Python 3.12** | The language. | Required by the course; best ecosystem for data and ML. | everything |
| **Streamlit** | A library that turns a Python script into a web app. Every click re-runs the script from top to bottom. | Gives a file uploader, slider, tables and charts without writing HTML/JavaScript. | `app.py` |
| **pandas** | Tables ("DataFrames") in Python. | Loading CSVs, validating columns, ranking alerts. | `src/data_loader.py`, `app.py` |
| **NumPy** | Fast arrays and math. | Percentile conversion, z-scores, threshold sweeps. | `src/model.py`, `src/metrics.py` |
| **scikit-learn** | The standard Python machine-learning library. | Provides `IsolationForest`, `Pipeline`, `OneHotEncoder`, `FunctionTransformer`. | `src/model.py` |
| **Isolation Forest** | The anomaly-detection algorithm. | See [ISOLATION_FOREST.md](ISOLATION_FOREST.md). | `src/model.py` |
| **Pipeline** | Chains preprocessing and the model into one object. | Training and scoring cannot accidentally use different preprocessing. | `build_pipeline` |
| **One-hot encoding** | Turns a category (`protocol_type = tcp`) into 0/1 columns, one per category. | The forest needs numbers. `handle_unknown="ignore"` means a new service in an upload becomes all zeros instead of an error. | `build_pipeline` |
| **`log1p` transform** | `log(1 + x)`: compresses huge values (byte counts can reach 10^9) while keeping 0 as 0. | Random splits over a huge range rarely separate ordinary values. See the ablation in the README. | `build_pipeline` |
| **joblib** | Saves a Python object (the trained model) to a file and loads it back. | Training takes about 20 s; the app should start instantly, so it loads a saved model. | `save_bundle` / `load_bundle` |
| **SQLite (`sqlite3`)** | A database stored in one file, built into Python. | Labels must survive a restart. Single user, local, nothing to install or host. | `src/db.py` |
| **SQL placeholders (`?`)** | Values are passed separately from the SQL text. | Prevents SQL injection from uploaded file contents. Covered by a test. | `src/db.py`, `tests/test_db.py` |
| **Plotly** | Interactive charts. | Score histogram with the threshold line, explanation bars, precision/recall curves. | `app.py` |
| **pytest** | Test runner. | 38 automated tests for loading, model, database and metrics. | `tests/` |
| **NSL-KDD** | A public benchmark of labeled network connections (41 features each), cleaned from KDD Cup '99. | Realistic columns for the demo and a way to measure the detector. | `data/`, `src/train.py` |

## Key concepts, in plain language

- **Anomaly score (0-100).** The percentage of held-out *normal* connections that look less unusual than this
  one. Score 97 = more unusual than 97% of normal traffic.
- **Threshold / sensitivity slider.** Connections at or above the threshold are flagged. Moving it down finds
  more threats and creates more false alarms.
- **Precision.** Of the alerts raised, the share that were real threats. **Recall.** Of the real threats, the
  share that were alerted. **F1.** Single number combining the two. **False-alarm rate.** Share of normal traffic
  wrongly flagged.
- **Why the explanation is "an approximation".** The forest scores combinations of features across 200 trees.
  The explanation compares each feature with normal traffic one at a time (z-score = how many standard
  deviations from the normal average). They usually agree, but the explanation is not literally what the forest did.
- **Why labels go to a database.** The analyst's judgement is the only ground truth in a real deployment.
  Storing it lets the app compute precision on reviewed alerts and suggest a better threshold.
- **Why train and test files differ.** The model trains on `KDDTrain+` only. All reported numbers come from
  `KDDTest+`, which the model never saw and which contains 17 attack types absent from training.

## Questions you should be able to answer

1. *Why is there no "accuracy" number?* Attacks are rare in real traffic, so a model that never alerts is
   "accurate". Precision, recall and false-alarm rate say what the analyst actually experiences.
2. *Why train only on normal traffic?* Isolation Forest models normal behaviour; including attacks in training
   teaches it that attacks are normal.
3. *Why calibrate on separate normal data?* Scores on rows the forest trained on are too optimistic. Held-out
   normal rows give an honest scale.
4. *What happens with a service the model has never seen?* It is encoded as all zeros, scored normally, and the
   explanation lists it as a rare category (tested in `tests/test_model.py`).
5. *Why is the model file tied to one scikit-learn version?* A saved model is a pickle of that library's
   internals. `load_bundle` refuses to load a mismatched version instead of silently misbehaving.
