"""Build data/sample_nslkdd.csv: a random 3,000-row slice of KDDTest+ with column headers.

The model is trained only on KDDTrain+, so this sample is genuinely unseen data. The
`label` column is kept so the app can show benchmark metrics, but the model never sees it.
"""
from pathlib import Path

from src.train import read_nslkdd

ROOT = Path(__file__).resolve().parent.parent
test = read_nslkdd(ROOT / "data/raw/KDDTest+.txt")
sample = test.sample(n=3000, random_state=42).reset_index(drop=True)
sample.to_csv(ROOT / "data/sample_nslkdd.csv", index=False)
print(f"Wrote {len(sample)} rows; attack share {(sample.label != 'normal').mean():.1%}")
