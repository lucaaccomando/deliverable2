# Isolation Forest: what it is and why it fits ThreatLens

## The idea in one paragraph

Most anomaly detectors first model what "normal" looks like and then measure distance from it. Isolation
Forest (Liu, Ting & Zhou, *Isolation Forest*, IEEE ICDM 2008) turns this around: **anomalies are few and
different, so they are easy to isolate.** Imagine splitting a table of connections with random yes/no
questions ("is `count` above 137?"). A weird connection ends up alone after only a few questions. A typical
connection sits in a crowd of similar ones and needs many questions before it is alone.

## How it works

1. **Build many random trees (200 in ThreatLens).** Each tree gets a small random sample of the training
   rows (256 by default).
2. **Grow each tree with random splits.** At every node, pick one feature at random, then pick a split value
   uniformly at random between that feature's minimum and maximum in the node. Repeat until every row is
   alone or a depth limit is hit. No labels are used anywhere.
3. **Score a connection.** Drop it down every tree and record the *path length*: the number of splits needed
   to isolate it. Average over the trees. Short average path = isolated quickly = anomalous.
4. **Normalize.** The paper converts the average path length `E[h(x)]` to `s = 2^(-E[h(x)] / c(n))`, where
   `c(n)` is the average path length of an unsuccessful search in a binary search tree of `n` points. `s`
   is near 1 for anomalies and near 0.5 or lower for normal points. scikit-learn's `score_samples` returns
   `-s`, so ThreatLens negates it again to get "higher = more anomalous".

```
 normal connection (crowded region)          unusual connection (sparse region)
 split 1 ── still with many neighbours       split 1 ── still with neighbours
 split 2 ── still with many neighbours       split 2 ── ALONE  (path length 2)
 ...                                         
 split 11 ── alone  (path length 11)         averaged over 200 trees: short path => high score
```

## Why it is a good fit for ThreatLens

| Project need | Why Isolation Forest meets it |
|---|---|
| **No labeled attacks in real deployments.** Analysts have mountains of traffic and almost no labels. | It is unsupervised: it trains on traffic assumed normal and never sees an attack label. |
| **New attack types appear constantly.** | It does not learn what specific attacks look like, only what normal looks like, so it can flag attacks it has never seen. |
| **Need a ranked list to triage.** | It outputs a continuous score, so connections can be ranked and the analyst can choose a threshold. |
| **41 mixed features, large files.** | Training and scoring cost grows roughly linearly with the data, and each tree uses only a small sample (256 rows). Scoring 22,544 connections takes about 0.2 s. |
| **Features on very different scales** (bytes vs rates vs 0/1 flags). | Splits are made per feature at a random point of its own range, so no distance metric is involved and linear rescaling changes nothing. |
| **Small, explainable, saveable model.** | The model is 200 small trees (about 0.5 MB with the calibration data), loaded in milliseconds. |

## Where it is a poor fit (and how ThreatLens deals with it)

- **It cannot explain a single prediction.** ThreatLens adds a separate z-score explanation and labels it
  an approximation.
- **Attacks that look like normal traffic are missed.** In my earlier Adaptive Threat Hunt study, a
  normal-only Isolation Forest recalled only 17% of held-out R2L attacks and 24% of U2R attacks at a 5%
  false-alarm target, because those attacks (stolen passwords, privilege escalation) look like ordinary
  sessions. ThreatLens is therefore a *triage aid*, not a complete detector.
- **Heavy-tailed features distort random splits.** A uniform split over `[0, 1e9]` bytes almost never
  separates two ordinary byte counts, so ThreatLens applies `log1p` first (see the ablation in the README).
- **The score is not a probability.** ThreatLens converts it to a percentile of normal traffic so the number
  has a concrete meaning.
- **It assumes anomalies are rare.** If attacks were the majority of traffic they would not be isolated
  quickly. NSL-KDD's test file is 57% attacks, which is unrealistic; real precision would be lower.

## Settings used

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 200 | Twice the scikit-learn default (100): averaging over more trees makes scores less random. Not tuned; I did not test other values. |
| `max_samples` | default (256) | The paper's recommendation; also keeps trees small. |
| `contamination` | default (`"auto"`) | Not used for decisions. The threshold comes from the 0-100 percentile score and the analyst's slider. |
| `random_state` | 42 | Reproducible training. |
