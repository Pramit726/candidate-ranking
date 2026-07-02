# Data directory

## Files included (needed to run + test the pipeline)

| File | Purpose |
|------|---------|
| `test_set.jsonl` | 120 full candidate profiles — the pipeline input |
| `golden_test.jsonl` | 120 golden labels for the test split |
| `golden_train.jsonl`, `golden_validation.jsonl`, `golden_set.jsonl` | Golden labels for other splits |
| `train_set.jsonl`, `validation_set.jsonl` | Other candidate splits |
| `candidates_sample_120.jsonl` | 120-candidate sample pool |
| `job_description.txt` | Senior AI Engineer JD (plain text) |
| `final_ranking*.csv` | Example pipeline outputs |

## Files NOT included (too large for the repo — regenerate or download separately)

The full raw candidate pools are multi-hundred-MB and are **not** required to run
the ranking pipeline or the tests (which operate on `test_set.jsonl`). If you
need them, place them here with these exact names:

- `candidates.jsonl` — full raw candidate pool (~487 MB)
- `candidates_excluding_sample.jsonl` — pool minus the sample (~463 MB)
- `final_candidates_12yoe.jsonl`, `final_candidates_15yoe.jsonl` — derived pools

These are git-ignored by size in practice; keep them out of version control.
