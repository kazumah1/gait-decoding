# Data Reencoding

This folder contains a single repo-specific conversion script.

It assumes:

- the raw source data is in `../data/RepositoryData`
- the output should be written to `../data/gait_decoding_combined.csv`

## Run

From this directory:

```bash
uv run main.py
```

`uv run` by itself does not execute a project entrypoint, so `uv run main.py` is the shortest supported form.

## Output

The script writes one CSV row per synchronized timestamp with:

- `subject_id`
- `trial_id`
- `time_seconds`
- 64 EEG/EOG values in the order from `impedances-before.txt`
- 12 joint values from `joints.txt`

The script validates:

- all trial folders match the expected `SLxx-Tyy` pattern
- all channel labels match the canonical ordering
- all joint labels match the canonical ordering
- every `eeg.txt` timestamp exactly matches the corresponding `joints.txt` timestamp
