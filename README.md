# Anomaly Detection (LRCN + YOLO + SORT)

This repository contains a pipeline that performs person detection (YOLO), tracking (SORT), and sequence classification using an LRCN-style model (CNN per-frame features + LSTM over time).

## Quick Start

- Recommended Python: 3.8 - 3.11
- Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Place your LRCN model in the `models/` folder (this folder is excluded by `.gitignore`): `models/LRCN_model.h5`.

### Dry-run (no heavy deps)
Validate the pipeline without loading YOLO or TensorFlow (useful for CI or quick smoke tests):

```powershell
python src\inference.py --dry_run --output outputs\dryrun.mp4
```

### Full inference
Run detection, tracking, and sequence classification with your model:

```powershell
python src\inference.py --config configs\config.yaml --video data\input.mp4 --model models\LRCN_model.h5 --output outputs\result.mp4
```

You can override values from `configs/config.yaml` via command-line arguments (see `src/inference.py --help`).

## Configuration

Default configuration is in `configs/config.yaml`. Key fields:

- `video`: input video path
- `model`: path to LRCN `.h5` model
- `seq_len`: the frame sequence length passed to the LRCN
- `detector_model`: optional detector weights or model name (detector init is skipped in `--dry_run`)

Adjust these either in the YAML file or pass them as CLI flags to `src/inference.py`.

## Outputs

When running inference the pipeline produces:
- `outputs/<name>.mp4` — visualization video with boxes and labels
- `outputs/<name>_tracks.json` — per-frame/per-track JSON records
- `outputs/<name>_tracks.csv` — per-frame/per-track CSV (same data in tabular form)

Create an `outputs/` directory if it does not exist; the runner will attempt to write into it.

## Models & Git

- Keep large binary files out of Git history. Add models to `.gitignore` (already present).
- If you accidentally committed a model file, untrack it without deleting the local copy:

```powershell
git rm --cached "LRCN_model__ano-4-_Date_Time_2024_01_03__15.h5"
git add models/
git commit -m "Move model to models/ and ignore binaries"
```

- If you intend to version large model files, consider using Git LFS:

```powershell
git lfs install
git lfs track "models/*.h5"
git add .gitattributes
git add models/*.h5
git commit -m "Track models with Git LFS"
```

## Troubleshooting

- Ultralytics/YOLO may auto-download weights on first run. Use `--dry_run` for smoke tests to avoid downloads.
- If TensorFlow/Keras is not installed, LRCN model loading will fail; install a compatible `tensorflow` package for your platform.
- On Windows, long paths or CRLF/LF differences can trigger warnings; these are usually safe but watch for line-ending warnings when committing notebooks.

## Notes on Licensing

- `src/sort.py` contains code under the original SORT license (GPL-style header). Ensure license compatibility if you redistribute.

## Contributing

Contributions are welcome. Open issues or PRs; for major changes, please describe the intent and provide a minimal reproducible example.

## Contact / Next Steps

- To run a full end-to-end test, place a sample video in `data/` and run the full inference command above.
- If you'd like, I can update this README further with an example `configs/config.yaml` snippet or add a CI workflow for smoke tests.
