Project: Anomaly Detection (LRCN + YOLO + SORT)

This repository contains an older project that performs person detection (YOLO), tracking (SORT), and sequence classification using an LRCN-style model (CNN per-frame features + LSTM over time).

Quick notes:
- Place your LRCN `.h5` model and any other large models under `models/` (this folder is ignored by git via `.gitignore`).
- Avoid hardcoded absolute paths; use the provided `configs/config.yaml` or pass arguments to `src/inference.py`.

Getting started (example):

1. Create a virtualenv and install dependencies from `requirements.txt`.
2. Place your LRCN model in `models/` (e.g. `models/LRCN_model.h5`).
3. Run inference:

```powershell
python src\inference.py --config configs\config.yaml --video data\input.mp4 --model models\LRCN_model.h5 --output outputs\result.mp4
```

Notes about repository hygiene:
- The existing model file at the repo root is currently tracked; if you move it into `models/` you should untrack the old file with:

```powershell
git rm --cached "LRCN_model__ano-4-_Date_Time_2024_01_03__15.h5"
git add models/
git commit -m "Move model to models/ and ignore binaries"
```
