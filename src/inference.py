"""Inference script for detection + tracking + sequence classification.
Usage: python src/inference.py --config configs/config.yaml --model models/LRCN_model.h5 --video data/input.mp4
"""
import argparse
import os
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

from sort import Sort


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    p.add_argument('--model', default=None, help='Path to LRCN model (.h5)')
    p.add_argument('--video', default=None, help='Input video path')
    p.add_argument('--output', default=None, help='Output video path')
    p.add_argument('--seq_len', type=int, default=None)
    p.add_argument('--conf', type=float, default=None)
    p.add_argument('--dry_run', action='store_true', help='Run a lightweight dry-run without detector/LRCN')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config) if Path(args.config).exists() else {}

    model_path = args.model or cfg.get('model', {}).get('lrcn')
    detector_weights = cfg.get('model', {}).get('detector')
    video_path = args.video or cfg.get('video', {}).get('input')
    output_path = args.output or cfg.get('video', {}).get('output', 'outputs/output.mp4')
    seq_len = args.seq_len or cfg.get('inference', {}).get('seq_len', 30)
    conf_thr = args.conf or cfg.get('inference', {}).get('conf_threshold', 0.3)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load detector only for real inference. Dry-run must not trigger downloads.
    detector = None
    if not args.dry_run:
        if YOLO is None:
            raise RuntimeError('ultralytics YOLO is not available in this environment')
        detector = YOLO(detector_weights) if detector_weights else None

    # Load LRCN if available
    lrcn = None
    if model_path and Path(model_path).exists():
        if load_model is None:
            print('TensorFlow not available; skipping LRCN loading')
        else:
            lrcn = load_model(model_path, compile=False)

    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # Dry-run mode: synthesize frames and simulated detections to validate pipeline
    if args.dry_run:
        print('Running dry-run: no detector or LRCN will be used. Generating synthetic frames.')
        cap = None
        frame_w, frame_h = 640, 480
        fps = 10
        total_frames = 100
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video_path}')
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS) or 10))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    to_model = {}
    preds = {}
    frame_idx = 0
    records = []

    # Main loop: either process video frames or synthesize frames in dry-run
    while True:
        frame_idx += 1
        if args.dry_run:
            if frame_idx > total_frames:
                break
            # synthesize a blank frame and a moving box
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            # moving box parameters
            cx = 50 + (frame_idx * 5) % (frame_w - 100)
            cy = 50 + (frame_idx * 2) % (frame_h - 100)
            x1, y1, x2, y2 = cx, cy, cx + 80, cy + 140
            detections = np.array([[x1, y1, x2, y2, 0.9]])
        else:
            ret, frame = cap.read()
            if not ret:
                break
            detections = np.empty((0, 5))

            results = detector(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    # only persons
                    if cls == 0 and conf >= conf_thr:
                        row = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, row))

        tracks = tracker.update(detections)

        # accumulate crops per-track id
        for t in tracks:
            x1, y1, x2, y2, pid = t
            x1, y1, x2, y2, pid = int(x1), int(y1), int(x2), int(y2), int(pid)
            crop = cv2.resize(frame[y1:y2, x1:x2], (64, 64)) if (y2>y1 and x2>x1) else None
            if crop is None:
                continue
            if pid not in to_model:
                to_model[pid] = []
            to_model[pid].append(crop)

            # when enough frames collected, run LRCN
            if lrcn is not None and len(to_model[pid]) >= seq_len:
                seq = np.array(to_model[pid][:seq_len])
                seq = seq.astype('float32') / 255.0
                seq = np.expand_dims(seq, axis=0)
                try:
                    probs = lrcn.predict(seq)[0]
                    label = int(np.argmax(probs))
                    preds[pid] = label
                except Exception as e:
                    print('LRCN predict failed:', e)
                # drop used frames
                to_model[pid] = to_model[pid][seq_len:]

        # draw
        for t in tracks:
            x1, y1, x2, y2, pid = t
            x1, y1, x2, y2, pid = int(x1), int(y1), int(x2), int(y2), int(pid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = preds.get(pid, 'N/A')
            cv2.putText(frame, f'ID:{pid} ACT:{label}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # record for outputs
            records.append({
                'frame': frame_idx,
                'id': int(pid),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'label': label
            })

        out.write(frame)

    if cap is not None:
        cap.release()
    out.release()
    # write JSON and CSV outputs
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    json_out = os.path.splitext(output_path)[0] + '_tracks.json'
    csv_out = os.path.splitext(output_path)[0] + '_tracks.csv'
    try:
        with open(json_out, 'w') as f:
            json.dump(records, f, indent=2)
        # write CSV
        import csv
        with open(csv_out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame','id','x1','y1','x2','y2','label'])
            for r in records:
                x1, y1, x2, y2 = r['bbox']
                writer.writerow([r['frame'], r['id'], x1, y1, x2, y2, r['label']])
    except Exception as e:
        print('Failed to write outputs:', e)
    print('Done. Output saved to', output_path)


if __name__ == '__main__':
    main()
