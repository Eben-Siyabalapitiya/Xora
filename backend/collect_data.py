"""
collect_data.py — Xora position training data collector
Records CSI snapshots labeled with x,y coordinates for regression training.
"""

import os
import time
import numpy as np

csi_buffers = None
NUM_SUB = 52
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")

# Zone definitions — name, x, y (0-1 normalised room coordinates)
ZONES = [
    {"name": "bed",      "x": 0.10, "y": 0.35},
    {"name": "table",    "x": 0.50, "y": 0.15},
    {"name": "cabinet",  "x": 0.88, "y": 0.15},
    {"name": "closet",   "x": 0.88, "y": 0.60},
    {"name": "s_table",  "x": 0.25, "y": 0.70},
    {"name": "door",     "x": 0.50, "y": 0.92},
    {"name": "center",   "x": 0.50, "y": 0.50},
]

SAMPLES_PER_ZONE = 500


def _ensure_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def collect_zone(zone_name: str, x: float, y: float, target: int = SAMPLES_PER_ZONE, status_cb=None):
    """
    Record CSI samples for a specific position (x, y).
    Saves to training_data/pos_<zone_name>.npy
    Each row: [csi_B(52), csi_C(52), csi_D(52), x, y]
    """
    if csi_buffers is None:
        raise RuntimeError("csi_buffers not set")

    _ensure_dir()
    rows = []
    last_snapshot = {"B": None, "C": None, "D": None}
    count = 0

    print(f"[collect] recording zone '{zone_name}' at ({x:.2f}, {y:.2f}) — need {target} samples")

    while count < target:
        new_data = False
        csi_row = []
        for bid in ("B", "C", "D"):
            buf = csi_buffers.get(bid)
            if not buf:
                csi_row.extend([0.0] * NUM_SUB)
                continue
            latest = buf[-1]
            if last_snapshot[bid] is None or not np.array_equal(latest, last_snapshot[bid]):
                last_snapshot[bid] = latest.copy()
                new_data = True
            csi_row.extend(latest.tolist())

        if new_data and len(csi_row) == NUM_SUB * 3:
            rows.append(csi_row + [x, y])
            count += 1
            if count % 100 == 0:
                print(f"[collect] {count} / {target}")
                if status_cb:
                    status_cb(count, target)
        else:
            time.sleep(0.005)

    arr = np.array(rows, dtype=np.float32)
    path = os.path.join(SAVE_DIR, f"pos_{zone_name}.npy")
    np.save(path, arr)
    print(f"[collect] saved {count} samples → {path}")
    if status_cb:
        status_cb(target, target)
    return path


def collect(label: str, target: int, status_cb=None):
    """Legacy classifier collection — kept for empty/outside recording."""
    if csi_buffers is None:
        raise RuntimeError("csi_buffers not set")

    _ensure_dir()
    rows = []
    last_snapshot = {"B": None, "C": None, "D": None}
    count = 0

    print(f"[collect] recording '{label}' — need {target} samples")

    while count < target:
        new_data = False
        csi_row = []
        for bid in ("B", "C", "D"):
            buf = csi_buffers.get(bid)
            if not buf:
                csi_row.extend([0.0] * NUM_SUB)
                continue
            latest = buf[-1]
            if last_snapshot[bid] is None or not np.array_equal(latest, last_snapshot[bid]):
                last_snapshot[bid] = latest.copy()
                new_data = True
            csi_row.extend(latest.tolist())

        if new_data and len(csi_row) == NUM_SUB * 3:
            rows.append(csi_row)
            count += 1
            if count % 100 == 0:
                print(f"[collect] {count} / {target}")
                if status_cb:
                    status_cb(count, target)
        else:
            time.sleep(0.005)

    arr = np.array(rows, dtype=np.float32)
    path = os.path.join(SAVE_DIR, f"{label}.npy")
    np.save(path, arr)
    print(f"[collect] saved → {path}")
    if status_cb:
        status_cb(target, target)
    return path

LABEL_SAMPLES = {"empty": 3000, "motion1": 5000, "motion2": 5000, "outside": 3000}