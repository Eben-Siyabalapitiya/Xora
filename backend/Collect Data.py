"""
collect_data.py
Records labeled CSI data for training the TensorFlow model.
Called automatically by the dashboard training tab — you don't run this manually.

Labels:
    empty    — nobody in room
    motion1  — one person moving
    motion2  — two people moving
    outside  — movement outside room
"""

import socket
import json
import numpy as np
import os
import time

UDP_PORT = 5005
NUM_SUB  = 52
DATA_DIR = 'training_data'

os.makedirs(DATA_DIR, exist_ok=True)

BOARD_IDS = ['B', 'C', 'D']

def collect(label, num_samples, on_progress=None):
    """
    Collect num_samples CSI windows and save with label.
    on_progress(current, total) called every 100 samples.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', UDP_PORT))
    sock.settimeout(1.0)

    buffers  = {bid: [] for bid in BOARD_IDS}
    samples  = []
    WINDOW   = 50   # frames per sample window

    print(f'[collect] recording "{label}" — need {num_samples} samples...')

    while len(samples) < num_samples:
        try:
            data, _ = sock.recvfrom(4096)
            pkt     = json.loads(data.decode())
            bid     = pkt.get('id')
            csi     = np.array(pkt.get('csi', []), dtype=float)

            if bid not in BOARD_IDS or len(csi) == 0:
                continue

            buffers[bid].append(csi)

            # Once all boards have enough frames, build one sample
            if all(len(buffers[b]) >= WINDOW for b in BOARD_IDS):
                # Stack: shape (3, WINDOW, NUM_SUB)
                window = np.stack([
                    np.array(buffers[b][-WINDOW:]) for b in BOARD_IDS
                ])
                samples.append(window)

                # Clear buffers for next window
                for b in BOARD_IDS:
                    buffers[b] = buffers[b][WINDOW//2:]

                if on_progress:
                    on_progress(len(samples), num_samples)

                if len(samples) % 100 == 0:
                    print(f'[collect] {len(samples)} / {num_samples}')

        except socket.timeout:
            continue
        except Exception as e:
            print(f'[collect] error: {e}')

    sock.close()

    # Save
    X = np.array(samples)                                    # (N, 3, WINDOW, NUM_SUB)
    y = np.full(len(samples), label)
    path = os.path.join(DATA_DIR, f'{label}.npz')
    np.savez(path, X=X, y=y)
    print(f'[collect] saved {len(samples)} samples → {path}')
    return X, y


def load_all():
    """Load all collected datasets and return X, y ready for training."""
    label_map = {'empty': 0, 'motion1': 1, 'motion2': 2, 'outside': 3}
    Xs, ys = [], []

    for label, idx in label_map.items():
        path = os.path.join(DATA_DIR, f'{label}.npz')
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            Xs.append(data['X'])
            ys.append(np.full(len(data['X']), idx))
            print(f'[load] {label}: {len(data["X"])} samples')

    if not Xs:
        raise ValueError('No training data found — collect data first')

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    print(f'[load] total: {len(X)} samples, {len(set(y))} classes')
    return X, y, len(label_map)