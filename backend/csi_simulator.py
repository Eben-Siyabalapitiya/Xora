"""
csi_simulator.py
Simulates 3 ESP32 receiver boards sending CSI data over UDP.
Run this while app.py is running to test the dashboard without hardware.

Usage:
    python csi_simulator.py
"""

import socket
import json
import numpy as np
import time
import math

UDP_IP   = '127.0.0.1'
UDP_PORT = 5005
NUM_SUB  = 52       # number of CSI subcarriers

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Simulated person position
px, py = 0.5, 0.5
tx, ty = 0.5, 0.5

# Board positions in room (0-1 coords)
BOARDS = {
    'B': (0.97, 0.03),
    'C': (0.03, 0.97),
    'D': (0.97, 0.97),
}

# Base CSI signal for each board (stable when nobody moving)
base_csi = {bid: np.random.uniform(20, 40, NUM_SUB) for bid in BOARDS}

move_timer  = 0
auto_timer  = 0
targets = [
    (0.26, 0.17), (0.74, 0.14), (0.50, 0.50),
    (0.17, 0.81), (0.70, 0.60), (0.30, 0.40),
]
t_idx = 0

def lerp(a, b, t):
    return a + (b - a) * t

def next_target():
    global t_idx, tx, ty, move_timer
    t_idx   = (t_idx + 1) % len(targets)
    tx      = max(0.1, min(0.9, targets[t_idx][0] + np.random.uniform(-0.1, 0.1)))
    ty      = max(0.1, min(0.9, targets[t_idx][1] + np.random.uniform(-0.1, 0.1)))
    move_timer = 120

def build_csi(bid, person_x, person_y, moving):
    """
    Build a realistic CSI amplitude array.
    Closer the person is to the opposite wall from this board = bigger disruption.
    """
    bx, by  = BOARDS[bid]
    base    = base_csi[bid].copy()

    if moving:
        # Distance from person to this board
        dist = math.hypot(person_x - bx, person_y - by)
        # Closer person = more disruption
        disruption = max(0, 1 - dist) * np.random.uniform(8, 18)
        noise      = np.random.normal(0, disruption, NUM_SUB)
        csi        = base + noise
    else:
        # Tiny ambient noise when still
        csi = base + np.random.normal(0, 0.4, NUM_SUB)

    return np.round(np.abs(csi), 2).tolist()

def send_packet(bid, csi):
    pkt = json.dumps({'id': bid, 'csi': csi}).encode('utf-8')
    sock.sendto(pkt, (UDP_IP, UDP_PORT))

print('[simulator] running — sending fake CSI to localhost:5005')
print('[simulator] press Ctrl+C to stop\n')

frame = 0
next_target()

try:
    while True:
        frame += 1

        # Auto move person every ~5 seconds
        auto_timer += 1
        if auto_timer > 165:
            auto_timer = 0
            next_target()

        # Move person toward target
        px = lerp(px, tx, 0.03)
        py = lerp(py, ty, 0.03)

        moving = move_timer > 0
        if move_timer > 0:
            move_timer -= 1

        # Send CSI from all 3 receiver boards
        for bid in BOARDS:
            csi = build_csi(bid, px, py, moving)
            send_packet(bid, csi)

        if frame % 100 == 0:
            print(f'[simulator] frame {frame} | pos ({px:.2f}, {py:.2f}) | moving: {moving}')

        # ~100 packets per second
        time.sleep(0.01)

except KeyboardInterrupt:
    print('\n[simulator] stopped')
    sock.close()