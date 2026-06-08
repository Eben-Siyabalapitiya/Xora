import threading
import socket
import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import json
import time
import os

app = Flask(__name__, template_folder='../frontend')
app.config['SECRET_KEY'] = 'xora-secret'
socketio = SocketIO(app, cors_allowed_origins='*')

# ─── Config ───────────────────────────────────────────────
UDP_PORT     = 5005
BOARD_IDS    = ['B', 'C', 'D']   # receivers
NUM_SUB      = 52                 # CSI subcarriers
SMOOTH       = 0.03               # lerp smoothing for orb
THRESHOLD    = 3.0                # motion detection variance threshold

# ─── Shared state ─────────────────────────────────────────
state = {
    'px': 0.5, 'py': 0.5,        # current position (0-1)
    'tx': 0.5, 'ty': 0.5,        # target position
    'variance': 0.0,
    'motion': False,
    'confidence': 0.0,
    'zone': 'unknown',
    'events': 0,
}

# Latest raw CSI from each board
csi_buffers = {bid: [] for bid in BOARD_IDS}
baseline    = {bid: None for bid in BOARD_IDS}
model       = None                # loaded after training

# ─── Zones (match your room layout) ───────────────────────
ZONES = [
    {'name': 'desk',   'x': 0.78, 'y': 0.22},
    {'name': 'bed',    'x': 0.22, 'y': 0.75},
    {'name': 'door',   'x': 0.50, 'y': 0.06},
    {'name': 'center', 'x': 0.50, 'y': 0.50},
]

def get_zone(px, py):
    best, best_d = 'unknown', 999
    for z in ZONES:
        d = np.hypot(px - z['x'], py - z['y'])
        if d < best_d:
            best_d = d
            best = z['name']
    return best

# ─── UDP listener (runs in background thread) ──────────────
def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', UDP_PORT))
    print(f'[xora] UDP listening on port {UDP_PORT}')

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            parse_packet(data)
        except Exception as e:
            print(f'[udp] error: {e}')

def parse_packet(raw):
    """
    Expected packet format from ESP32 receiver (JSON):
    {"id": "B", "csi": [32, 14, 9, 41, ...]}
    """
    try:
        pkt    = json.loads(raw.decode('utf-8'))
        bid    = pkt.get('id')
        csi    = np.array(pkt.get('csi', []), dtype=float)

        if bid not in BOARD_IDS or len(csi) == 0:
            return

        # Subtract baseline if calibrated
        if baseline[bid] is not None:
            csi = np.abs(csi - baseline[bid])

        # Keep rolling buffer of last 100 frames per board
        csi_buffers[bid].append(csi)
        if len(csi_buffers[bid]) > 100:
            csi_buffers[bid].pop(0)

        process_csi()

    except Exception as e:
        print(f'[parse] error: {e}')

def process_csi():
    """Compute variance per board → estimate position → emit to dashboard."""
    variances = {}
    for bid in BOARD_IDS:
        buf = csi_buffers[bid]
        if len(buf) < 10:
            continue
        arr = np.array(buf[-10:])          # last 10 frames
        variances[bid] = float(np.var(arr))

    if len(variances) < 2:
        return

    total_var = sum(variances.values())
    state['variance'] = round(total_var, 2)
    state['motion']   = total_var > THRESHOLD

    if state['motion']:
        state['events'] += 1

        # Simple weighted position estimate
        # Board layout: B=top-right, C=bottom-left, D=bottom-right
        board_positions = {
            'B': (0.97, 0.03),
            'C': (0.03, 0.97),
            'D': (0.97, 0.97),
        }
        wx, wy, wt = 0, 0, 0
        for bid, var in variances.items():
            if bid in board_positions:
                bx, by = board_positions[bid]
                # Higher variance = person closer to opposite side
                wx += (1 - bx) * var
                wy += (1 - by) * var
                wt += var

        if wt > 0:
            state['tx'] = wx / wt
            state['ty'] = wy / wt

        state['confidence'] = min(1.0, total_var / 15.0)

    # Lerp current position toward target
    state['px'] = lerp(state['px'], state['tx'], SMOOTH)
    state['py'] = lerp(state['py'], state['ty'], SMOOTH)
    state['zone'] = get_zone(state['px'], state['py'])

    # Push to dashboard
    socketio.emit('state', {
        'px':        round(state['px'], 3),
        'py':        round(state['py'], 3),
        'variance':  state['variance'],
        'motion':    state['motion'],
        'confidence': round(state['confidence'], 2),
        'zone':      state['zone'],
        'events':    state['events'],
    })

def lerp(a, b, t):
    return a + (b - a) * t

# ─── Calibration endpoint ──────────────────────────────────
@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Call this with empty room.
    Records 30 seconds of CSI and saves mean as baseline.
    """
    print('[xora] calibrating — keep room empty...')
    samples = {bid: [] for bid in BOARD_IDS}
    start = time.time()

    while time.time() - start < 30:
        for bid in BOARD_IDS:
            if csi_buffers[bid]:
                samples[bid].append(csi_buffers[bid][-1])
        time.sleep(0.1)

    for bid in BOARD_IDS:
        if samples[bid]:
            baseline[bid] = np.mean(samples[bid], axis=0)
            print(f'[calibrate] board {bid} baseline saved ({len(samples[bid])} samples)')

    # Save to disk
    np.save('baseline.npy', baseline)
    return jsonify({'status': 'calibrated'})

@app.route('/calibrate/load', methods=['POST'])
def load_calibration():
    global baseline
    if os.path.exists('baseline.npy'):
        baseline = np.load('baseline.npy', allow_pickle=True).item()
        print('[xora] baseline loaded from disk')
        return jsonify({'status': 'loaded'})
    return jsonify({'status': 'no baseline found'}), 404

# ─── Routes ───────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({
        'boards_connected': [b for b in BOARD_IDS if len(csi_buffers[b]) > 0],
        'calibrated': all(v is not None for v in baseline.values()),
        'motion': state['motion'],
        'zone': state['zone'],
    })

# ─── Start ────────────────────────────────────────────────
if __name__ == '__main__':
    # Load baseline if it exists
    if os.path.exists('baseline.npy'):
        baseline = np.load('baseline.npy', allow_pickle=True).item()
        print('[xora] baseline loaded')

    # Start UDP listener in background
    t = threading.Thread(target=udp_listener, daemon=True)
    t.start()

    print('[xora] starting server → http://localhost:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)