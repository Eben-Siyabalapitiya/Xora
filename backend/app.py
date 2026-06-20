import threading
import socket
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import time
import os

app = Flask(__name__, template_folder='../frontend')
app.config['SECRET_KEY'] = 'xora-secret'
socketio = SocketIO(app, cors_allowed_origins='*')

UDP_PORT  = 5005
BOARD_IDS = ['B', 'C', 'D']
NUM_SUB   = 52
SMOOTH    = 0.06
THRESHOLD = 3.0

state = {
    'px': 0.5, 'py': 0.5,
    'tx': 0.5, 'ty': 0.5,
    'variance': 0.0,
    'motion': False,
    'confidence': 0.0,
    'zone': 'unknown',
    'events': 0,
}

csi_buffers = {bid: [] for bid in BOARD_IDS}
baseline    = {bid: None for bid in BOARD_IDS}
last_seen   = {bid: 0    for bid in BOARD_IDS}
model       = None
model_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xora_model.h5')

ZONES = [
    {'name': 'bed',     'x': 0.10, 'y': 0.35},
    {'name': 'table',   'x': 0.50, 'y': 0.15},
    {'name': 'cabinet', 'x': 0.88, 'y': 0.15},
    {'name': 'closet',  'x': 0.88, 'y': 0.60},
    {'name': 's_table', 'x': 0.25, 'y': 0.70},
    {'name': 'door',    'x': 0.50, 'y': 0.92},
    {'name': 'center',  'x': 0.50, 'y': 0.50},
]

def get_zone(px, py):
    best, best_d = 'unknown', 999
    for z in ZONES:
        d = np.hypot(px - z['x'], py - z['y'])
        if d < best_d:
            best_d = d
            best = z['name']
    return best

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', UDP_PORT))
    print(f'[xora] UDP listening on port {UDP_PORT}')
    while True:
        try:
            data, _ = sock.recvfrom(4096)
            parse_packet(data)
        except Exception as e:
            print(f'[udp] error: {e}')

def parse_packet(raw):
    decoded = raw.decode('utf-8', errors='ignore').strip()
    if decoded.startswith('XORA_TX') or not decoded.startswith('XORA_CSI'):
        return
    try:
        parts = decoded.split(',')
        if len(parts) < 4:
            return
        bid = parts[1]
        if bid not in BOARD_IDS:
            return
        csi = np.array([float(x) for x in parts[3:]], dtype=float)
        if len(csi) == 0:
            return
        if len(csi) >= NUM_SUB:
            csi = csi[:NUM_SUB]
        else:
            csi = np.pad(csi, (0, NUM_SUB - len(csi)))
        last_seen[bid] = time.time()
        if baseline[bid] is not None:
            csi = np.abs(csi - baseline[bid])
        csi_buffers[bid].append(csi)
        if len(csi_buffers[bid]) > 100:
            csi_buffers[bid].pop(0)
        process_csi()
    except Exception as e:
        print(f'[parse] error: {e}')

def process_csi():
    variances = {}
    for bid in BOARD_IDS:
        buf = csi_buffers[bid]
        if len(buf) < 10:
            continue
        arr = np.array(buf[-10:])
        variances[bid] = float(np.var(arr))

    if len(variances) < 2:
        return

    total_var = sum(variances.values())
    state['variance'] = round(total_var, 2)
    state['motion']   = total_var > THRESHOLD

    if state['motion']:
        state['events'] += 1

        # Use trained model for position if available
        if model is not None:
            try:
                csi_row = []
                for bid in ['B', 'C', 'D']:
                    if csi_buffers[bid]:
                        csi_row.extend(csi_buffers[bid][-1].tolist())
                    else:
                        csi_row.extend([0.0] * NUM_SUB)
                import train as train_module
                tx, ty = train_module.predict_position(model, np.array(csi_row, dtype=np.float32), model_path)
                state['tx'] = float(np.clip(tx, 0.0, 1.0))
                state['ty'] = float(np.clip(ty, 0.0, 1.0))
                state['confidence'] = min(1.0, total_var / 15.0)
            except Exception as e:
                print(f'[predict] error: {e}')
                _variance_position(variances)
        else:
            _variance_position(variances)

    state['px'] = lerp(state['px'], state['tx'], SMOOTH)
    state['py'] = lerp(state['py'], state['ty'], SMOOTH)
    state['zone'] = get_zone(state['px'], state['py'])

    now = time.time()
    boards_active = [bid for bid in BOARD_IDS if now - last_seen[bid] < 3]
    socketio.emit('state', {
        'px':          round(state['px'], 3),
        'py':          round(state['py'], 3),
        'variance':    state['variance'],
        'motion':      state['motion'],
        'confidence':  round(state['confidence'], 2),
        'zone':        state['zone'],
        'events':      state['events'],
        'boards_active': boards_active,
    })

def _variance_position(variances):
    board_positions = {'B': (0.97, 0.03), 'C': (0.03, 0.97), 'D': (0.97, 0.97)}
    wx, wy, wt = 0, 0, 0
    for bid, var in variances.items():
        if bid in board_positions:
            bx, by = board_positions[bid]
            wx += (1 - bx) * var
            wy += (1 - by) * var
            wt += var
    if wt > 0:
        state['tx'] = wx / wt
        state['ty'] = wy / wt
    state['confidence'] = 0.0

def lerp(a, b, t):
    return a + (b - a) * t

def heartbeat_loop():
    time.sleep(3)
    while True:
        try:
            now = time.time()
            boards_active = [bid for bid in BOARD_IDS if now - last_seen[bid] < 3]
            socketio.emit('state', {
                'px':          round(state['px'], 3),
                'py':          round(state['py'], 3),
                'variance':    state['variance'],
                'motion':      state['motion'],
                'confidence':  round(state['confidence'], 2),
                'zone':        state['zone'],
                'events':      state['events'],
                'boards_active': boards_active,
            })
        except Exception as e:
            print(f'[heartbeat] error: {e}')
        time.sleep(1)

@app.route('/calibrate', methods=['POST'])
def calibrate():
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
    np.save('baseline.npy', baseline)
    return jsonify({'status': 'calibrated'})

import collect_data
collect_data.csi_buffers = csi_buffers

try:
    import train as train_module
    print('[xora] train module loaded')
except Exception as e:
    print(f'[xora] train import failed: {e}')
    train_module = None

@app.route('/train/record_zone', methods=['POST'])
def train_record_zone():
    data      = request.get_json()
    zone_name = data.get('zone')
    x         = float(data.get('x', 0.5))
    y         = float(data.get('y', 0.5))
    target    = int(data.get('samples', collect_data.SAMPLES_PER_ZONE))

    def run():
        try:
            collect_data.collect_zone(zone_name, x, y, target)
            print(f'[train] finished zone "{zone_name}"')
        except Exception as e:
            print(f'[train] zone error: {e}')

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'status': 'started', 'zone': zone_name})

@app.route('/train/record', methods=['POST'])
def train_record():
    data    = request.get_json()
    label   = data.get('label')
    samples = data.get('samples', 3000)

    def run():
        try:
            print(f'[train] starting collection for "{label}" ({samples} samples)')
            collect_data.collect(label, samples)
            print(f'[train] finished collecting "{label}"')
        except Exception as e:
            print(f'[train] collection error: {e}')

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'status': 'recording started', 'label': label})

@app.route('/train/run', methods=['POST'])
def train_run():
    global model
    try:
        print('[train] starting position model training...')
        accuracy = train_module.train_and_save()
        model = train_module.load_model(model_path)
        print(f'[train] training complete — score: {accuracy:.4f}')
        return jsonify({'status': 'trained', 'accuracy': float(accuracy)})
    except Exception as e:
        print(f'[train] error: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({
        'boards_connected': [b for b in BOARD_IDS if len(csi_buffers[b]) > 0],
        'model_loaded': model is not None,
        'motion': state['motion'],
        'zone': state['zone'],
    })

if __name__ == '__main__':
    if os.path.exists('baseline.npy'):
        baseline.update(np.load('baseline.npy', allow_pickle=True).item())
        print('[xora] baseline loaded')

    if os.path.exists(model_path) and train_module is not None:
        try:
            model = train_module.load_model(model_path)
            print('[xora] model loaded')
        except Exception as e:
            print(f'[xora] model load failed: {e}')

    threading.Thread(target=udp_listener, daemon=True).start()
    threading.Thread(target=heartbeat_loop, daemon=True).start()

    print('[xora] starting server → http://localhost:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)