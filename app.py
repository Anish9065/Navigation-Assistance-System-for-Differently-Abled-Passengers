"""
Navigation Assistance System for Differently-Abled Passengers
Main Flask Application
"""

import os, cv2, json, time, threading, shutil
from flask import (Flask, render_template, request, Response,
                   jsonify, send_file, redirect, url_for)
from werkzeug.utils import secure_filename

from config.settings import Settings
from utils.detector import Detector
from utils.visualization import Visualizer
from navigation.navigator import Navigator
from navigation.audio_guide import AudioGuide

# ── App Setup ────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024   # 500 MB
app.secret_key = Settings.SECRET_KEY

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('runs', exist_ok=True)

ALLOWED_IMAGES = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_VIDEOS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# ── Global State ──────────────────────────────────────────────────
detector    = Detector()
visualizer  = Visualizer()
navigator   = Navigator()
audio_guide = AudioGuide()

camera_active   = False
camera_lock     = threading.Lock()
current_camera  = None
training_status = {'running': False, 'progress': 0, 'log': [], 'done': False, 'error': None}


# ── Helpers ───────────────────────────────────────────────────────
def allowed_file(filename, allowed):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

def save_upload(file, allowed_set):
    if file and allowed_file(file.filename, allowed_set):
        fname = secure_filename(file.filename)
        ts    = str(int(time.time()))
        fname = f"{ts}_{fname}"
        path  = os.path.join(UPLOAD_FOLDER, fname)
        file.save(path)
        return path
    return None


# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def index():
    models = [f for f in os.listdir('models') if f.endswith('.pt')]
    return render_template('index.html', models=models)


@app.route('/train')
def train_page():
    return render_template('train.html')


@app.route('/detect')
def detect_page():
    models = [f for f in os.listdir('models') if f.endswith('.pt')]
    return render_template('detect.html', models=models)


@app.route('/navigation')
def navigation_page():
    station_map = navigator.get_station_map()
    models = [f for f in os.listdir('models') if f.endswith('.pt')]
    return render_template('navigation.html', station_map=station_map, models=models)


# ── API: Load Model ───────────────────────────────────────────────
@app.route('/api/load_model', methods=['POST'])
def load_model():
    data       = request.json
    model_name = data.get('model', 'yolov8n.pt')
    model_path = os.path.join('models', model_name) if not model_name.startswith('yolo') else model_name
    ok, msg    = detector.load(model_path)
    return jsonify({'success': ok, 'message': msg})


# ── API: Training ─────────────────────────────────────────────────
@app.route('/api/train', methods=['POST'])
def start_training():
    global training_status
    if training_status['running']:
        return jsonify({'success': False, 'message': 'Training already running'})

    data       = request.json
    yaml_path  = data.get('yaml_path', 'config/dataset.yaml')
    epochs     = int(data.get('epochs', 50))
    imgsz      = int(data.get('imgsz', 640))
    batch      = int(data.get('batch', 16))
    model_base = data.get('model_base', 'yolov8n.pt')
    project    = data.get('project', 'runs/train')
    name       = data.get('name', 'navigation_model')

    training_status = {'running': True, 'progress': 0, 'log': [], 'done': False, 'error': None}

    def run_training():
        global training_status
        try:
            from ultralytics import YOLO
            training_status['log'].append(f"Loading base model: {model_base}")
            model = YOLO(model_base)

            training_status['log'].append(f"Starting training | epochs={epochs} | imgsz={imgsz} | batch={batch}")

            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project,
                name=name,
                exist_ok=True,
                verbose=True,
            )

            best_pt = os.path.join(project, name, 'weights', 'best.pt')
            dst     = os.path.join('models', f'{name}_best.pt')
            if os.path.exists(best_pt):
                shutil.copy(best_pt, dst)
                training_status['log'].append(f"✅ Model saved to: {dst}")

            training_status['progress'] = 100
            training_status['done']    = True
            training_status['log'].append("Training complete!")

        except Exception as e:
            training_status['error'] = str(e)
            training_status['log'].append(f"❌ Error: {e}")
        finally:
            training_status['running'] = False

    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    return jsonify({'success': True, 'message': 'Training started'})


@app.route('/api/train/status')
def training_status_api():
    return jsonify(training_status)


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    training_status['running'] = False
    return jsonify({'success': True})


# ── API: Upload Dataset YAML ──────────────────────────────────────
@app.route('/api/upload_yaml', methods=['POST'])
def upload_yaml():
    f = request.files.get('yaml')
    if f and f.filename.endswith('.yaml'):
        path = os.path.join('config', secure_filename(f.filename))
        f.save(path)
        return jsonify({'success': True, 'path': path})
    return jsonify({'success': False, 'message': 'Invalid file'})


@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    """Upload a zip of the dataset and extract into data/dataset/"""
    f = request.files.get('dataset')
    if f and f.filename.endswith('.zip'):
        zpath = os.path.join('data', 'dataset.zip')
        f.save(zpath)
        import zipfile
        with zipfile.ZipFile(zpath, 'r') as z:
            z.extractall('data/dataset')
        os.remove(zpath)
        return jsonify({'success': True, 'message': 'Dataset extracted to data/dataset/'})
    return jsonify({'success': False, 'message': 'Please upload a .zip file'})


# ── API: Detect Image ─────────────────────────────────────────────
@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    file = request.files.get('image')
    if not file:
        return jsonify({'success': False, 'message': 'No image uploaded'})

    path = save_upload(file, ALLOWED_IMAGES)
    if not path:
        return jsonify({'success': False, 'message': 'Invalid file type'})

    if not detector.is_loaded():
        return jsonify({'success': False, 'message': 'No model loaded. Please load a model first.'})

    frame    = cv2.imread(path)
    results  = detector.detect(frame)
    annotated = visualizer.draw(frame, results, navigator)

    out_name = 'result_' + os.path.basename(path)
    out_path = os.path.join(UPLOAD_FOLDER, out_name)
    cv2.imwrite(out_path, annotated)

    detections = detector.format_results(results)
    directions = navigator.get_directions_from_detections(detections, frame.shape)

    return jsonify({
        'success': True,
        'result_image': '/' + out_path.replace('\\', '/'),
        'detections': detections,
        'directions': directions,
        'count': len(detections),
    })


# ── API: Detect Video ─────────────────────────────────────────────
@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    file = request.files.get('video')
    if not file:
        return jsonify({'success': False, 'message': 'No video uploaded'})

    path = save_upload(file, ALLOWED_VIDEOS)
    if not path:
        return jsonify({'success': False, 'message': 'Invalid file type'})

    if not detector.is_loaded():
        return jsonify({'success': False, 'message': 'No model loaded.'})

    out_name = 'result_' + os.path.basename(path).rsplit('.', 1)[0] + '.mp4'
    out_path = os.path.join(UPLOAD_FOLDER, out_name)

    cap     = cv2.VideoCapture(path)
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out     = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results   = detector.detect(frame)
        annotated = visualizer.draw(frame, results, navigator)
        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()

    return jsonify({
        'success': True,
        'result_video': '/' + out_path.replace('\\', '/'),
        'total_frames': frame_idx,
    })


# ── Live Camera Stream ────────────────────────────────────────────
def generate_frames(destination='platform_1'):
    global camera_active, current_camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with camera_lock:
        camera_active  = True
        current_camera = cap

    try:
        while camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            if detector.is_loaded():
                results   = detector.detect(frame)
                annotated = visualizer.draw(frame, results, navigator, destination=destination)
                detections = detector.format_results(results)

                # Audio cue every 3 seconds
                if int(time.time()) % 3 == 0 and detections:
                    dirs = navigator.get_directions_from_detections(detections, frame.shape)
                    if dirs:
                        audio_guide.speak_async(dirs[0]['instruction'])
            else:
                annotated = frame.copy()
                cv2.putText(annotated, 'No model loaded — use /api/load_model',
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            ret2, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret2:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        with camera_lock:
            camera_active = False


@app.route('/video_feed')
def video_feed():
    destination = request.args.get('destination', 'platform_1')
    return Response(generate_frames(destination),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'success': True})


# ── API: Navigation ───────────────────────────────────────────────
@app.route('/api/navigation/route', methods=['POST'])
def get_route():
    data        = request.json
    src         = data.get('from', 'entrance')
    dst         = data.get('to', 'platform_1')
    route, steps = navigator.get_route(src, dst)
    return jsonify({'route': route, 'steps': steps})


@app.route('/api/navigation/map')
def get_map():
    return jsonify(navigator.get_station_map())


# ── API: TTS ──────────────────────────────────────────────────────
@app.route('/api/speak', methods=['POST'])
def speak():
    text = request.json.get('text', '')
    audio_guide.speak_async(text)
    return jsonify({'success': True})


# ── Model Management ──────────────────────────────────────────────
@app.route('/api/models')
def list_models():
    models = [f for f in os.listdir('models') if f.endswith('.pt')]
    return jsonify({'models': models})


@app.route('/api/models/upload', methods=['POST'])
def upload_model():
    f = request.files.get('model')
    if f and f.filename.endswith('.pt'):
        path = os.path.join('models', secure_filename(f.filename))
        f.save(path)
        return jsonify({'success': True, 'model': f.filename})
    return jsonify({'success': False, 'message': 'Please upload a .pt file'})



# ── Demo Page ──────────────────────────────────────────────────────
@app.route('/demo')
def demo_page():
    models = [f for f in os.listdir('models') if f.endswith('.pt')]
    return render_template('demo.html', models=models)


# ── API: Generate Synthetic Dataset ──────────────────────────────
@app.route('/api/dataset/synthetic', methods=['POST'])
def gen_synthetic():
    data    = request.json
    n_train = int(data.get('n_train', 200))
    n_val   = int(data.get('n_val', 50))

    def run():
        import subprocess, sys
        subprocess.run([
            sys.executable, 'download_dataset.py',
            '--dataset', 'synthetic',
            '--train', str(n_train),
            '--val',   str(n_val)
        ])

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({
        'success': True,
        'message': f'Generating {n_train} train + {n_val} val images. Check terminal. Then go to Train page.'
    })


# ── API: Download Roboflow Dataset ────────────────────────────────
@app.route('/api/dataset/roboflow', methods=['POST'])
def dl_roboflow():
    data      = request.json
    api_key   = data.get('api_key', '')
    workspace = data.get('workspace', '')
    project   = data.get('project', '')
    version   = int(data.get('version', 1))

    if not all([api_key, workspace, project]):
        return jsonify({'success': False, 'message': 'Missing api_key / workspace / project'})

    def run():
        import subprocess, sys
        subprocess.run([
            sys.executable, 'download_dataset.py',
            '--dataset',   'roboflow',
            '--api-key',   api_key,
            '--workspace', workspace,
            '--project',   project,
            '--version',   str(version)
        ])

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({
        'success': True,
        'message': 'Downloading from Roboflow. Check terminal. Then go to Train page.'
    })


if __name__ == '__main__':
    print("="*55)
    print("  Navigation Assistance System for Differently-Abled")
    print("  http://127.0.0.1:5000")
    print("="*55)
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
