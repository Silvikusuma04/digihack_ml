import os
import io
import uuid
import base64
import tempfile
from flask import Flask, request, render_template, jsonify, url_for
from ultralytics import YOLO
from PIL import Image
import cv2

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = os.path.join('static', 'results')
ALLOWED_EXT = {'png','jpg','jpeg'}

MODEL_PATH = os.environ.get('MODEL_PATH', './best.pt')
DEVICE = os.environ.get('YOLO_DEVICE', 'cpu')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

print('Loading model from', MODEL_PATH)
model = YOLO(MODEL_PATH)
print('Model loaded')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_b64 = data['image']
    if img_b64.startswith('data:'):
        img_b64 = img_b64.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception:
        return jsonify({'error': 'Invalid image encoding'}), 400

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
    os.close(tmp_fd)
    with open(tmp_path, 'wb') as f:
        f.write(img_bytes)

    try:
        results = model.predict(source=tmp_path, device=DEVICE, save=False)
        r = results[0]
        plotted = r.plot()
        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        out_name = f"{uuid.uuid4().hex}_pred.png"
        out_path = os.path.join(app.config['RESULT_FOLDER'], out_name)
        Image.fromarray(plotted_rgb).save(out_path)
        result_url = url_for('static', filename=f'results/{out_name}')

        return jsonify({'result_url': result_url})
    finally:
        os.remove(tmp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)