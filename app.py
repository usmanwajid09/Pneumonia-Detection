import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from utils import load_config, get_model, predict_image, get_governance_metrics

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load configuration and model globally
try:
    config = load_config('config.yaml')
    model = get_model(config)
    governance_metrics = get_governance_metrics()
except Exception as e:
    print(f"Error loading model/config: {e}")
    config = None
    model = None
    governance_metrics = {}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', metrics=governance_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            privacy_shield = request.form.get('privacy_shield', 'false').lower() == 'true'
            
            # Run prediction
            result = predict_image(model, filepath, config, privacy_shield=privacy_shield)
            result['image_url'] = filepath
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
            
    return jsonify({'error': 'Invalid file type'})

@app.route('/log_review', methods=['POST'])
def log_review():
    data = request.json
    action = data.get('action')
    filename = data.get('filename')
    # In a real app, save to database. Here, we just print to console.
    print(f"HITL AUDIT LOG: Doctor performed '{action}' on {filename}")
    return jsonify({"status": "success", "message": f"Action '{action}' securely logged for audit."})

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, port=5000)
