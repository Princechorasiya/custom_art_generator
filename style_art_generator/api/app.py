from flask import Flask, request, jsonify, send_file,render_template
from flask_cors import CORS
import os
import json
import uuid
from werkzeug.utils import secure_filename
import subprocess
import threading
import time

app = Flask(__name__,template_folder='../templates')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploaded_images'
MODELS_FOLDER = 'models'
RESULTS_FOLDER = 'generated_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/',methods=['GET'])
# def hello():
#     return jsonify({'message': 'Welcome to the Style Art Generator API!'}), 200

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/upload_style', methods=['POST'])
def upload_style_images():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files')
    if len(files) < 2 or len(files) > 10:
        return jsonify({'error': 'Please upload 2-10 style images'}), 400
    
    # Create a unique folder for this style set
    style_id = str(uuid.uuid4())
    style_folder = os.path.join(UPLOAD_FOLDER, style_id)
    os.makedirs(style_folder, exist_ok=True)
    
    # Save all uploaded files
    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(style_folder, filename)
            file.save(file_path)
            filenames.append(filename)
    
    # Store metadata
    metadata = {
        'style_id': style_id,
        'images': filenames,
        'timestamp': time.time(),
        'status': 'uploaded'
    }
    
    with open(os.path.join(style_folder, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    return jsonify({
        'message': 'Style images uploaded successfully',
        'style_id': style_id,
        'count': len(filenames)
    }), 200

@app.route('/api/finetune', methods=['POST'])
def finetune_model():
    data = request.json
    style_id = data.get('style_id')
    
    if not style_id:
        return jsonify({'error': 'Style ID is required'}), 400
    
    style_folder = os.path.join(UPLOAD_FOLDER, style_id)
    if not os.path.exists(style_folder):
        return jsonify({'error': 'Style not found'}), 404
    
    # Update status
    metadata_path = os.path.join(style_folder, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    metadata['status'] = 'finetuning'
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    # Start fine-tuning in a background thread
    thread = threading.Thread(target=run_finetuning, args=(style_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Fine-tuning started',
        'style_id': style_id
    }), 202

def run_finetuning(style_id):
    # Call the LoRA fine-tuning script
    # This is a placeholder - will be implemented in the fine-tuning module
    from style_art_generator.models.finetune import finetune_lora
    
    try:
        style_folder = os.path.join(UPLOAD_FOLDER, style_id)
        output_folder = os.path.join(MODELS_FOLDER, style_id)
        os.makedirs(output_folder, exist_ok=True)
        
        # Run fine-tuning
        lora_path = finetune_lora(style_folder, output_folder)
        
        # Update metadata
        metadata_path = os.path.join(style_folder, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['status'] = 'finetuned'
        metadata['lora_path'] = lora_path
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
    except Exception as e:
        # Update metadata with error
        print(e)
        metadata_path = os.path.join(style_folder, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['status'] = 'error'
        metadata['error'] = str(e)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

@app.route('/api/generate', methods=['POST'])
def generate_image():
    data = request.json
    style_id = data.get('style_id')
    prompt = data.get('prompt')
    negative_prompt = data.get('negative_prompt', '')
    guidance_scale = data.get('guidance_scale', 7.5)
    steps = data.get('steps', 30)
    
    if not style_id or not prompt:
        return jsonify({'error': 'Style ID and prompt are required'}), 400
    
    # Check if style exists and is fine-tuned
    metadata_path = os.path.join(UPLOAD_FOLDER, style_id, 'metadata.json')
    if not os.path.exists(metadata_path):
        return jsonify({'error': 'Style not found'}), 404
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if metadata.get('status') != 'finetuned':
        return jsonify({'error': 'Style is not fine-tuned yet'}), 400
    
    # Generate a unique ID for this generation
    generation_id = str(uuid.uuid4())
    output_path = os.path.join(RESULTS_FOLDER, f"{generation_id}.png")
    
    # Call ComfyUI for generation
    from style_art_generator.models.generate import generate_with_comfyui
    
    try:
        generate_with_comfyui(
            metadata['lora_path'],
            prompt,
            negative_prompt,
            guidance_scale,
            steps,
            output_path
        )
        
        return jsonify({
            'message': 'Image generated successfully',
            'generation_id': generation_id
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/result/<generation_id>', methods=['GET'])
def get_result(generation_id):
    output_path = os.path.join(RESULTS_FOLDER, f"{generation_id}.png")
    
    if not os.path.exists(output_path):
        return jsonify({'error': 'Generated image not found'}), 404
    
    return send_file(output_path, mimetype='image/png')

@app.route('/api/status/<style_id>', methods=['GET'])
def get_status(style_id):
    metadata_path = os.path.join(UPLOAD_FOLDER, style_id, 'metadata.json')
    
    if not os.path.exists(metadata_path):
        return jsonify({'error': 'Style not found'}), 404
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return jsonify({
        'style_id': style_id,
        'status': metadata.get('status', 'unknown'),
        'timestamp': metadata.get('timestamp')
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)