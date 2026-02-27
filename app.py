import os
import uuid
import hashlib
import zipfile
import io
import logging
import threading
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from werkzeug.utils import secure_filename
from datetime import datetime
from functools import wraps

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Security: Load secret key from environment or generate secure key
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUTS_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUTS_FOLDER'] = OUTPUTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

# Import after config
from database import (
    init_db, add_document, get_document, get_all_documents, 
    update_document_status, delete_document, get_document_files,
    get_folder_tree, get_file_content, get_safe_path, SecurityError, get_document_log
)
from tasks import (
    register_listener, unregister_listener, send_log, start_processing_thread,
    create_log_queue, get_log_queue, remove_log_queue, get_task_status,
    get_queue_position
)


# Rate limiting storage (simple in-memory implementation)
_request_counts = {}
_rate_limit_lock = threading.Lock()
RATE_LIMIT_REQUESTS = 100  # Max requests per window
RATE_LIMIT_WINDOW = 60  # seconds


def rate_limit_exceeded():
    """Check if rate limit is exceeded"""
    import threading
    import time
    
    thread_id = threading.get_ident()
    current_time = time.time()
    
    with _rate_limit_lock:
        if thread_id not in _request_counts:
            _request_counts[thread_id] = []
        
        # Clean old requests
        _request_counts[thread_id] = [
            t for t in _request_counts[thread_id] 
            if current_time - t < RATE_LIMIT_WINDOW
        ]
        
        # Check limit
        if len(_request_counts[thread_id]) >= RATE_LIMIT_REQUESTS:
            return True
        
        _request_counts[thread_id].append(current_time)
        return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_unique_filename(original_filename):
    """Generate a unique filename using timestamp and hash"""
    ext = original_filename.rsplit('.', 1)[1].lower()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_hash = hashlib.md5(f"{original_filename}{timestamp}".encode()).hexdigest()[:8]
    return f"{timestamp}_{short_hash}.{ext}"


@app.route('/')
def index():
    """Main page showing all documents"""
    documents = get_all_documents()
    return render_template('index.html', documents=documents)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    # Get optional parameters
    backend = request.form.get('backend', 'hybrid-auto-engine')
    lang = request.form.get('lang', 'ch')
    
    # Generate unique filename and save
    original_filename = secure_filename(file.filename)
    unique_filename = generate_unique_filename(original_filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Add to database
    doc_id = add_document(unique_filename, original_filename, file_size, backend, lang)
    
    # Start processing in background
    start_processing_thread(doc_id, file_path, backend, lang)
    
    return jsonify({
        'message': 'File uploaded successfully',
        'doc_id': doc_id,
        'filename': original_filename
    })


@app.route('/documents', methods=['GET'])
def list_documents():
    """Get list of all documents"""
    documents = get_all_documents()
    return jsonify(documents)


@app.route('/documents/<int:doc_id>', methods=['GET'])
def get_doc(doc_id):
    """Get document details"""
    doc = get_document(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    files = get_document_files(doc_id)
    return jsonify({
        'document': doc,
        'files': files
    })


@app.route('/documents/<int:doc_id>', methods=['DELETE'])
def delete_doc(doc_id):
    """Delete a document"""
    success = delete_document(doc_id)
    if success:
        return jsonify({'message': 'Document deleted successfully'})
    return jsonify({'error': 'Document not found'}), 404


@app.route('/documents/<int:doc_id>/logs', methods=['GET'])
def get_doc_logs(doc_id):
    """Get historical logs for a document"""
    result = get_document_log(doc_id)
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


@app.route('/documents/<int:doc_id>/download/<path:filename>', methods=['GET'])
def download_file(doc_id, filename):
    """Download a processed file"""
    doc = get_document(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    # Use safe path validation
    try:
        file_path = os.path.join(doc['output_path'], filename)
        safe_path = get_safe_path(file_path)
    except SecurityError as e:
        return jsonify({'error': str(e)}), 403
    
    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(safe_path, as_attachment=True)


@app.route('/logs/<int:doc_id>')
def stream_logs(doc_id):
    """Stream logs for a document using Server-Sent Events"""
    task_id = f"task_{doc_id}"
    
    # Create log queue for this stream
    log_queue = create_log_queue(task_id)
    
    def generate():
        import time
        start_time = time.time()
        timeout_seconds = 600  # 10 minutes timeout
        
        try:
            while True:
                elapsed = time.time() - start_time
                
                # Check if task is completed or failed
                status = get_task_status(task_id)
                if status.get('status') in ['completed', 'failed']:
                    # Send final status message
                    yield f"data: [SYSTEM] Task {status.get('status')}\n\n"
                    break
                
                # Check timeout
                if elapsed > timeout_seconds:
                    yield f"data: [SYSTEM] Timeout - no logs received\n\n"
                    break
                
                # Try to get log from queue with timeout
                try:
                    log_message = log_queue.get(timeout=1)
                    yield f"data: {log_message}\n\n"
                except:
                    # Queue empty, continue waiting
                    # Only send waiting message on first few seconds to avoid flooding
                    if elapsed < 3:
                        yield f"data: [SYSTEM] Waiting for logs...\n\n"
                    continue
        except GeneratorExit:
            pass
        finally:
            # Clean up queue
            remove_log_queue(task_id)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/folder-tree', methods=['GET'])
def get_tree():
    """Get folder tree structure"""
    doc_id = request.args.get('doc_id', type=int)
    tree = get_folder_tree(doc_id)
    return jsonify(tree)


@app.route('/file-content', methods=['GET'])
def get_content():
    """Get file content for preview"""
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({'error': 'No file path provided'}), 400
    
    content = get_file_content(file_path)
    return jsonify(content)


@app.route('/download-zip/<int:doc_id>', methods=['GET'])
def download_zip(doc_id):
    """Download entire document folder as ZIP"""
    doc = get_document(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    output_path = doc['output_path']
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'Output folder not found'}), 404
    
    # Create ZIP in memory
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_path)
                zf.write(file_path, arcname)
    
    memory_file.seek(0)
    
    # Generate filename for ZIP
    zip_filename = f"{doc['original_filename']}_output.zip"
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename
    )


@app.route('/task-status/<int:doc_id>', methods=['GET'])
def task_status(doc_id):
    """Check task processing status"""
    doc = get_document(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    task_id = f"task_{doc_id}"
    queue_position = get_queue_position(task_id)
    
    return jsonify({
        'doc_id': doc_id,
        'status': doc['status'],
        'error_message': doc.get('error_message'),
        'queue_position': queue_position
    })


@app.route('/download-file', methods=['GET'])
def download_single_file():
    """Download a single file by path"""
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({'error': 'No file path provided'}), 400
    
    # Use safe path validation
    try:
        safe_path = get_safe_path(file_path)
    except SecurityError as e:
        return jsonify({'error': str(e)}), 403
    
    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404
    
    filename = os.path.basename(safe_path)
    return send_file(safe_path, as_attachment=True, download_name=filename)


@app.route('/view-image', methods=['GET'])
def view_image():
    """View an image file"""
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({'error': 'No file path provided'}), 400
    
    # Use safe path validation
    try:
        safe_path = get_safe_path(file_path)
    except SecurityError as e:
        return jsonify({'error': str(e)}), 403
    
    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Determine content type
    ext = os.path.splitext(safe_path)[1].lower()
    content_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    content_type = content_types.get(ext, 'application/octet-stream')
    
    return send_file(safe_path, mimetype=content_type)


@app.route('/view-pdf', methods=['GET'])
def view_pdf():
    """View a PDF file"""
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({'error': 'No file path provided'}), 400
    
    # Use safe path validation
    try:
        safe_path = get_safe_path(file_path)
    except SecurityError as e:
        return jsonify({'error': str(e)}), 403
    
    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Validate the file is actually a PDF
    ext = os.path.splitext(safe_path)[1].lower()
    if ext != '.pdf':
        return jsonify({'error': 'Not a PDF file'}), 400
    
    return send_file(safe_path, mimetype='application/pdf')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
