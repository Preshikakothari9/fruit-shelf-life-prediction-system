import os
import uuid
import zipfile
from threading import Thread
from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory, redirect, url_for
from flask_login import login_required, current_user
from app import db
from app.models import FreshnessHistory
from app.predictor import predict_freshness

main_bp = Blueprint('main', __name__)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('landing.html')


@main_bp.route('/dashboard')
@login_required
def dashboard():
    history = (
        FreshnessHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(FreshnessHistory.created_at.desc())
        .limit(20)
        .all()
    )
    return render_template('dashboard.html', history=history)


@main_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = predict_freshness(filepath)
    except Exception as e:
        current_app.logger.exception("Prediction failed")
        return jsonify({'error': 'Could not analyse this image. Please try a different photo.'}), 500

    try:
        record = FreshnessHistory(
            user_id=current_user.id,
            image_filename=filename,
            fruit_type=result['fruit_type'],
            predicted_days=result['predicted_days'],
            confidence=result['confidence'],
            freshness_status=result['freshness_status'],
            storage_tip=result['storage_tip'],
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            'success': True,
            'prediction': {
                'id': record.id,
                'fruit_type': result['fruit_type'],
                'predicted_days': result['predicted_days'],
                'confidence': round(result['confidence'] * 100, 1),
                'freshness_status': result['freshness_status'],
                'storage_tip': result['storage_tip'],
                'temperature': result.get('temperature', 'N/A'),
                'image_url': f'/uploads/{filename}',
                'emoji': result.get('emoji', '🍎'),
            },
        })
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Failed to save prediction")
        return jsonify({'error': 'Could not save prediction.'}), 500


@main_bp.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@main_bp.route('/history')
@login_required
def history():
    records = (
        FreshnessHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(FreshnessHistory.created_at.desc())
        .all()
    )
    return jsonify([r.to_dict() for r in records])


@main_bp.route('/history/<int:record_id>', methods=['DELETE'])
@login_required
def delete_history(record_id):
    record = FreshnessHistory.query.filter_by(
        id=record_id, user_id=current_user.id
    ).first()
    if not record:
        return jsonify({'error': 'Not found'}), 404

    filename = record.image_filename
    db.session.delete(record)
    db.session.commit()

    # Remove the image only after the DB row is gone.
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            current_app.logger.warning("Could not remove file %s", filepath)
    return jsonify({'success': True})


@main_bp.route('/train_model', methods=['GET', 'POST'])
@login_required
def train_model():
    if request.method == 'GET':
        return render_template('train.html')
        
    if 'dataset' not in request.files:
        return jsonify({'error': 'No dataset uploaded'}), 400
        
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'Only ZIP files are supported for training datasets'}), 400
        
    # Save zip file
    zip_path = os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset.zip")
    file.save(zip_path)
    
    # Extract zip file
    extract_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], "dataset_extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Defend against Zip-Slip: ensure every member stays inside extract_dir.
            extract_root = os.path.realpath(extract_dir)
            for member in zip_ref.namelist():
                target = os.path.realpath(os.path.join(extract_dir, member))
                if not (target == extract_root or target.startswith(extract_root + os.sep)):
                    return jsonify({'error': 'Unsafe path detected in ZIP archive.'}), 400
            zip_ref.extractall(extract_dir)
            
        # Run training in background
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from train import start_training
        
        # Look for the first directory inside extract_dir in case the zip contains a top-level folder
        dataset_dir = extract_dir
        subdirs = [os.path.join(extract_dir, d) for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
        if len(subdirs) == 1:
            dataset_dir = subdirs[0]
            
        thread = Thread(target=start_training, args=(dataset_dir,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Dataset uploaded and training started in background!'})
    except Exception:
        current_app.logger.exception("Failed to process dataset")
        return jsonify({'error': 'Failed to process dataset.'}), 500

