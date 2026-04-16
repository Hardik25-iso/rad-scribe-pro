import os

from flask import Blueprint, current_app, jsonify
from flask_login import current_user, login_required

from models.db_models import Scan
from models.inference_d_sandbox import _discover_model_b_ckpt, generate_model_d_sandbox


sandbox_bp = Blueprint('sandbox', __name__)


def _doctor_only():
    return current_user.is_authenticated and current_user.role == 'doctor'


@sandbox_bp.route('/health', methods=['GET'])
@login_required
def health():
    if not _doctor_only():
        return jsonify({'error': 'Access denied'}), 403

    base_dir = os.path.dirname(os.path.dirname(__file__))
    try:
        ckpt = _discover_model_b_ckpt(base_dir)
    except FileNotFoundError:
        ckpt = None
    return jsonify({
        'status': 'ok',
        'model_b_ckpt_found': bool(ckpt),
        'model_b_ckpt': ckpt,
    })


@sandbox_bp.route('/model-d/scan/<int:scan_id>', methods=['GET'])
@login_required
def run_model_d_for_scan(scan_id):
    if not _doctor_only():
        return jsonify({'error': 'Access denied'}), 403

    scan = Scan.query.get_or_404(scan_id)
    if scan.doctor_id != current_user.doctor_profile.id:
        return jsonify({'error': 'Access denied'}), 403

    image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], scan.filename)
    result = generate_model_d_sandbox(image_path)
    result['scan_id'] = scan.id
    result['filename'] = scan.original_name
    return jsonify(result)
