"""
routes/api.py — JSON API endpoints
Used by the frontend JavaScript for async operations.
"""

import os
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from models.db_models import Scan, Report
from app_factory import db

api_bp = Blueprint('api', __name__)


@api_bp.route('/scan/<int:scan_id>/reports', methods=['GET'])
@login_required
def get_reports(scan_id):
    """Return all reports for a scan as JSON."""
    scan = Scan.query.get_or_404(scan_id)

    # Security: doctor must own scan, patient must be linked to scan
    if current_user.role == 'doctor':
        if scan.doctor_id != current_user.doctor_profile.id:
            return jsonify({'error': 'Access denied'}), 403
    elif current_user.role == 'patient':
        if scan.patient_id != current_user.patient_profile.id:
            return jsonify({'error': 'Access denied'}), 403

    return jsonify({
        'scan_id': scan.id,
        'status':  scan.status,
        'reports': [r.to_dict() for r in scan.reports]
    })


@api_bp.route('/model-scores', methods=['GET'])
def model_scores():
    """Return pre-computed model scores from JSON files."""
    import json
    RESULTS_DIR = current_app.config['RESULTS_DIR']
    scores = {}
    for model in ['a', 'b', 'c', 'd']:
        path = os.path.join(current_app.config['MODEL_DIR'], f'scores_{model}.json')
        if not os.path.exists(path):
            path = os.path.join(RESULTS_DIR, f'scores_{model}.json')
        if os.path.exists(path):
            with open(path) as f:
                scores[f'model_{model}'] = json.load(f)
    return jsonify(scores)


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check — confirms models are loaded."""
    from models.inference import _models_loaded, _model_a, _model_b, _model_e_classifier
    return jsonify({
        'status':      'ok',
        'models_loaded': _models_loaded,
        'model_a':     _model_a is not None,
        'model_b':     _model_b is not None,
        'classifier':  _model_e_classifier is not None,
    })
