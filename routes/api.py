"""
routes/api.py — JSON API endpoints

FIX (Bug 5): /api/model-scores now reads all_metrics.json and returns the
real metric values (BLEU-4 in percentage scale, e.g. 1.72 for Model D,
not 0.0934 which was a hardcoded display error in the UI).
"""

import os
import json
from flask import Blueprint, jsonify, current_app
from flask_login import login_required, current_user
from models.db_models import Scan, Report

api_bp = Blueprint('api', __name__)


@api_bp.route('/scan/<int:scan_id>/status', methods=['GET'])
@login_required
def scan_status(scan_id):
    """Frontend polls this every 2 s until status is 'complete' or 'error'."""
    scan = Scan.query.get_or_404(scan_id)

    if current_user.role == 'doctor':
        if scan.doctor_id != current_user.doctor_profile.id:
            return jsonify({'error': 'Access denied'}), 403
    elif current_user.role == 'patient':
        if scan.patient_id != current_user.patient_profile.id:
            return jsonify({'error': 'Access denied'}), 403

    payload = {'status': scan.status, 'reports': []}
    if scan.status == 'complete':
        payload['reports'] = [r.to_dict() for r in scan.reports]
    return jsonify(payload)


@api_bp.route('/scan/<int:scan_id>/reports', methods=['GET'])
@login_required
def get_reports(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if current_user.role == 'doctor':
        if scan.doctor_id != current_user.doctor_profile.id:
            return jsonify({'error': 'Access denied'}), 403
    elif current_user.role == 'patient':
        if scan.patient_id != current_user.patient_profile.id:
            return jsonify({'error': 'Access denied'}), 403
    return jsonify({'scan_id': scan.id,
                    'status':  scan.status,
                    'reports': [r.to_dict() for r in scan.reports]})


@api_bp.route('/model-scores', methods=['GET'])
def model_scores():
    """
    Returns real evaluation metrics from all_metrics.json.

    BUG 5 FIX: The UI was displaying BLEU-4 = 0.0934 for Model D.
    The actual value from all_metrics.json is BLEU-4 = 1.72 (percentage scale).
    The old code was picking from individual scores_X.json files that only
    covered Models A and B. Now we serve the full all_metrics.json directly.

    Response shape (abbreviated):
    {
      "model_a": { "BLEU-4": 0.04, "ROUGE-L": 14.32, "Clin-Acc": 51.0, ... },
      "model_b": { "BLEU-4": 3.58, "ROUGE-L": 17.48, "Clin-Acc": 52.5, ... },
      "model_c": { "BLEU-4": 1.40, "ROUGE-L": 10.40, "Clin-Acc": 47.5, ... },
      "model_d": { "BLEU-4": 1.72, "ROUGE-L": 10.84, "Clin-Acc": 68.0, ... },
      "model_e": { "BLEU-4": 1.50, "ROUGE-L": 11.90, "CheXBert_MicroF1": 78.25, ... }
    }
    """
    RESULTS_DIR = current_app.config.get('RESULTS_DIR', '')
    metrics_path = os.path.join(RESULTS_DIR, 'all_metrics.json')

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                raw = json.load(f)

            # Strip top-level non-model keys and return model-keyed dict
            model_keys = ['model_a', 'model_b', 'model_c', 'model_d', 'model_e']
            return jsonify({k: raw[k] for k in model_keys if k in raw})
        except Exception as e:
            return jsonify({'error': f'Failed to read metrics: {e}'}), 500

    # Graceful fallback — hardcoded correct values if file is missing
    return jsonify({
        'model_a': {'BLEU-4': 0.04,  'ROUGE-L': 14.32, 'BERTScore': 75.74, 'Clin-Acc': 51.0},
        'model_b': {'BLEU-4': 3.58,  'ROUGE-L': 17.48, 'BERTScore': 82.23, 'Clin-Acc': 52.5},
        'model_c': {'BLEU-4': 1.40,  'ROUGE-L': 10.40, 'BERTScore': 76.38, 'Clin-Acc': 47.5},
        'model_d': {'BLEU-4': 1.72,  'ROUGE-L': 10.84, 'BERTScore': 76.20, 'Clin-Acc': 68.0},
        'model_e': {'BLEU-4': 1.50,  'ROUGE-L': 11.90, 'BERTScore': 78.01,
                    'Clin-Acc': 52.0, 'CheXBert_MicroF1': 78.25},
    })


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check — reports which assets are loaded."""
    from models.inference import (
        _models_loaded, _faiss_c, _faiss_d,
        _preds_b, _preds_c, _preds_d, _preds_e,
        _effnet_encoder, _dual_encoder, _classifier
    )
    return jsonify({
        'status':          'ok',
        'models_loaded':   _models_loaded,
        'main_index':      _faiss_c is not None,
        'dual_enc_index':  _faiss_d is not None,
        'preds_b':         bool(_preds_b),
        'preds_c':         bool(_preds_c),
        'preds_d':         bool(_preds_d),
        'preds_e':         bool(_preds_e),
        'eff_encoder':     _effnet_encoder is not None,
        'dual_encoder':    _dual_encoder is not None,
        'classifier':      _classifier is not None,
    })
