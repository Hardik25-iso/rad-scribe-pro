"""
routes/patient.py — Patient portal routes
"""

from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from models.db_models import Scan, Report

# THIS is the line your app was looking for and couldn't find!
patient_bp = Blueprint('patient', __name__)

def _patient_required(f):
    """Decorator: only patients can access this route."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'patient':
            flash('Access restricted to patients.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated

# ── Dashboards & Operations ───────────────────────────────────────────────────

@patient_bp.route('/dashboard')
@login_required
@_patient_required
def dashboard():
    patient = current_user.patient_profile
    scans   = Scan.query.filter_by(patient_id=patient.id).order_by(Scan.uploaded_at.desc()).all()
    return render_template('patient_dashboard.html', patient=patient, scans=scans)

@patient_bp.route('/report/<int:scan_id>')
@login_required
@_patient_required
def view_report(scan_id):
    patient = current_user.patient_profile
    scan    = Scan.query.get_or_404(scan_id)
    
    # Security check: Ensure the patient can only view their own scans
    if scan.patient_id != patient.id:
        flash('Access denied.', 'error')
        return redirect(url_for('patient.dashboard'))
        
    # Show best available report (prefer model_d, fallback to model_c, etc.)
    report = None
    for model in ['model_d', 'model_c', 'model_b', 'model_e', 'model_a']:
        report = Report.query.filter_by(scan_id=scan.id, model_name=model).first()
        if report:
            break
            
    return render_template('patient_report.html', scan=scan, report=report, patient=patient)