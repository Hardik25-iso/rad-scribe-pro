"""
routes/doctor.py — Doctor portal routes
"""

import os
import uuid
import threading
import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app_factory import db
from models.db_models import Scan, Report, ConfidenceSentence, PatientProfile

doctor_bp = Blueprint('doctor', __name__)

def _doctor_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'doctor':
            flash('Access restricted to doctors.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated

def _allowed_file(filename):
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', {'jpg','jpeg','png'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

# ── Image Serving Route ───────────────────────────────────────────────────────
@doctor_bp.route('/uploads/<path:filename>')
@login_required
@_doctor_required
def uploaded_file(filename):
    """Securely serve uploaded X-ray images to the dashboard."""
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

# ── Background Inference Engine ───────────────────────────────────────────────
def background_inference(app, scan_id, save_path, models_to_run):
    """Runs heavy ML models in a separate thread so Flask doesn't freeze."""
    with app.app_context():
        try:
            scan = Scan.query.get(scan_id)
            if not scan: return

            from models.inference import run_inference
            results = run_inference(save_path, models_to_run)

            for model_name, res in results.items():
                report = Report(
                    scan_id         = scan.id,
                    model_name      = model_name,
                    generated_text  = res['text'],
                    clinical_label  = res.get('clinical_label'),
                    avg_log_prob    = res.get('avg_log_prob'),
                    # 👇 CRITICAL: Saves the RAG cases to the database 👇
                    retrieved_cases = res.get('retrieved_cases') 
                )
                db.session.add(report)
                db.session.flush()

                for sent_data in res.get('sentences', []):
                    sentence = ConfidenceSentence(
                        report_id    = report.id,
                        sentence     = sent_data['sentence'],
                        avg_log_prob = sent_data.get('avg_log_prob'),
                        is_flagged   = sent_data.get('is_flagged', False),
                    )
                    db.session.add(sentence)

            scan.status = 'complete'
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print("!!! AI PIPELINE CRASHED !!! ->", str(e))
            logging.error(f"Inference thread failed for scan {scan_id}: {e}")
            scan = Scan.query.get(scan_id)
            if scan:
                scan.status = 'error'
                db.session.commit()

# ── Dashboards & Operations ───────────────────────────────────────────────────
@doctor_bp.route('/dashboard')
@login_required
@_doctor_required
def dashboard():
    doc     = current_user.doctor_profile
    scans   = Scan.query.filter_by(doctor_id=doc.id).order_by(Scan.uploaded_at.desc()).limit(10).all()
    scan_count = Scan.query.filter_by(doctor_id=doc.id).count()
    return render_template('doctor_dashboard.html', doctor=doc, scans=scans, scan_count=scan_count)

@doctor_bp.route('/new-scan', methods=['GET', 'POST'])
@login_required
@_doctor_required
def new_scan():
    doc = current_user.doctor_profile
    if request.method == 'POST':
        file = request.files.get('xray_file')
        if not file or file.filename == '':
            flash('Please select a chest X-ray file.', 'error')
            return redirect(request.url)
        if not _allowed_file(file.filename):
            flash('File type not supported. Use JPEG, PNG, or DICOM.', 'error')
            return redirect(request.url)

        try:
            ext      = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            patient_id_str = request.form.get('patient_id', '').strip()
            patient_profile = None
            if patient_id_str:
                patient_profile = PatientProfile.query.filter_by(patient_id=patient_id_str).first()

            scan = Scan(
                doctor_id     = doc.id,
                patient_id    = patient_profile.id if patient_profile else None,
                filename      = filename,
                original_name = secure_filename(file.filename),
                notes         = request.form.get('notes', ''),
                status        = 'processing',
            )
            db.session.add(scan)
            db.session.commit()

            # Ensure we are only passing the single selected model from the hidden input
            models_to_run = [request.form.get('models')]  

            # Start inference in the background without blocking the UI
            app = current_app._get_current_object()
            thread = threading.Thread(target=background_inference, args=(app, scan.id, save_path, models_to_run))
            thread.daemon = True
            thread.start()

            # Pass the scan_id as a query parameter so the frontend knows to start polling
            return redirect(url_for('doctor.view_scan', scan_id=scan.id, polling=scan.id))

        except Exception as e:
            db.session.rollback()
            logging.error(f"Scan creation failed: {e}")
            flash('Failed to process upload due to a database error.', 'error')
            return redirect(request.url)
        
    patients = PatientProfile.query.all()
    return render_template('new_scan.html', doctor=doc, patients=patients)

@doctor_bp.route('/scan/<int:scan_id>')
@login_required
@_doctor_required
def view_scan(scan_id):
    doc  = current_user.doctor_profile
    scan = Scan.query.get_or_404(scan_id)
    if scan.doctor_id != doc.id:
        flash('Access denied.', 'error')
        return redirect(url_for('doctor.dashboard'))
    
    reports = {r.model_name: r for r in scan.reports}
    is_polling = request.args.get('polling') is not None
    
    return render_template('view_scan.html', scan=scan, reports=reports, doctor=doc, is_polling=is_polling)

# ── Polling & Deletion Endpoints ──────────────────────────────────────────────
@doctor_bp.route('/scan/<int:scan_id>/status')
@login_required
@_doctor_required
def scan_status(scan_id):
    """Frontend JavaScript hits this to check if the background thread finished."""
    scan = Scan.query.get_or_404(scan_id)
    if scan.doctor_id != current_user.doctor_profile.id:
        return jsonify({'status': 'unauthorized'}), 403
    return jsonify({'status': scan.status})

@doctor_bp.route('/scan/<int:scan_id>/delete', methods=['POST'])
@login_required
@_doctor_required
def delete_scan(scan_id):
    """Securely deletes a scan, its reports, and the physical X-ray file."""
    doc = current_user.doctor_profile
    scan = Scan.query.get_or_404(scan_id)

    if scan.doctor_id != doc.id:
        flash('Access denied. You can only delete your own scans.', 'error')
        return redirect(url_for('doctor.dashboard'))

    try:
        if scan.filename:
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], scan.filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        db.session.delete(scan)
        db.session.commit()
        flash('Scan deleted successfully.', 'success')
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Failed to delete scan {scan_id}: {e}")
        flash('Failed to delete scan due to a database error.', 'error')

    if request.args.get('from') == 'archive':
        return redirect(url_for('doctor.all_reports'))
    return redirect(url_for('doctor.dashboard'))

# ── Finished Operational Routes ───────────────────────────────────────────────
@doctor_bp.route('/reports')
@login_required
@_doctor_required
def all_reports():
    doc   = current_user.doctor_profile
    scans = Scan.query.filter_by(doctor_id=doc.id).order_by(Scan.uploaded_at.desc()).all()
    return render_template('all_reports.html', scans=scans, doctor=doc)

@doctor_bp.route('/patients')
@login_required
@_doctor_required
def patient_list():
    doc = current_user.doctor_profile
    patients = PatientProfile.query.all()
    return render_template('patient_list.html', doctor=doc, patients=patients)

@doctor_bp.route('/model-comparison')
@login_required
@_doctor_required
def model_comparison():
    return render_template('model_comparison.html')

@doctor_bp.route('/performance-stats')
@login_required
@_doctor_required
def performance_stats():
    doc = current_user.doctor_profile
    return render_template('performance_stats.html', doctor=doc)

@doctor_bp.route('/settings')
@login_required
@_doctor_required
def settings():
    doc = current_user.doctor_profile
    user_account = current_user
    return render_template('settings.html', doctor=doc, user=user_account)