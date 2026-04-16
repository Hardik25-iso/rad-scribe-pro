"""
routes/doctor.py — Doctor portal routes

Fixes applied:
  - Bug 2: background_inference no longer queries DB after rollback
  - Bug 4: models_to_run sanitised — None values filtered out
  - Image serving kept intact (send_file approach)
"""

import os
import uuid
import threading
import logging
from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, current_app, jsonify, send_file, abort)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app_factory import db
from models.db_models import Scan, Report, ConfidenceSentence, PatientProfile

doctor_bp = Blueprint('doctor', __name__)
log = logging.getLogger(__name__)


# ── Decorators ────────────────────────────────────────────────────────────────
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
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', {'jpg', 'jpeg', 'png'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


# ── Secure image serving ──────────────────────────────────────────────────────
@doctor_bp.route('/uploads/<path:filename>')
@login_required
@_doctor_required
def uploaded_file(filename):
    """Securely serves uploaded X-ray images — only accessible to logged-in doctors."""
    folder    = current_app.config['UPLOAD_FOLDER']
    full_path = os.path.join(folder, filename)
    if os.path.isfile(full_path):
        return send_file(full_path)
    abort(404)


# ── Background inference thread ───────────────────────────────────────────────
def background_inference(app, scan_id, save_path, models_to_run):
    """
    Runs ML inference in a background thread so the Flask server stays responsive.

    Bug 2 fix: we capture the scan status update in a separate try/except so that
    a DB error in the inference loop can't prevent the status being marked 'error'.
    We never re-query the session after a rollback inside the same except block.
    """
    with app.app_context():
        scan = None
        try:
            scan = Scan.query.get(scan_id)
            if not scan:
                log.error(f'background_inference: scan {scan_id} not found.')
                return

            from models.inference import run_inference
            results = run_inference(save_path, models_to_run)

            for model_name, res in results.items():
                report = Report(
                    scan_id         = scan.id,
                    model_name      = model_name,
                    generated_text  = res['text'],
                    clinical_label  = res.get('clinical_label'),
                    avg_log_prob    = res.get('avg_log_prob'),
                    retrieved_cases = res.get('retrieved_cases'),
                )
                db.session.add(report)
                db.session.flush()

                for sent_data in res.get('sentences', []):
                    db.session.add(ConfidenceSentence(
                        report_id    = report.id,
                        sentence     = sent_data['sentence'],
                        avg_log_prob = sent_data.get('avg_log_prob'),
                        is_flagged   = sent_data.get('is_flagged', False),
                    ))

            scan.status = 'complete'
            db.session.commit()
            log.info(f'Inference complete for scan {scan_id}.')

        except Exception as e:
            # ── Bug 2 fix ──────────────────────────────────────────────────
            # Rollback the failed transaction first, then open a FRESH query
            # to mark the scan as error. Never re-use objects from a rolled-
            # back session — they are in an invalid state.
            log.error(f'Inference thread failed for scan {scan_id}: {e}', exc_info=True)
            try:
                db.session.rollback()
                # Fresh query after rollback — safe
                failed_scan = Scan.query.get(scan_id)
                if failed_scan:
                    failed_scan.status = 'error'
                    db.session.commit()
            except Exception as inner:
                log.error(f'Could not update scan {scan_id} status to error: {inner}')


# ── Dashboard ─────────────────────────────────────────────────────────────────
@doctor_bp.route('/dashboard')
@login_required
@_doctor_required
def dashboard():
    doc        = current_user.doctor_profile
    scans      = (Scan.query.filter_by(doctor_id=doc.id)
                  .order_by(Scan.uploaded_at.desc()).limit(10).all())
    scan_count = Scan.query.filter_by(doctor_id=doc.id).count()
    return render_template('doctor_dashboard.html',
                           doctor=doc, scans=scans, scan_count=scan_count)


# ── New scan ──────────────────────────────────────────────────────────────────
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
            flash('File type not supported. Use JPEG or PNG.', 'error')
            return redirect(request.url)

        try:
            ext       = file.filename.rsplit('.', 1)[1].lower()
            filename  = f'{uuid.uuid4().hex}.{ext}'
            save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Resolve optional patient link
            patient_id_str  = request.form.get('patient_id', '').strip()
            patient_profile = None
            if patient_id_str:
                patient_profile = PatientProfile.query.filter_by(
                    patient_id=patient_id_str).first()

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

            # ── Bug 4 fix ──────────────────────────────────────────────────
            # getlist handles checkboxes; get handles single hidden input.
            # Filter out any None / empty values so run_inference gets a
            # clean list like ['model_d'] or ['model_a','model_b','model_d'].
            raw_models  = (request.form.getlist('models')
                           or [request.form.get('models')])
            models_to_run = [m for m in raw_models if m]
            if not models_to_run:
                models_to_run = ['model_a', 'model_b', 'model_c',
                                 'model_d', 'model_e']

            # Fire background thread — doesn't block the HTTP response
            app_obj = current_app._get_current_object()
            thread  = threading.Thread(
                target=background_inference,
                args=(app_obj, scan.id, save_path, models_to_run),
                daemon=True,
            )
            thread.start()

            return redirect(url_for('doctor.view_scan',
                                    scan_id=scan.id, polling=scan.id))

        except Exception as e:
            db.session.rollback()
            log.error(f'Scan creation failed: {e}', exc_info=True)
            flash('Failed to process upload due to a server error.', 'error')
            return redirect(request.url)

    patients = PatientProfile.query.all()
    return render_template('new_scan.html', doctor=doc, patients=patients)


# ── View scan results ─────────────────────────────────────────────────────────
@doctor_bp.route('/scan/<int:scan_id>')
@login_required
@_doctor_required
def view_scan(scan_id):
    doc  = current_user.doctor_profile
    scan = Scan.query.get_or_404(scan_id)
    if scan.doctor_id != doc.id:
        flash('Access denied.', 'error')
        return redirect(url_for('doctor.dashboard'))

    reports    = {r.model_name: r for r in scan.reports}
    is_polling = request.args.get('polling') is not None
    return render_template('view_scan.html',
                           scan=scan, reports=reports,
                           doctor=doc, is_polling=is_polling)


# ── Status polling endpoint ───────────────────────────────────────────────────
@doctor_bp.route('/scan/<int:scan_id>/status')
@login_required
@_doctor_required
def scan_status(scan_id):
    """Frontend JS polls this every 2 s to check if background inference finished."""
    scan = Scan.query.get_or_404(scan_id)
    if scan.doctor_id != current_user.doctor_profile.id:
        return jsonify({'status': 'unauthorized'}), 403
    return jsonify({'status': scan.status})


# ── Delete scan ───────────────────────────────────────────────────────────────
@doctor_bp.route('/scan/<int:scan_id>/delete', methods=['POST'])
@login_required
@_doctor_required
def delete_scan(scan_id):
    doc  = current_user.doctor_profile
    scan = Scan.query.get_or_404(scan_id)
    if scan.doctor_id != doc.id:
        flash('Access denied.', 'error')
        return redirect(url_for('doctor.dashboard'))

    try:
        # Delete physical file first
        if scan.filename:
            fp = os.path.join(current_app.config['UPLOAD_FOLDER'], scan.filename)
            if os.path.isfile(fp):
                os.remove(fp)
        db.session.delete(scan)
        db.session.commit()
        flash('Scan deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        log.error(f'Failed to delete scan {scan_id}: {e}')
        flash('Failed to delete scan.', 'error')

    if request.args.get('from') == 'archive':
        return redirect(url_for('doctor.all_reports'))
    return redirect(url_for('doctor.dashboard'))


# ── Stub routes (Issue 5 — no more dead href="#") ─────────────────────────────
@doctor_bp.route('/reports')
@login_required
@_doctor_required
def all_reports():
    doc   = current_user.doctor_profile
    scans = (Scan.query.filter_by(doctor_id=doc.id)
             .order_by(Scan.uploaded_at.desc()).all())
    return render_template('all_reports.html', scans=scans, doctor=doc)


@doctor_bp.route('/patients')
@login_required
@_doctor_required
def patient_list():
    doc      = current_user.doctor_profile
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
    return render_template('settings.html', doctor=doc, user=current_user)