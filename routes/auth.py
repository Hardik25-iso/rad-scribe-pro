"""
routes/auth.py — Authentication routes
Handles login, registration, and logout for both doctors and patients.
"""

import re
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app_factory import db, bcrypt
from models.db_models import User, DoctorProfile, PatientProfile

auth_bp = Blueprint('auth', __name__)


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
@auth_bp.route('/')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return _redirect_by_role(current_user)

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        role     = request.form.get('role', 'doctor')   # 'doctor' or 'patient'

        user = User.query.filter_by(email=email, role=role).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Signed in successfully.', 'success')
            return _redirect_by_role(user)
        else:
            flash('Invalid credentials. Please check your email and password.', 'error')

    return render_template('login.html')


def _redirect_by_role(user):
    if user.role == 'doctor':
        return redirect(url_for('doctor.dashboard'))
    return redirect(url_for('patient.dashboard'))


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER — DOCTOR
# ─────────────────────────────────────────────────────────────────────────────
@auth_bp.route('/register/doctor', methods=['GET', 'POST'])
def register_doctor():
    if request.method == 'POST':
        email          = request.form.get('email', '').strip().lower()
        password       = request.form.get('password', '')
        full_name      = request.form.get('full_name', '').strip()
        license_number = request.form.get('license_number', '').strip()
        specialization = request.form.get('specialization', '').strip()
        hospital       = request.form.get('hospital', '').strip()
        city           = request.form.get('city', '').strip()

        # Basic validation
        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists.', 'error')
            return render_template('register_doctor.html')

        if DoctorProfile.query.filter_by(license_number=license_number).first():
            flash('A doctor with this license number already exists.', 'error')
            return render_template('register_doctor.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('register_doctor.html')

        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        user   = User(email=email, password=hashed, role='doctor')
        db.session.add(user)
        db.session.flush()   # get user.id

        profile = DoctorProfile(
            user_id=user.id,
            full_name=full_name,
            license_number=license_number,
            specialization=specialization,
            hospital=hospital,
            city=city,
        )
        db.session.add(profile)
        db.session.commit()

        flash('Account created. Please sign in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register_doctor.html')


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER — PATIENT
# ─────────────────────────────────────────────────────────────────────────────
@auth_bp.route('/register/patient', methods=['GET', 'POST'])
def register_patient():
    if request.method == 'POST':
        email      = request.form.get('email', '').strip().lower()
        password   = request.form.get('password', '')
        full_name  = request.form.get('full_name', '').strip()
        patient_id = request.form.get('patient_id', '').strip()
        dob        = request.form.get('date_of_birth', None)
        mobile     = request.form.get('mobile', '').strip()
        hospital   = request.form.get('hospital', '').strip()

        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists.', 'error')
            return render_template('register_patient.html')

        if PatientProfile.query.filter_by(patient_id=patient_id).first():
            flash('A patient with this ID already exists.', 'error')
            return render_template('register_patient.html')

        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        user   = User(email=email, password=hashed, role='patient')
        db.session.add(user)
        db.session.flush()

        from datetime import datetime
        dob_parsed = None
        if dob:
            try:
                dob_parsed = datetime.strptime(dob, '%Y-%m-%d').date()
            except ValueError:
                pass

        profile = PatientProfile(
            user_id=user.id,
            full_name=full_name,
            patient_id=patient_id,
            date_of_birth=dob_parsed,
            mobile=mobile,
            hospital=hospital,
        )
        db.session.add(profile)
        db.session.commit()

        flash('Account created. Please sign in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register_patient.html')


# ─────────────────────────────────────────────────────────────────────────────
# LOGOUT
# ─────────────────────────────────────────────────────────────────────────────
@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been signed out.', 'info')
    return redirect(url_for('auth.login'))
