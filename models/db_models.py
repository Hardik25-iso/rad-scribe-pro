"""
models/db_models.py — All SQLAlchemy database models

Tables:
  - User         (doctors and patients — role-separated)
  - DoctorProfile
  - PatientProfile
  - Scan         (one per uploaded X-ray)
  - Report       (one per model run per scan)
  - ConfidenceSentence (per sentence in a report)
"""

from datetime import datetime
from flask_login import UserMixin
from app_factory import db, login_manager


# ── User loader for Flask-Login ───────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ─────────────────────────────────────────────────────────────────────────────
# USER
# ─────────────────────────────────────────────────────────────────────────────
class User(db.Model, UserMixin):
    __tablename__ = 'user'

    id           = db.Column(db.Integer, primary_key=True)
    email        = db.Column(db.String(120), unique=True, nullable=False)
    password     = db.Column(db.String(128), nullable=False)
    role         = db.Column(db.Enum('doctor', 'patient'), nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    is_active    = db.Column(db.Boolean, default=True)

    # Relationships
    doctor_profile  = db.relationship('DoctorProfile',  back_populates='user', uselist=False, cascade='all, delete-orphan')
    patient_profile = db.relationship('PatientProfile', back_populates='user', uselist=False, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.email} ({self.role})>'


# ─────────────────────────────────────────────────────────────────────────────
# DOCTOR PROFILE
# ─────────────────────────────────────────────────────────────────────────────
class DoctorProfile(db.Model):
    __tablename__ = 'doctor_profile'

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    full_name      = db.Column(db.String(150), nullable=False)
    license_number = db.Column(db.String(60),  unique=True, nullable=False)
    specialization = db.Column(db.String(100), nullable=True)
    hospital       = db.Column(db.String(200), nullable=True)
    city           = db.Column(db.String(80),  nullable=True)

    user  = db.relationship('User', back_populates='doctor_profile')
    scans = db.relationship('Scan', back_populates='doctor', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<DoctorProfile {self.full_name}>'


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT PROFILE
# ─────────────────────────────────────────────────────────────────────────────
class PatientProfile(db.Model):
    __tablename__ = 'patient_profile'

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    full_name    = db.Column(db.String(150), nullable=False)
    patient_id   = db.Column(db.String(30),  unique=True, nullable=False)   # e.g. PID-2025-0047
    date_of_birth= db.Column(db.Date,        nullable=True)
    mobile       = db.Column(db.String(20),  nullable=True)
    hospital     = db.Column(db.String(200), nullable=True)

    user  = db.relationship('User', back_populates='patient_profile')
    scans = db.relationship('Scan', back_populates='patient', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<PatientProfile {self.full_name} ({self.patient_id})>'


# ─────────────────────────────────────────────────────────────────────────────
# SCAN  (one per uploaded X-ray image)
# ─────────────────────────────────────────────────────────────────────────────
class Scan(db.Model):
    __tablename__ = 'scan'

    id             = db.Column(db.Integer, primary_key=True)
    doctor_id      = db.Column(db.Integer, db.ForeignKey('doctor_profile.id'), nullable=False)
    patient_id     = db.Column(db.Integer, db.ForeignKey('patient_profile.id'), nullable=True)
    filename       = db.Column(db.String(255), nullable=False)          # saved filename on disk
    original_name  = db.Column(db.String(255), nullable=True)           # original upload name
    uploaded_at    = db.Column(db.DateTime, default=datetime.utcnow)
    notes          = db.Column(db.Text, nullable=True)                  # optional doctor notes
    status         = db.Column(
        db.Enum('pending', 'processing', 'complete', 'error'),
        default='pending', nullable=False
    )

    doctor  = db.relationship('DoctorProfile',  back_populates='scans')
    patient = db.relationship('PatientProfile',  back_populates='scans')
    reports = db.relationship('Report', back_populates='scan', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Scan {self.id} — {self.filename}>'


# ─────────────────────────────────────────────────────────────────────────────
# REPORT  (one per model per scan)
# ─────────────────────────────────────────────────────────────────────────────
class Report(db.Model):
    __tablename__ = 'report'

    id              = db.Column(db.Integer, primary_key=True)
    scan_id         = db.Column(db.Integer, db.ForeignKey('scan.id'), nullable=False)
    model_name      = db.Column(
        db.Enum('model_a', 'model_b', 'model_c', 'model_d', 'model_e'),
        nullable=False
    )
    generated_text  = db.Column(db.Text, nullable=False)
    clinical_label  = db.Column(db.Enum('Normal', 'Abnormal', 'Unclear'), nullable=True)
    avg_log_prob    = db.Column(db.Float, nullable=True)    # Model D/E confidence
    bleu4           = db.Column(db.Float, nullable=True)
    rouge_l         = db.Column(db.Float, nullable=True)
    bertscore       = db.Column(db.Float, nullable=True)
    
    # 👇 NEW COLUMN FOR FAISS / COUNTERFACTUAL RAG OUTPUT 👇
    retrieved_cases = db.Column(db.JSON, nullable=True) 
    
    generated_at    = db.Column(db.DateTime, default=datetime.utcnow)

    scan      = db.relationship('Scan', back_populates='reports')
    sentences = db.relationship('ConfidenceSentence', back_populates='report', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id':              self.id,
            'model_name':      self.model_name,
            'generated_text':  self.generated_text,
            'clinical_label':  self.clinical_label,
            'avg_log_prob':    self.avg_log_prob,
            'bleu4':           self.bleu4,
            'rouge_l':         self.rouge_l,
            'bertscore':       self.bertscore,
            'retrieved_cases': self.retrieved_cases, # Included in dictionary export
            'sentences':       [s.to_dict() for s in self.sentences],
        }

    def __repr__(self):
        return f'<Report scan={self.scan_id} model={self.model_name}>'


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SENTENCE  (per sentence in a report)
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceSentence(db.Model):
    __tablename__ = 'confidence_sentence'

    id          = db.Column(db.Integer, primary_key=True)
    report_id   = db.Column(db.Integer, db.ForeignKey('report.id'), nullable=False)
    sentence    = db.Column(db.Text, nullable=False)
    avg_log_prob= db.Column(db.Float, nullable=True)
    is_flagged  = db.Column(db.Boolean, default=False)  # True = low confidence, verify manually
    position    = db.Column(db.Integer, nullable=True)  # order in report

    report = db.relationship('Report', back_populates='sentences')

    def to_dict(self):
        return {
            'sentence':     self.sentence,
            'avg_log_prob': self.avg_log_prob,
            'is_flagged':   self.is_flagged,
        }

    def __repr__(self):
        return f'<Sentence report={self.report_id} flagged={self.is_flagged}>'