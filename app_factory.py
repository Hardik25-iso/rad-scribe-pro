"""
app_factory.py — Flask application factory

FIX: load_all_models() is now called inside create_app() after db.create_all(),
     so all numpy arrays, FAISS indices, and the Model E classifier are loaded
     into memory once at server startup instead of never being loaded at all.
"""

import os
from flask import Flask, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

db            = SQLAlchemy()
bcrypt        = Bcrypt()
login_manager = LoginManager()


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')

    # ── Config ────────────────────────────────────────────────────────────────
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'radscribe-dev-secret-2025')

    if os.environ.get('RENDER'):
        db_url = os.environ.get('DATABASE_URL', '')
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    else:
        DB_USER = os.environ.get('DB_USER', 'root')
        DB_PASS = os.environ.get('DB_PASS', 'root')
        DB_HOST = os.environ.get('DB_HOST', 'localhost')
        DB_NAME = os.environ.get('DB_NAME', 'radscribedb')
        app.config['SQLALCHEMY_DATABASE_URI'] = (
            f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}'
        )

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER']      = os.path.join(os.path.dirname(__file__), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'dcm'}

    BASE = os.path.dirname(__file__)
    app.config['MODEL_DIR']   = os.path.join(BASE, 'model_files', 'models')
    app.config['INDEX_DIR']   = os.path.join(BASE, 'model_files', 'index')
    app.config['INDEX_D_DIR'] = os.path.join(BASE, 'model_files', 'index_d')
    app.config['RESULTS_DIR'] = os.path.join(BASE, 'model_files', 'results')
    app.config['MODEL_E_DIR'] = os.path.join(BASE, 'model_files', 'model_e')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # ── Init extensions ───────────────────────────────────────────────────────
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view    = 'auth.login'
    login_manager.login_message = 'Please sign in to access this page.'

    # ── Register blueprints ───────────────────────────────────────────────────
    from routes.auth    import auth_bp
    from routes.doctor  import doctor_bp
    from routes.patient import patient_bp

    app.register_blueprint(auth_bp,    url_prefix='/auth')
    app.register_blueprint(doctor_bp,  url_prefix='/doctor')
    app.register_blueprint(patient_bp, url_prefix='/patient')

    try:
        from routes.api import api_bp
        app.register_blueprint(api_bp, url_prefix='/api')
    except ImportError:
        pass

    try:
        from routes.sandbox import sandbox_bp
        app.register_blueprint(sandbox_bp, url_prefix='/sandbox')
    except ImportError:
        pass

    # ── Root route ────────────────────────────────────────────────────────────
    @app.route('/')
    def index():
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        if os.path.exists(os.path.join(template_dir, 'index.html')):
            return render_template('index.html')
        return redirect(url_for('auth.login'))

    # ── DB tables ─────────────────────────────────────────────────────────────
    with app.app_context():
        db.create_all()

    # ── BUG FIX: load ML assets at startup ───────────────────────────────────
    # Previously load_all_models() was NEVER called, so _train_reports,
    # _train_labels, FAISS indices, and the Model E classifier were all None
    # at runtime. This is the single line that was missing.
    from models.inference import load_all_models
    load_all_models(app)

    return app
