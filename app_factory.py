"""
app_factory.py — Flask application factory
Creates and configures the Flask app, database, login manager, and registers all blueprints.
"""

import os
from flask import Flask, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

# ── Extension instances ──────────────────────────────────────────────────────
db     = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')

    # ── Config ───────────────────────────────────────────────────────────────
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'radscribe-dev-secret-2025')

    DB_USER = os.environ.get('DB_USER', 'root')
    DB_PASS = os.environ.get('DB_PASS', 'root')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_NAME = os.environ.get('DB_NAME', 'radscribedb')
    app.config['SQLALCHEMY_DATABASE_URI'] = (
        f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    app.config['UPLOAD_FOLDER']      = os.path.join(os.path.dirname(__file__), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB
    app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'dcm'}

    BASE = os.path.dirname(__file__)
    app.config['MODEL_DIR']   = os.path.join(BASE, 'model_files', 'models')
    app.config['INDEX_DIR']   = os.path.join(BASE, 'model_files', 'index')
    app.config['INDEX_D_DIR'] = os.path.join(BASE, 'model_files', 'index_d')
    app.config['RESULTS_DIR'] = os.path.join(BASE, 'model_files', 'results')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # ── Init extensions ──────────────────────────────────────────────────────
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view     = 'auth.login'
    login_manager.login_message  = 'Please sign in to access this page.'

    # ── Register blueprints ──────────────────────────────────────────────────
    from routes.auth    import auth_bp
    from routes.doctor  import doctor_bp
    from routes.patient import patient_bp
    
    # We add prefixes here so Flask knows how to route the URLs
    app.register_blueprint(auth_bp,    url_prefix='/auth')
    app.register_blueprint(doctor_bp,  url_prefix='/doctor')
    app.register_blueprint(patient_bp, url_prefix='/patient')

    try:
        from routes.api import api_bp
        app.register_blueprint(api_bp, url_prefix='/api')
    except ImportError:
        pass # Ignore if API isn't built yet

    # ── The Front Door (Root Route) ──────────────────────────────────────────
    @app.route('/')
    def index():
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        
        # Check if index.html exists in the templates folder and serve it
        if os.path.exists(os.path.join(template_dir, 'index.html')):
            return render_template('index.html')
        else:
            return redirect(url_for('auth.login'))

    # ── Create tables ────────────────────────────────────────────────────────
    with app.app_context():
        db.create_all()

    return app