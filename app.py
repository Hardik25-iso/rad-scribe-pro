"""
Rad-Scribe Pro — Flask Backend
Symbiosis Institute of Technology, Pune
B.Tech AI & ML — Semester IV PBL

Run:
    python app.py

Requires:
    pip install flask flask-sqlalchemy flask-bcrypt flask-login pymysql
    pip install torch torchvision transformers faiss-cpu sacremoses
    pip install pillow numpy
"""

import os
from app_factory import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
