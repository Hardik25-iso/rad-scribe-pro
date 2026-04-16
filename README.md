# Rad-Scribe Pro

**AI-Powered Chest X-Ray Report Generation System**

> **Symbiosis Institute of Technology, Pune** · Department of AI & ML · B.Tech Semester IV Academic Project

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0069FF?logo=facebook&logoColor=white)](https://github.com/facebookresearch/faiss)

---

## Overview

**Rad-Scribe Pro** is a full-stack web application that assists radiologists and healthcare professionals by generating automated chest X-ray reports using multiple deep learning models. The system employs a hybrid architecture combining real-time inference, Retrieval-Augmented Generation (RAG), and pre-computed model outputs to provide clinicians with AI-driven diagnostic suggestions.

### Key Features

- **Multi-Model Architecture**: 5 distinct AI models (A through E) with varying approaches
- **Role-Based Access**: Separate portals for doctors and patients
- **RAG-Powered Retrieval**: FAISS vector search against 4,308 IU X-Ray training cases
- **Sentence-Level Confidence**: Per-sentence confidence scoring with low-confidence flagging
- **Secure Image Handling**: Authenticated upload and viewing of medical images
- **Asynchronous Processing**: Background inference threads for non-blocking uploads

---

## Team & Mentorship

| Team Members | Mentor |
|-------------|--------|
| Tejas Kale | Dr. Zulfikar Ali Ansari |
| Hardik Gulati | |
| Swaraj Deogirkar | |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAD-SCRIBE PRO ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   Doctor     │     │   Patient    │     │   Auth       │                │
│  │   Portal     │     │   Portal     │     │   Routes     │                │
│  │  (Flask BP)  │     │  (Flask BP)  │     │  (Flask BP)  │                │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘                │
│         │                    │                    │                         │
│         └────────────────────┼────────────────────┘                         │
│                              │                                              │
│                     ┌────────▼────────┐                                     │
│                     │  Flask App Core │                                     │
│                     │  (app_factory)  │                                     │
│                     └────────┬────────┘                                     │
│                              │                                              │
│         ┌────────────────────┼────────────────────┐                         │
│         │                    │                    │                         │
│  ┌──────▼───────┐   ┌────────▼────────┐  ┌───────▼───────┐                │
│  │  SQLAlchemy  │   │  Inference      │  │  FAISS        │                │
│  │  Database    │   │  Engine         │  │  Index        │                │
│  │  (MySQL)     │   │  (PyTorch)      │  │  (4308 vec)   │                │
│  └──────────────┘   └────────┬────────┘  └───────────────┘                │
│                              │                                              │
│                    ┌─────────▼─────────┐                                    │
│                    │   ML Models       │                                    │
│                    ├───────────────────┤                                    │
│                    │ Model C: EffNet-B3│                                    │
│                    │ Model D: ViT+Dense│                                    │
│                    │ Model E: EffNet-C │                                    │
│                    │ Model B: BioGPT   │                                    │
│                    │ Model A: LSTM     │                                    │
│                    └───────────────────┘                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Five Models

| Model | Architecture | Status | Description |
|-------|-------------|--------|-------------|
| **Model A** | LSTM Baseline | Simulated | Fragment-based LSTM output showing baseline quality |
| **Model B** | BioGPT + EfficientNet | Pre-computed | 200 real outputs from fine-tuned BioGPT model |
| **Model C** | EfficientNet-B3 + RAG | **Live Inference** | Frozen encoder with FAISS retrieval against IU X-Ray index |
| **Model D** | ViT-B/16 + DenseNet-121 | **Live Retrieval** | Dual-encoder with counterfactual RAG (optional timm dependency) |
| **Model E** | EfficientNet-B3 Classifier | **Live Inference** | Trained 3-class classifier (Normal/Abnormal/Unclear) |

### Model Details

#### Model C — RAG-Powered Report Generation
- **Encoder**: EfficientNet-B3 (frozen, torchvision)
- **Index**: FAISS IndexFlatIP, 1536-dimensional, 4,308 vectors
- **Retrieval**: 3 supportive + 1 counterfactual nearest neighbors
- **Output**: Real IU X-Ray reports from training set

#### Model D — Dual-Encoder Retrieval
- **Architecture**: ViT-B/16 + DenseNet-121 fusion (requires `timm`)
- **Fallback**: EfficientNet-B3 encoder (already loaded for Model C)
- **Index**: 1024-dimensional FAISS index
- **Retrieval**: Counterfactual examples for clinical reasoning

#### Model E — Verified Classification
- **Checkpoint**: `classifier_best.pth` (44 MB)
- **Classes**: Abnormal (0), Normal (1), Unclear (2)
- **Output**: 3-class softmax probabilities + confidence scores

---

## Tech Stack

### Backend
- **Framework**: Flask 3.0.0 with Application Factory pattern
- **Database**: MySQL with SQLAlchemy ORM
- **Authentication**: Flask-Login + Flask-Bcrypt
- **ML Engine**: PyTorch 2.0+, torchvision, FAISS
- **Image Processing**: Pillow, NumPy

### Frontend
- **Templates**: Jinja2 with custom CSS
- **Styling**: Google Fonts (Source Serif 4, Nunito Sans)
- **UI Components**: Role-based dashboards, polling for async status

### Infrastructure
- **Deployment**: Gunicorn WSGI server
- **Environment**: python-dotenv for configuration

---

## Installation

### Prerequisites

```bash
Python 3.10+
MySQL Server 8.0+
```

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Hardik25-iso/rad-scribe-pro.git
cd Radscribe_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Database Configuration

```bash
# Create MySQL database
mysql -u root -p
CREATE DATABASE radscribedb;
CREATE USER 'radscribe'@'localhost' IDENTIFIED BY 'root';
GRANT ALL PRIVILEGES ON radscribedb.* TO 'radscribe'@'localhost';
FLUSH PRIVILEGES;
```

### Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=radscribe-dev-secret-2025
DB_USER=radscribe
DB_PASS=your_password
DB_HOST=localhost
DB_NAME=radscribedb
```

### Model Files Setup

The ML assets are stored in `model_files/`:

```
model_files/
├── models/           # Model checkpoints (scores_a.json, scores_b.json)
├── index/            # FAISS index for Model C (faiss.index, embeddings.npy)
├── index_d/          # FAISS index for Model D (faiss_d.index, embeddings_d.npy)
├── results/          # Pre-computed predictions (model_*_predictions.json)
├── model_e/          # Model E classifier (classifier_best.pth)
└── training_notebooks/  # Jupyter notebooks from training phase
```

> **Note**: Due to file size constraints, some model weights and FAISS indices may need to be downloaded separately or generated from the training notebooks.

---

## Running the Application

### Development Server

```bash
python wsgi.py
# or
flask run
```

Access the application at `http://localhost:5000`

### Production Deployment

```bash
gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 4
```

For Render/Heroku deployment, ensure `DATABASE_URL` is set in environment.

---

## Usage Guide

### For Doctors

1. **Register**: Navigate to `/auth/register/doctor` and provide:
   - Hospital email
   - License number (unique verification)
   - Specialization, hospital, city

2. **Upload X-Ray**: From dashboard, click "New Scan":
   - Select patient (optional)
   - Upload JPEG/PNG X-ray image
   - Choose which models to run (default: all)
   - Add optional clinical notes

3. **View Results**: After processing (~30-60 seconds):
   - Compare outputs from all 5 models
   - View sentence-level confidence scores
   - See retrieved similar cases from IU X-Ray dataset
   - Download verified reports

### For Patients

1. **Register**: Use hospital-provided Patient ID
2. **View Reports**: Access your scan history and AI-generated reports
3. **Plain Language**: Reports are displayed in understandable format

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/auth/login` | Login page |
| POST | `/auth/login` | Authenticate user |
| GET | `/auth/register/doctor` | Doctor registration form |
| POST | `/auth/register/doctor` | Create doctor account |
| GET | `/auth/register/patient` | Patient registration form |
| POST | `/auth/register/patient` | Create patient account |
| GET | `/auth/logout` | Logout current user |

### Doctor Portal
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/doctor/dashboard` | Doctor's dashboard |
| GET | `/doctor/new-scan` | Upload new X-ray |
| POST | `/doctor/new-scan` | Process X-ray upload |
| GET | `/doctor/scan/<id>` | View scan results |
| GET | `/doctor/scan/<id>/status` | Poll inference status |
| POST | `/doctor/scan/<id>/delete` | Delete a scan |
| GET | `/doctor/uploads/<path>` | Serve uploaded images |

### Patient Portal
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/patient/dashboard` | Patient's dashboard |
| GET | `/patient/report/<id>` | View specific report |

---

## Database Schema

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      User       │     │  DoctorProfile  │     │  PatientProfile │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id (PK)         │────<│ id (PK)         │     │ id (PK)         │
│ email           │     │ user_id (FK)    │     │ user_id (FK)    │
│ password (hash) │     │ full_name       │     │ full_name       │
│ role            │     │ license_number  │     │ patient_id      │
│ created_at      │     │ specialization  │     │ date_of_birth   │
│ is_active       │     │ hospital        │     │ mobile          │
└─────────────────┘     │ city            │     │ hospital        │
                        └─────────────────┘     └─────────────────┘
                                │                       │
                                │                       │
                                ▼                       ▼
                        ┌───────────────────────────────────────┐
                        │                 Scan                  │
                        ├───────────────────────────────────────┤
                        │ id (PK)                               │
                        │ doctor_id (FK)                        │
                        │ patient_id (FK)                       │
                        │ filename                              │
                        │ original_name                         │
                        │ uploaded_at                           │
                        │ notes                                 │
                        │ status (enum)                         │
                        └───────────────────────────────────────┘
                                        │
                                        │
                                        ▼
                        ┌───────────────────────────────────────┐
                        │                Report                 │
                        ├───────────────────────────────────────┤
                        │ id (PK)                               │
                        │ scan_id (FK)                          │
                        │ model_name (enum)                     │
                        │ generated_text                        │
                        │ clinical_label                        │
                        │ avg_log_prob                          │
                        │ retrieved_cases (JSON)                │
                        │ bleu4, rouge_l, bertscore             │
                        └───────────────────────────────────────┘
                                        │
                                        │
                                        ▼
                        ┌───────────────────────────────────────┐
                        │        ConfidenceSentence             │
                        ├───────────────────────────────────────┤
                        │ id (PK)                               │
                        │ report_id (FK)                        │
                        │ sentence                              │
                        │ avg_log_prob                          │
                        │ is_flagged                            │
                        │ position                              │
                        └───────────────────────────────────────┘
```

---

## Project Structure

```
Radscribe_backend/
├── app_factory.py          # Flask application factory
├── wsgi.py                 # WSGI entry point
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for deployment
├── .env                    # Environment variables
│
├── models/
│   ├── db_models.py        # SQLAlchemy models
│   └── inference.py        # ML inference engine
│
├── routes/
│   ├── auth.py             # Authentication routes
│   ├── doctor.py           # Doctor portal routes
│   └── patient.py          # Patient portal routes
│
├── templates/
│   ├── login.html
│   ├── register_doctor.html
│   ├── register_patient.html
│   ├── doctor_dashboard.html
│   ├── patient_dashboard.html
│   ├── new_scan.html
│   ├── view_scan.html
│   ├── all_reports.html
│   └── ...
│
├── model_files/
│   ├── models/             # Model checkpoints
│   ├── index/              # FAISS indices (Model C)
│   ├── index_d/            # FAISS indices (Model D)
│   ├── results/            # Pre-computed predictions
│   ├── model_e/            # Model E classifier
│   └── training_notebooks/ # Jupyter notebooks
│
└── uploads/                # Uploaded X-ray images
```

---

## Dataset

This project uses the **Indiana University Chest X-Ray Collection**:
- **Source**: HuggingFace — `MLforHealthcare/Indiana_University_Chest_X-ray_Collection`
- **Size**: ~7,000 image-report pairs
- **Training Split**: 4,308 samples used for FAISS index
- **Class Distribution**: ~70% Normal, ~25% Abnormal, ~5% Unclear

### Data Understanding Insights

Key findings from exploratory data analysis:
- Median report length: ~60 words (~80 tokens with GPT-2 tokenizer)
- 95%+ of reports fit within 128 tokens
- Dominant medical terms: normal, heart, lungs, clear, cardiac
- Class imbalance causes "lazy model" problem — addressed with weighted sampling

---

## Honesty Statement

This is an **academic research prototype** built for educational purposes. The system is transparent about its architecture:

- **Pre-computed outputs** (Models A, B) are clearly disclosed as such
- **Live inference** (Models C, D, E) runs real PyTorch models on uploaded images
- **FAISS retrieval** performs genuine nearest-neighbor search against real IU X-Ray training data
- **No fabricated results** — all outputs trace back to actual model predictions or retrieved cases

> **Disclaimer**: This system is NOT approved for clinical use. It is a demonstration project for academic evaluation only.

---

## Training Notebooks

The model training was conducted in Jupyter notebooks located in `model_files/training_notebooks/`:

| Notebook | Purpose |
|----------|---------|
| `01_data_understanding.ipynb` | EDA on IU X-Ray dataset |
| `02_preprocessing.ipynb` | Data cleaning and preparation |
| `03RestNet+LSTM.ipynb` | Model A (LSTM baseline) |
| `04_model_B_biogpt.ipynb` | Model B (BioGPT fine-tuning) |
| `05_Embeddings_and_Retrieval_Final.ipynb` | RAG pipeline |
| `06_Model_C.ipynb` | Model C training |
| `07_model_D_advanced_encoder.ipynb` | Model D (dual-encoder) |
| `08_model_E_verification.ipynb` | Model E classifier |

---

## Known Issues & Limitations

1. **Model D Dual-Encoder**: Requires `pip install timm` (~600 MB) for true ViT+DenseNet retrieval. Defaults to EfficientNet fallback otherwise.

2. **File Size**: Model weights and FAISS indices are large — excluded from Git repository.

3. **Database**: Currently configured for local MySQL. For production, update connection string and run migrations.

4. **Inference Time**: Background threads handle ML inference; large images may take 30-60 seconds.

---

## Future Enhancements

- [ ] DICOM image support for direct PACS integration
- [ ] Real-time collaborative report editing
- [ ] Integration with hospital EMR systems
- [ ] Mobile application for patient access
- [ ] Multi-language report generation
- [ ] Explainable AI heatmaps (Grad-CAM)

---

## License

This project is created for academic purposes at Symbiosis Institute of Technology, Pune.

**For educational and research use only. Not approved for clinical deployment.**

---

## Acknowledgments

- **Dataset**: Indiana University Chest X-Ray Collection (NIH)
- **Framework**: Flask, PyTorch, FAISS communities
- **Mentorship**: Dr. Zulfikar Ali Ansari, SIT Pune

---

## Contact

For questions or collaboration inquiries, please reach out to the development team.

**Built with ❤️ for advancing AI in healthcare**
