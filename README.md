# FruitFresh — Fruit Shelf-Life Prediction System

A full-stack web app that predicts the freshness and remaining shelf life of
fruit from a photo. Upload an image of an apple, banana, or orange, and the
app classifies it as fresh or rotten, estimates predicted days of shelf life
remaining, and gives a confidence score and storage tip — all saved to a
per-user history you can revisit later.

## Features

- **User accounts** — register/login with hashed passwords (Flask-Login)
- **Image-based prediction** — upload a photo, get an instant freshness
  classification (fresh/rotten) across apple, banana, and orange, plus an
  estimated shelf-life-remaining figure and confidence score
- **Prediction history** — every prediction is saved per-user with the
  original image, and can be deleted later
- **Custom model training** — upload your own labeled dataset (as a ZIP) to
  retrain the underlying model from the in-app training page
- **Storage tips** — practical advice based on the detected fruit and
  freshness state

## Tech Stack

- **Backend:** Flask, Flask-SQLAlchemy, Flask-Login
- **ML:** TensorFlow/Keras (CNN image classifier), scikit-learn
- **Database:** SQLite
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Deployment:** Gunicorn (WSGI server), Render

## Project Structure

```
FRUIT_SHELF_LIFE_PREDICTION_SYSTEM/
├── app/
│   ├── __init__.py       # App factory
│   ├── auth.py           # Login/register routes
│   ├── routes.py         # Main app routes (predict, history, training)
│   ├── models.py         # SQLAlchemy models (User, FreshnessHistory)
│   ├── predictor.py      # Model loading & inference
│   ├── config.py         # App configuration
│   ├── ml_model/         # Trained model (.h5) + class labels
│   ├── static/           # CSS/JS
│   ├── templates/        # HTML templates
│   └── uploads/          # User-uploaded images (gitignored)
├── train.py               # Training script (used by /train_model)
├── prepare_dataset.py      # Dataset preprocessing helper
├── run.py                  # Local dev entry point
├── requirements.txt
└── Procfile                 # Render/Heroku-style start command
```

## Running Locally

**Requirements:** Python 3.11 or 3.12 (TensorFlow 2.18 does not support 3.13+)

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python run.py
```

Open **http://localhost:5000**, register an account, and start uploading
fruit photos.

## Deployment

This app is configured for **Render** (Python environment, not Docker):

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn run:app --bind 0.0.0.0:$PORT --timeout 120`

⚠️ **Note:** TensorFlow is a heavy dependency (500MB+ installed). Free-tier
hosts with limited RAM (e.g. Render's 512MB free plan) may struggle to load
the model reliably. A paid instance with more memory is recommended for
consistent uptime in production.

⚠️ **Note on persistence:** on most free hosting tiers, the filesystem is
ephemeral — uploaded images and the SQLite database will be wiped on every
restart/redeploy. For durable storage in production, migrate to a managed
database (e.g. Postgres) and object storage (e.g. S3) instead of local
SQLite/disk.

## License

Personal/educational project.
