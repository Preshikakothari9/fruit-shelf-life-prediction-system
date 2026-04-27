import os

basedir = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'fruitfresh-antigravity-secret-2026')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, '..', 'instance', 'fruitfresh.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
    MODEL_PATH = os.path.join(basedir, 'ml_model', 'fruit_model.h5')
    CLASSES_PATH = os.path.join(basedir, 'ml_model', 'classes.json')
