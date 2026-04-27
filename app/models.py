from datetime import datetime
from flask_login import UserMixin
from app import db, login_manager


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship('FreshnessHistory', backref='user', lazy=True)


class FreshnessHistory(db.Model):
    __tablename__ = 'freshness_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_filename = db.Column(db.String(255), nullable=False)
    fruit_type = db.Column(db.String(100), nullable=False)
    predicted_days = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    freshness_status = db.Column(db.String(50), nullable=False)
    storage_tip = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'image_filename': self.image_filename,
            'fruit_type': self.fruit_type,
            'predicted_days': self.predicted_days,
            'confidence': round(self.confidence * 100, 1),
            'freshness_status': self.freshness_status,
            'storage_tip': self.storage_tip,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M'),
        }
