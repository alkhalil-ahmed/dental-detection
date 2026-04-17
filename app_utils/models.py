"""
app_utils/models.py
SQLAlchemy database models for DentalAI
"""
from datetime import datetime
import secrets
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20),  nullable=False, default="dentist")  # admin | dentist
    is_active     = db.Column(db.Boolean, default=True)
    reset_token   = db.Column(db.String(100), nullable=True)
    avatar        = db.Column(db.String(200), nullable=True)   # relative path under static/
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    detections    = db.relationship("Detection", backref="dentist", lazy=True,
                                    foreign_keys="Detection.dentist_id")
    patients_created = db.relationship("Patient", backref="created_by_user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        return self.reset_token

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


class Patient(db.Model):
    __tablename__ = "patients"

    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(120), nullable=False)
    dob          = db.Column(db.String(20),  nullable=True)
    gender       = db.Column(db.String(10),  nullable=True)
    phone        = db.Column(db.String(30),  nullable=True)
    email        = db.Column(db.String(120), nullable=True)
    ic_number    = db.Column(db.String(50),  nullable=True)
    notes        = db.Column(db.Text,        nullable=True)
    created_by   = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    detections   = db.relationship("Detection", backref="patient", lazy=True,
                                   cascade="all, delete-orphan")

    def age(self):
        if not self.dob:
            return "N/A"
        try:
            birth = datetime.strptime(self.dob, "%Y-%m-%d")
            today = datetime.today()
            return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        except Exception:
            return "N/A"

    def __repr__(self):
        return f"<Patient {self.name}>"


class Detection(db.Model):
    __tablename__ = "detections"

    id                 = db.Column(db.Integer, primary_key=True)
    patient_id         = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    dentist_id         = db.Column(db.Integer, db.ForeignKey("users.id"),    nullable=True)
    original_image     = db.Column(db.String(256), nullable=True)
    annotated_image    = db.Column(db.String(256), nullable=True)
    results_json       = db.Column(db.Text, nullable=True)   # JSON string
    total_findings     = db.Column(db.Integer, default=0)
    summary_text       = db.Column(db.String(300), nullable=True)
    created_at         = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Detection {self.id} patient={self.patient_id}>"
