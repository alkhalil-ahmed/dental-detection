import os
import uuid
import json
import cv2
import smtplib
import base64
from io import BytesIO
from email.message import EmailMessage
from email.utils import format_datetime, formataddr, make_msgid
from functools import wraps
from datetime import datetime, timezone
import requests as http_requests

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    # Load local .env for SMTP and other runtime settings.
    load_dotenv()

from flask import (Flask, render_template, request, jsonify,
                   url_for, redirect, flash, abort, send_file)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from werkzeug.utils import secure_filename

from app_utils.detector import DentalDetector
from app_utils.models import db, User, Patient, Detection

# ── App Configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]                  = os.path.join("static", "uploads")

from datetime import timedelta as _td
@app.template_filter("myt")
def myt_filter(dt, fmt="%Y-%m-%d %H:%M"):
    """Convert UTC datetime to Malaysia Time (UTC+8) and format it."""
    if not dt:
        return "—"
    return (dt + _td(hours=8)).strftime(fmt)

app.config["MAX_CONTENT_LENGTH"]             = 16 * 1024 * 1024
app.secret_key                               = "dental-ai-super-secret-2026"
app.config["SQLALCHEMY_DATABASE_URI"]        = "sqlite:///dental.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# ── Extensions ───────────────────────────────────────────────────────────────
db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view             = "login"
login_manager.login_message          = "Please log in to access this page."
login_manager.login_message_category = "warning"

# ── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "best.pt")
detector   = DentalDetector(model_path=MODEL_PATH)


# ── Helpers ──────────────────────────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_pending_detection(det):
    """Pending means an original image exists but findings have not been saved yet."""
    return bool(det and det.original_image and not (det.results_json or "").strip())


def attach_detection_crop_images(source_abs_path, detections):
    """Create clearer per-finding crops from the source image and attach relative paths."""
    if not detections:
        return detections

    source = cv2.imread(source_abs_path)
    if source is None:
        return detections

    h, w = source.shape[:2]
    crops_dir = os.path.join(app.config["UPLOAD_FOLDER"], "crops")
    os.makedirs(crops_dir, exist_ok=True)

    for d in detections:
        bbox = d.get("bbox") or {}
        try:
            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))
        except Exception:
            continue

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad = max(16, int(0.28 * max(bw, bh)))

        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = source[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # Upscale tiny crops for clearer tooth structures in UI and PDF.
        ch, cw = crop.shape[:2]
        min_w, min_h = 320, 220
        if cw < min_w or ch < min_h:
            scale = max(min_w / max(1, cw), min_h / max(1, ch))
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)

        # Mild sharpening keeps edges readable after upscaling.
        soft = cv2.GaussianBlur(crop, (0, 0), 1.0)
        blur = cv2.GaussianBlur(crop, (0, 0), 2.2)
        crop = cv2.addWeighted(soft, 1.35, blur, -0.35, 0)

        crop_name = f"crop_{uuid.uuid4().hex[:12]}.jpg"
        crop_abs = os.path.join(crops_dir, crop_name)
        cv2.imwrite(crop_abs, crop)
        d["crop_image"] = f"uploads/crops/{crop_name}"

    return detections


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated


def dentist_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role not in ("admin", "dentist"):
            abort(403)
        return f(*args, **kwargs)
    return decorated


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _send_via_resend(
    *,
    patient_email: str,
    sender_name: str,
    sender_email: str,
    subject: str,
    text_body: str,
    html_body: str,
    pdf_data: bytes,
    det_id: int,
):
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("RESEND_API_KEY is not configured.")

    resend_from = (os.getenv("RESEND_FROM_EMAIL", "").strip()
                    or sender_email
                    or "onboarding@resend.dev")

    payload = {
        "from": formataddr((sender_name, resend_from)),
        "to": [patient_email],
        "subject": subject,
        "text": text_body,
        "html": html_body,
        "reply_to": sender_email or resend_from,
        "attachments": [
            {
                "filename": f"DentAI-X_Clinical_Report_{det_id}.pdf",
                "content": base64.b64encode(pdf_data).decode("ascii"),
                "content_type": "application/pdf",
            }
        ],
    }

    response = http_requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if response.status_code >= 300:
        detail = ""
        try:
            detail = response.json().get("message", "")
        except Exception:
            detail = response.text[:250]
        raise RuntimeError(f"Resend send failed ({response.status_code}): {detail}")


def send_detection_report_email(det, findings) -> str:
    patient_email = (det.patient.email or "").strip() if det.patient else ""
    if not patient_email:
        raise ValueError("Patient does not have an email address.")

    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "465").strip() or "465")
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()
    smtp_use_ssl = _env_bool("SMTP_USE_SSL", default=(smtp_port == 465))
    smtp_use_tls = _env_bool("SMTP_USE_TLS", default=(smtp_port == 587))

    sender_email = os.getenv("MAIL_FROM_EMAIL", "").strip() or smtp_user
    sender_name = os.getenv("MAIL_FROM_NAME", "DentAI-X Clinic").strip() or "DentAI-X Clinic"

    # For Gmail SMTP, aligning the visible sender with authenticated account
    # helps reduce spam placement.
    if smtp_user and "gmail.com" in smtp_host.lower():
        sender_email = smtp_user

    if not smtp_host or not sender_email:
        # SMTP may be intentionally omitted when API provider is used.
        pass

    patient_name = (det.patient.name or "Patient").strip() if det.patient else "Patient"
    report_date = det.created_at.strftime("%Y-%m-%d %H:%M") if det.created_at else "N/A"
    summary_text = (det.summary_text or "No findings detected").strip()

    text_body = (
        f"Dear {patient_name},\n\n"
        "Please find attached your DentAI-X clinical report in PDF format.\n"
        f"Report ID: {det.id}\n"
        f"Exam Date: {report_date}\n"
        f"Summary: {summary_text}\n\n"
        "This report was generated as part of your dental assessment.\n"
        "If you have questions, please contact your dentist.\n\n"
        f"Regards,\n{sender_name}"
    )

    html_body = f"""
<html>
  <body style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#1f2937;line-height:1.55;">
    <p>Dear {patient_name},</p>
    <p>Please find attached your <strong>DentAI-X clinical report</strong> in PDF format.</p>
    <table style="border-collapse:collapse;margin:10px 0 14px;">
      <tr><td style="padding:4px 10px 4px 0;color:#4b5563;">Report ID:</td><td style="padding:4px 0;"><strong>{det.id}</strong></td></tr>
      <tr><td style="padding:4px 10px 4px 0;color:#4b5563;">Exam Date:</td><td style="padding:4px 0;">{report_date}</td></tr>
      <tr><td style="padding:4px 10px 4px 0;color:#4b5563;">Summary:</td><td style="padding:4px 0;">{summary_text}</td></tr>
    </table>
    <p>This report was generated as part of your dental assessment. If you have questions, please contact your dentist.</p>
    <p style="margin-top:20px;">Regards,<br><strong>{sender_name}</strong></p>
  </body>
</html>
"""

    pdf_file = build_detection_pdf(det, findings)
    pdf_file.seek(0)
    pdf_data = pdf_file.read()

    subject = f"DentAI-X Clinical Report #{det.id}"

    provider = os.getenv("EMAIL_PROVIDER", "auto").strip().lower()
    resend_configured = bool(os.getenv("RESEND_API_KEY", "").strip())

    if provider == "resend" or (provider == "auto" and resend_configured):
        _send_via_resend(
            patient_email=patient_email,
            sender_name=sender_name,
            sender_email=sender_email,
            subject=subject,
            text_body=text_body,
            html_body=html_body,
            pdf_data=pdf_data,
            det_id=det.id,
        )
        return patient_email

    if not smtp_host or not sender_email:
        raise RuntimeError(
            "Email is not configured. Set SMTP_HOST and MAIL_FROM_EMAIL (or SMTP_USER), "
            "or configure Resend with EMAIL_PROVIDER=resend and RESEND_API_KEY."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((sender_name, sender_email))
    msg["To"] = patient_email
    msg["Reply-To"] = sender_email
    msg["Date"] = format_datetime(datetime.now(timezone.utc))
    if "@" in sender_email:
        msg["Message-ID"] = make_msgid(idstring=f"detection-{det.id}", domain=sender_email.split("@", 1)[1])
    else:
        msg["Message-ID"] = make_msgid(idstring=f"detection-{det.id}")
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    msg.add_attachment(
        pdf_data,
        maintype="application",
        subtype="pdf",
        filename=f"DentAI-X_Clinical_Report_{det.id}.pdf",
    )

    if smtp_use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            if smtp_user:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            if smtp_use_tls:
                server.starttls()
            if smtp_user:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)

    return patient_email


def build_detection_pdf(det, findings):
    """Build a printable clinical PDF report for a detection record."""
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image, HRFlowable)

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title=f"DentAI-X Clinical Report #{det.id}",
    )

    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "ClinicalNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
    )
    title_style = ParagraphStyle(
        "ClinicalTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "ClinicalSubtitle",
        parent=normal,
        fontSize=9,
        textColor=colors.HexColor("#475569"),
    )
    section_style = ParagraphStyle(
        "ClinicalSection",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
    )
    label_style = ParagraphStyle(
        "ClinicalLabel",
        parent=normal,
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor("#334155"),
    )
    small_style = ParagraphStyle(
        "ClinicalSmall",
        parent=normal,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )
    finding_title_style = ParagraphStyle(
        "FindingTitle",
        parent=normal,
        fontName="Helvetica-Bold",
        fontSize=10.5,
        textColor=colors.HexColor("#0f172a"),
    )
    finding_body_style = ParagraphStyle(
        "FindingBody",
        parent=normal,
        fontSize=9.2,
        leading=12,
    )

    story = []

    # Co-brand logos strip
    logo_specs = []
    for logo_name in ("UMPSA.png", "IIUM.png"):
        logo_abs = os.path.join(app.static_folder, "images", logo_name)
        if not os.path.exists(logo_abs):
            continue
        try:
            iw, ih = ImageReader(logo_abs).getSize()
            if iw > 0 and ih > 0:
                target_h = 1.2 * cm
                target_w = (iw / ih) * target_h
                logo_specs.append(Image(logo_abs, width=target_w, height=target_h))
        except Exception:
            continue

    if logo_specs:
        logos = logo_specs[:2]
        if len(logos) == 1:
            logos_tbl = Table([logos], colWidths=[logos[0].drawWidth])
        else:
            gap = 0.55 * cm
            logos_tbl = Table(
                [[logos[0], "", logos[1]]],
                colWidths=[logos[0].drawWidth, gap, logos[1].drawWidth],
            )

        logos_tbl.hAlign = "CENTER"
        logos_tbl.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(logos_tbl)
        story.append(Spacer(1, 4))

    created_str = det.created_at.strftime("%Y-%m-%d %H:%M") if det.created_at else "N/A"
    story.append(Paragraph("DentAI-X Clinical Detection Report", title_style))
    story.append(Paragraph(f"Report ID: {det.id} &nbsp;&nbsp;|&nbsp;&nbsp; Generated: {created_str}", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 8))

    patient_name = det.patient.name if det.patient else "Unknown"
    patient_dob = det.patient.dob if (det.patient and det.patient.dob) else "N/A"
    patient_gender = det.patient.gender if (det.patient and det.patient.gender) else "N/A"
    dentist_name = det.dentist.username if det.dentist else "N/A"
    summary_text = det.summary_text or "No findings"

    story.append(Paragraph("Patient and Exam Information", section_style))
    story.append(Spacer(1, 4))

    info_data = [
        [Paragraph("Patient", label_style), Paragraph(escape(patient_name), normal), Paragraph("Dentist", label_style), Paragraph(escape(dentist_name), normal)],
        [Paragraph("DOB", label_style), Paragraph(escape(patient_dob), normal), Paragraph("Gender", label_style), Paragraph(escape(patient_gender), normal)],
        [Paragraph("Total Findings", label_style), Paragraph(str(det.total_findings or 0), normal), Paragraph("Summary", label_style), Paragraph(escape(summary_text), normal)],
    ]
    info_tbl = Table(info_data, colWidths=[2.8 * cm, 5.7 * cm, 2.8 * cm, 6.5 * cm])
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Diagnostic Images", section_style))
    story.append(Spacer(1, 4))

    image_row = []
    image_width = 8.1 * cm
    image_height = 5.0 * cm
    for filename, label in ((det.original_image, "Original"), (det.annotated_image, "Annotated")):
        if filename:
            img_path = os.path.join(app.static_folder, "uploads", filename)
            if os.path.exists(img_path):
                image_row.append(Table([
                    [Paragraph(label, label_style)],
                    [Image(img_path, width=image_width, height=image_height)],
                ], colWidths=[8.3 * cm]))
            else:
                image_row.append(Paragraph(f"{label} image not found", small_style))
        else:
            image_row.append(Paragraph(f"{label} image not available", small_style))
    image_tbl = Table([image_row], colWidths=[8.5 * cm, 8.5 * cm])
    image_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(image_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Final Findings and Dentist Advice", section_style))
    story.append(Spacer(1, 4))

    if findings:
        for i, f in enumerate(findings, start=1):
            label = escape(str(f.get("label", "") or "Unknown"))
            severity = escape(str(f.get("severity", "") or "Review"))
            confidence = escape(str(f.get("confidence_pct", "") or "N/A"))
            review = escape(str(f.get("description", "") or "Not provided")).replace("\n", "<br/>")
            advice = escape(str(f.get("recommendation", "") or "Not provided")).replace("\n", "<br/>")

            thumb_cell = Paragraph("No tooth crop", small_style)
            crop_rel = str(f.get("crop_image", "")).strip()
            if crop_rel:
                crop_abs = os.path.join(app.static_folder, *crop_rel.split("/"))
                if os.path.exists(crop_abs):
                    thumb_cell = Image(crop_abs, width=4.4 * cm, height=3.0 * cm)

            header = Paragraph(
                f"Finding {i}: {label} &nbsp;&nbsp;|&nbsp;&nbsp; Severity: {severity} &nbsp;&nbsp;|&nbsp;&nbsp; Confidence: {confidence}",
                finding_title_style,
            )
            right_text = Paragraph(
                f"<b>Review:</b> {review}<br/><br/><b>Advice:</b> {advice}",
                finding_body_style,
            )

            finding_tbl = Table(
                [[header, ""], [thumb_cell, right_text]],
                colWidths=[5.0 * cm, 12.2 * cm],
            )
            finding_tbl.setStyle(TableStyle([
                ("SPAN", (0, 0), (1, 0)),
                ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#eef2ff")),
                ("BACKGROUND", (0, 1), (1, 1), colors.HexColor("#ffffff")),
                ("GRID", (0, 0), (1, 1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (1, 1), "TOP"),
                ("ALIGN", (0, 1), (0, 1), "CENTER"),
                ("LEFTPADDING", (0, 0), (1, 1), 6),
                ("RIGHTPADDING", (0, 0), (1, 1), 6),
                ("TOPPADDING", (0, 0), (1, 1), 6),
                ("BOTTOMPADDING", (0, 0), (1, 1), 6),
            ]))
            story.append(finding_tbl)
            story.append(Spacer(1, 7))
    else:
        story.append(Paragraph("No findings detected.", normal))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Clinical note: This report supports dentist review and is not a standalone diagnosis.", small_style))

    doc.build(story)
    buf.seek(0)
    return buf


# ── Database init + default admin ────────────────────────────────────────────
def create_tables():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(role="admin").first():
            admin = User(username="admin", email="admin@dentalai.com", role="admin")
            admin.set_password("admin123")
            db.session.add(admin)
            db.session.commit()
            print("[INFO] Default admin created  →  username: admin  password: admin123")


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            if not user.is_active:
                flash("Your account has been deactivated. Contact admin.", "danger")
                return render_template("auth/login.html")
            login_user(user, remember=request.form.get("remember") == "on")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("auth/login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        role     = request.form.get("role", "dentist")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
        else:
            user = User(username=username, email=email, role=role)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for("login"))
    return render_template("auth/register.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user  = User.query.filter_by(email=email).first()
        if user:
            token = user.generate_reset_token()
            db.session.commit()
            reset_url = url_for("reset_password", token=token, _external=False)
            flash(f'Reset link ready: <a href="{reset_url}" class="alert-link">Click here to reset your password</a>', "info")
        else:
            flash("If this email exists, a reset link has been generated.", "info")
    return render_template("auth/forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user:
        flash("Invalid or expired reset link.", "danger")
        return redirect(url_for("forgot_password"))
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        else:
            user.set_password(password)
            user.reset_token = None
            db.session.commit()
            flash("Password reset successfully! Please log in.", "success")
            return redirect(url_for("login"))
    return render_template("auth/reset_password.html", token=token)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_info":
            username = request.form.get("username", "").strip()
            email    = request.form.get("email", "").strip()
            if not username or not email:
                flash("Username and email are required.", "danger")
            elif username != current_user.username and User.query.filter_by(username=username).first():
                flash("That username is already taken.", "danger")
            elif email != current_user.email and User.query.filter_by(email=email).first():
                flash("That email is already in use.", "danger")
            else:
                current_user.username = username
                current_user.email    = email
                db.session.commit()
                flash("Profile updated successfully.", "success")

        elif action == "change_password":
            current_pw = request.form.get("current_password", "")
            new_pw     = request.form.get("new_password", "")
            confirm_pw = request.form.get("confirm_password", "")
            if not current_user.check_password(current_pw):
                flash("Current password is incorrect.", "danger")
            elif len(new_pw) < 6:
                flash("New password must be at least 6 characters.", "danger")
            elif new_pw != confirm_pw:
                flash("New passwords do not match.", "danger")
            else:
                current_user.set_password(new_pw)
                db.session.commit()
                flash("Password changed successfully.", "success")

        elif action == "upload_avatar":
            file = request.files.get("avatar_file")
            if file and allowed_file(file.filename):
                ext      = file.filename.rsplit(".", 1)[1].lower()
                filename = f"avatar_{current_user.id}_{uuid.uuid4().hex[:8]}.{ext}"
                avatars_dir = os.path.join(app.config["UPLOAD_FOLDER"], "avatars")
                os.makedirs(avatars_dir, exist_ok=True)
                file.save(os.path.join(avatars_dir, filename))
                current_user.avatar = f"uploads/avatars/{filename}"
                db.session.commit()
                flash("Profile picture updated.", "success")
            else:
                flash("Invalid file. Use PNG, JPG, or JPEG.", "danger")

        return redirect(url_for("profile"))

    return render_template("profile.html")


@app.route("/dashboard")
@login_required
def dashboard():
    if current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("dentist_dashboard"))


# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/admin/dashboard")
@login_required
@admin_required
def admin_dashboard():
    total_patients   = Patient.query.count()
    total_dentists   = User.query.filter_by(role="dentist").count()
    total_detections = Detection.query.count()
    recent           = (Detection.query
                        .order_by(Detection.created_at.desc())
                        .limit(10).all())
    return render_template("admin/dashboard.html",
                           total_patients=total_patients,
                           total_dentists=total_dentists,
                           total_detections=total_detections,
                           recent=recent)


# ── User Management ───────────────────────────────────────────────────────────

@app.route("/admin/users")
@login_required
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin/users.html", users=users)


@app.route("/admin/users/add", methods=["GET", "POST"])
@login_required
@admin_required
def admin_add_user():
    if request.method == "POST":
        username  = request.form.get("username", "").strip()
        email     = request.form.get("email", "").strip().lower()
        password  = request.form.get("password", "")
        role      = request.form.get("role", "dentist")
        is_active = request.form.get("is_active") == "on"

        if not username or not email or not password:
            flash("All fields are required.", "danger")
        elif User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
        else:
            user = User(username=username, email=email, role=role, is_active=is_active)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash(f"User '{username}' created successfully.", "success")
            return redirect(url_for("admin_users"))
    return render_template("admin/user_form.html", user=None, action="Add")


@app.route("/admin/users/<int:uid>/edit", methods=["GET", "POST"])
@login_required
@admin_required
def admin_edit_user(uid):
    user = db.session.get(User, uid)
    if not user:
        abort(404)
    if request.method == "POST":
        user.username  = request.form.get("username", user.username).strip()
        user.email     = request.form.get("email", user.email).strip().lower()
        user.role      = request.form.get("role", user.role)
        user.is_active = request.form.get("is_active") == "on"
        new_pw = request.form.get("password", "").strip()
        if new_pw:
            user.set_password(new_pw)
        db.session.commit()
        flash(f"User '{user.username}' updated.", "success")
        return redirect(url_for("admin_users"))
    return render_template("admin/user_form.html", user=user, action="Edit")


@app.route("/admin/users/<int:uid>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_user(uid):
    user = db.session.get(User, uid)
    if not user:
        abort(404)
    if user.id == current_user.id:
        flash("You cannot delete your own account.", "danger")
        return redirect(url_for("admin_users"))
    db.session.delete(user)
    db.session.commit()
    flash(f"User '{user.username}' deleted.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<int:uid>/toggle", methods=["POST"])
@login_required
@admin_required
def admin_toggle_user(uid):
    user = db.session.get(User, uid)
    if not user or user.id == current_user.id:
        abort(400)
    user.is_active = not user.is_active
    db.session.commit()
    status = "activated" if user.is_active else "deactivated"
    flash(f"User '{user.username}' {status}.", "success")
    return redirect(url_for("admin_users"))


# ── Patient Management ────────────────────────────────────────────────────────

@app.route("/admin/patients")
@login_required
@admin_required
def admin_patients():
    q        = request.args.get("q", "").strip()
    query    = Patient.query
    if q:
        query = query.filter(Patient.name.ilike(f"%{q}%"))
    patients = query.order_by(Patient.created_at.desc()).all()
    return render_template("admin/patients.html", patients=patients, q=q)


@app.route("/admin/patients/add", methods=["GET", "POST"])
@login_required
@admin_required
def admin_add_patient():
    if request.method == "POST":
        name      = request.form.get("name", "").strip()
        dob       = request.form.get("dob", "").strip()
        gender    = request.form.get("gender", "")
        phone     = request.form.get("phone", "").strip()
        email     = request.form.get("email", "").strip()
        ic_number = request.form.get("ic_number", "").strip()
        notes     = request.form.get("notes", "").strip()
        if not name:
            flash("Patient name is required.", "danger")
        else:
            p = Patient(name=name, dob=dob, gender=gender,
                        phone=phone, email=email, ic_number=ic_number,
                        notes=notes, created_by=current_user.id)
            db.session.add(p)
            db.session.commit()
            flash(f"Patient '{name}' added. You can upload an image for detection now.", "success")
            return redirect(url_for("admin_patient_detail", pid=p.id))
    return render_template("admin/patient_form.html", patient=None, action="Add")


@app.route("/admin/patients/<int:pid>/edit", methods=["GET", "POST"])
@login_required
@admin_required
def admin_edit_patient(pid):
    patient = db.session.get(Patient, pid)
    if not patient:
        abort(404)
    if request.method == "POST":
        patient.name      = request.form.get("name",      patient.name).strip()
        patient.dob       = request.form.get("dob",       patient.dob)
        patient.gender    = request.form.get("gender",    patient.gender)
        patient.phone     = request.form.get("phone",     patient.phone).strip()
        patient.email     = request.form.get("email",     patient.email or "").strip()
        patient.ic_number = request.form.get("ic_number", patient.ic_number or "").strip()
        patient.notes     = request.form.get("notes",     patient.notes).strip()
        db.session.commit()
        flash(f"Patient '{patient.name}' updated.", "success")
        return redirect(url_for("admin_patients"))
    return render_template("admin/patient_form.html", patient=patient, action="Edit")


@app.route("/admin/patients/<int:pid>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_patient(pid):
    patient = db.session.get(Patient, pid)
    if not patient:
        abort(404)
    db.session.delete(patient)
    db.session.commit()
    flash(f"Patient '{patient.name}' deleted.", "success")
    return redirect(url_for("admin_patients"))


@app.route("/admin/patients/<int:pid>")
@login_required
@admin_required
def admin_patient_detail(pid):
    patient    = db.session.get(Patient, pid)
    if not patient:
        abort(404)
    detections = (Detection.query
                  .filter_by(patient_id=pid)
                  .order_by(Detection.created_at.desc()).all())
    return render_template("admin/patient_detail.html",
                           patient=patient, detections=detections)


@app.route("/admin/patients/<int:pid>/upload-image", methods=["POST"])
@login_required
@admin_required
def admin_upload_patient_image(pid):
    patient = db.session.get(Patient, pid)
    if not patient:
        abort(404)

    redirect_to = request.referrer or url_for("admin_patient_detail", pid=pid)
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Please choose an image to upload.", "warning")
        return redirect(redirect_to)
    if not allowed_file(file.filename):
        flash("Invalid image type. Use PNG, JPG, JPEG, BMP, or WEBP.", "danger")
        return redirect(redirect_to)

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], unique_name))

    pending = Detection(
        patient_id=pid,
        dentist_id=None,
        original_image=unique_name,
        annotated_image=None,
        results_json=None,
        total_findings=0,
        summary_text="Pending dentist detection",
    )
    db.session.add(pending)
    db.session.commit()

    flash(f"Image uploaded for '{patient.name}'. Dentist can now detect it.", "success")
    return redirect(redirect_to)


# ── Detection History ─────────────────────────────────────────────────────────

@app.route("/admin/history")
@login_required
@admin_required
def admin_history():
    q     = request.args.get("q", "").strip()
    query = (Detection.query
             .join(Patient)
             .filter(Detection.results_json.isnot(None))
             .order_by(Detection.created_at.desc()))
    if q:
        query = query.filter(Patient.name.ilike(f"%{q}%"))
    detections = query.all()
    return render_template("admin/history.html", detections=detections, q=q)


@app.route("/admin/history/<int:did>")
@login_required
@admin_required
def admin_detection_detail(did):
    det      = db.session.get(Detection, did)
    if not det:
        abort(404)
    findings = json.loads(det.results_json) if det.results_json else []
    return render_template("admin/detection_detail.html", det=det, findings=findings)


# ══════════════════════════════════════════════════════════════════════════════
#  DENTIST ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dentist/dashboard")
@login_required
@dentist_required
def dentist_dashboard():
    patients      = Patient.query.order_by(Patient.name).all()
    my_detections = (Detection.query
                     .filter_by(dentist_id=current_user.id)
                     .order_by(Detection.created_at.desc())
                     .limit(10).all())
    total_my = Detection.query.filter_by(dentist_id=current_user.id).count()
    return render_template("dentist/dashboard.html",
                           patients=patients,
                           my_detections=my_detections,
                           total_my=total_my)


@app.route("/dentist/patients")
@login_required
@dentist_required
def dentist_patients():
    patients = Patient.query.order_by(Patient.created_at.desc()).all()
    return render_template("dentist/patients.html", patients=patients)


@app.route("/dentist/patients/<int:pid>")
@login_required
@dentist_required
def dentist_patient_detail(pid):
    patient    = db.session.get(Patient, pid)
    if not patient:
        abort(404)
    detections = (Detection.query
                  .filter_by(patient_id=pid)
                  .order_by(Detection.created_at.desc()).all())
    return render_template("dentist/patient_detail.html",
                           patient=patient, detections=detections)


@app.route("/dentist/detect")
@login_required
@dentist_required
def dentist_detect():
    if current_user.role != "dentist":
        flash("Admins can upload patient images, but only dentists can run detection.", "warning")
        return redirect(url_for("admin_patients"))

    patients            = Patient.query.order_by(Patient.name).all()
    selected_patient_id = request.args.get("patient_id", type=int)
    source_detection_id = request.args.get("source_detection_id", type=int)
    source_detection = None

    if source_detection_id:
        source_det = db.session.get(Detection, source_detection_id)
        if not source_det:
            flash("Selected uploaded image was not found.", "danger")
        elif not is_pending_detection(source_det):
            flash("This uploaded image has already been analyzed.", "warning")
        else:
            if not selected_patient_id:
                selected_patient_id = source_det.patient_id

            if source_det.patient_id != selected_patient_id:
                flash("Selected image does not match the chosen patient.", "danger")
            elif not source_det.original_image:
                flash("Uploaded image is missing.", "danger")
            else:
                source_abs = os.path.join(app.config["UPLOAD_FOLDER"], source_det.original_image)
                if not os.path.exists(source_abs):
                    flash("Uploaded image file not found on server.", "danger")
                else:
                    source_detection = {
                        "id": source_det.id,
                        "patient_id": source_det.patient_id,
                        "original_image_name": source_det.original_image,
                        "original_image_url": url_for("static", filename=f"uploads/{source_det.original_image}"),
                    }

    email_configured = bool(
        (os.getenv("SMTP_HOST", "").strip() and
         (os.getenv("MAIL_FROM_EMAIL", "").strip() or os.getenv("SMTP_USER", "").strip()))
        or os.getenv("RESEND_API_KEY", "").strip()
    )

    return render_template("dentist/detect.html",
                           patients=patients,
                           selected_patient_id=selected_patient_id,
                           source_detection=source_detection,
                           email_configured=email_configured)


@app.route("/dentist/history/<int:did>")
@login_required
@dentist_required
def dentist_detection_detail(did):
    det      = db.session.get(Detection, did)
    if not det:
        abort(404)
    findings = json.loads(det.results_json) if det.results_json else []
    return render_template("dentist/detection_detail.html", det=det, findings=findings)


@app.route("/dentist/history/<int:did>/report.pdf")
@login_required
@dentist_required
def dentist_detection_report_pdf(did):
    det = db.session.get(Detection, did)
    if not det:
        abort(404)

    # Dentists can only export their own reports; admins can export all.
    if current_user.role != "admin" and det.dentist_id != current_user.id:
        abort(403)

    findings = json.loads(det.results_json) if det.results_json else []
    try:
        pdf_bytes = build_detection_pdf(det, findings)
    except ImportError:
        flash("PDF engine is not installed. Install reportlab to enable report export.", "danger")
        return redirect(url_for("dentist_detection_detail", did=did))

    filename = f"DentAI-X_Clinical_Report_{det.id}.pdf"
    return send_file(
        pdf_bytes,
        as_attachment=False,
        download_name=filename,
        mimetype="application/pdf",
    )


@app.route("/dentist/detections/<int:did>/save", methods=["POST"])
@login_required
@dentist_required
def dentist_save_detection_edits(did):
    if current_user.role != "dentist":
        return jsonify({"error": "Only dentists can save detection results"}), 403

    det = db.session.get(Detection, did)
    if not det:
        return jsonify({"error": "Detection not found"}), 404

    pending_before_save = is_pending_detection(det)

    # Dentist can save their own detections, or claim a pending admin-uploaded image.
    if not pending_before_save and det.dentist_id != current_user.id:
        return jsonify({"error": "You are not allowed to edit this detection"}), 403

    if pending_before_save:
        det.dentist_id = current_user.id

    payload = request.get_json(silent=True) or {}
    raw_findings = payload.get("detections")
    if not isinstance(raw_findings, list):
        return jsonify({"error": "Invalid detections payload"}), 400

    original_image = str(payload.get("original_image", "") or "").strip()
    annotated_image = str(payload.get("annotated_image", "") or "").strip()
    if original_image:
        det.original_image = original_image
    if annotated_image:
        det.annotated_image = annotated_image

    cleaned = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "")).strip() or "Unknown"
        description = str(item.get("description", "")).strip()
        recommendation = str(item.get("recommendation", "")).strip()
        confidence_pct = str(item.get("confidence_pct", "")).strip()
        severity = str(item.get("severity", "")).strip() or "Review"
        color_hex = str(item.get("color_hex", "")).strip() or "#22c55e"

        cleaned.append({
            "label": label,
            "description": description,
            "recommendation": recommendation,
            "confidence_pct": confidence_pct,
            "confidence": item.get("confidence"),
            "class": item.get("class"),
            "severity": severity,
            "color_hex": color_hex,
            "crop_image": item.get("crop_image"),
            # Keep extra optional detector fields if present.
            "class_id": item.get("class_id"),
            "bbox": item.get("bbox"),
        })

    class_counts = {}
    for f in cleaned:
        class_counts[f["label"]] = class_counts.get(f["label"], 0) + 1

    det.results_json = json.dumps(cleaned)
    det.total_findings = len(cleaned)
    det.summary_text = ", ".join(f"{k} ({v})" for k, v in class_counts.items()) if cleaned else "No findings"
    db.session.commit()

    return jsonify({
        "message": "Detection changes saved",
        "detection_id": det.id,
        "summary": {
            "total": len(cleaned),
            "class_counts": class_counts,
        },
        "detections": [
            {
                **f,
                "crop_image_url": url_for("static", filename=f["crop_image"]) if f.get("crop_image") else None,
            }
            for f in cleaned
        ],
    })


@app.route("/dentist/detections/save-new", methods=["POST"])
@login_required
@dentist_required
def dentist_save_new_detection():
    if current_user.role != "dentist":
        return jsonify({"error": "Only dentists can save detection results"}), 403

    payload = request.get_json(silent=True) or {}

    patient_id = payload.get("patient_id")
    try:
        patient_id = int(patient_id)
    except Exception:
        return jsonify({"error": "Invalid patient"}), 400

    patient = db.session.get(Patient, patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    original_image = str(payload.get("original_image", "")).strip()
    annotated_image = str(payload.get("annotated_image", "")).strip()
    raw_findings = payload.get("detections")
    if not isinstance(raw_findings, list):
        return jsonify({"error": "Invalid detections payload"}), 400

    cleaned = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "")).strip() or "Unknown"
        description = str(item.get("description", "")).strip()
        recommendation = str(item.get("recommendation", "")).strip()
        confidence_pct = str(item.get("confidence_pct", "")).strip()
        severity = str(item.get("severity", "")).strip() or "Review"
        color_hex = str(item.get("color_hex", "")).strip() or "#22c55e"

        cleaned.append({
            "label": label,
            "description": description,
            "recommendation": recommendation,
            "confidence_pct": confidence_pct,
            "confidence": item.get("confidence"),
            "class": item.get("class"),
            "severity": severity,
            "color_hex": color_hex,
            "crop_image": item.get("crop_image"),
            "class_id": item.get("class_id"),
            "bbox": item.get("bbox"),
        })

    class_counts = {}
    for f in cleaned:
        class_counts[f["label"]] = class_counts.get(f["label"], 0) + 1

    summary_text = ", ".join(f"{k} ({v})" for k, v in class_counts.items()) if cleaned else "No findings"

    det = Detection(
        patient_id=patient_id,
        dentist_id=current_user.id,
        original_image=original_image,
        annotated_image=annotated_image,
        results_json=json.dumps(cleaned),
        total_findings=len(cleaned),
        summary_text=summary_text,
    )
    db.session.add(det)
    db.session.commit()

    return jsonify({
        "message": "Detection saved to patient history",
        "detection_id": det.id,
        "detail_url": url_for("dentist_detection_detail", did=det.id),
        "summary": {
            "total": len(cleaned),
            "class_counts": class_counts,
        },
        "detections": [
            {
                **f,
                "crop_image_url": url_for("static", filename=f["crop_image"]) if f.get("crop_image") else None,
            }
            for f in cleaned
        ],
    })


@app.route("/dentist/detections/<int:did>/email-report", methods=["POST"])
@login_required
@dentist_required
def dentist_email_detection_report(did):
    det = db.session.get(Detection, did)
    if not det:
        return jsonify({"error": "Detection not found"}), 404

    if current_user.role != "admin" and det.dentist_id != current_user.id:
        return jsonify({"error": "You are not allowed to email this detection"}), 403

    findings = json.loads(det.results_json) if det.results_json else []

    try:
        sent_to = send_detection_report_email(det, findings)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to send email: {exc}"}), 500

    return jsonify({"message": f"Report sent to {sent_to}"})


# ══════════════════════════════════════════════════════════════════════════════
#  DETECT API  (called by JS fetch)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/detect", methods=["POST"])
@login_required
def detect():
    if current_user.role != "dentist":
        return jsonify({"error": "Admins can upload images, but only dentists can run detection."}), 403

    patient_id = request.form.get("patient_id", type=int)
    source_detection_id = request.form.get("source_detection_id", type=int)
    if not patient_id:
        return jsonify({"error": "No patient selected"}), 400
    patient = db.session.get(Patient, patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    source_det = None
    if source_detection_id:
        source_det = db.session.get(Detection, source_detection_id)
        if not source_det:
            return jsonify({"error": "Selected uploaded image was not found"}), 404
        if source_det.patient_id != patient_id:
            return jsonify({"error": "Selected image does not belong to this patient"}), 400
        if not is_pending_detection(source_det):
            return jsonify({"error": "This uploaded image has already been analyzed"}), 400
        if not source_det.original_image:
            return jsonify({"error": "Uploaded image is missing"}), 400

    file = request.files.get("file")
    if file and file.filename:
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # New upload from dentist: save as a fresh source image.
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)
        source_det = None
    elif source_det:
        # Use an existing pending image uploaded by admin.
        unique_name = source_det.original_image
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        if not os.path.exists(save_path):
            return jsonify({"error": "Uploaded image file not found on server"}), 404
    else:
        return jsonify({"error": "Please upload an image or select a patient image first"}), 400

    # Run detection
    result = detector.detect(save_path)
    if result["status"] == "error":
        return jsonify({"error": result["message"]}), 500

    attach_detection_crop_images(save_path, result["detections"])

    return jsonify({
        "detection_id":    source_det.id if source_det else None,
        "is_saved":        False,
        "patient_id":      patient_id,
        "source_detection_id": source_det.id if source_det else None,
        "original_image_name": unique_name,
        "annotated_image_name": result["annotated_filename"],
        "original_image":  url_for("static", filename=f"uploads/{unique_name}"),
        "annotated_image": url_for("static", filename=f"uploads/{result['annotated_filename']}"),
        "detections": [
            {
                **f,
                "crop_image_url": url_for("static", filename=f["crop_image"]) if f.get("crop_image") else None,
            }
            for f in result["detections"]
        ],
        "summary":         result["summary"],
    })


# ── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(403)
def forbidden(e):
    return render_template("errors/403.html"), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("errors/404.html"), 404


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    create_tables()

    # Diagnostic: log ALL email-related env vars at startup
    _email_vars = {k: v for k, v in os.environ.items()
                   if any(t in k.upper() for t in ("MAIL", "SMTP", "RESEND", "EMAIL"))}
    print(f"[EMAIL-CONFIG] Found {len(_email_vars)} email-related env vars:")
    for _k, _v in sorted(_email_vars.items()):
        _display = "<set>" if "KEY" in _k.upper() or "PASS" in _k.upper() else repr(_v)
        print(f"  {_k} = {_display}")
    if not _email_vars:
        print("  (none — Railway may not be injecting Service Variables)")

    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
