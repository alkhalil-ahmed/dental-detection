"""
utils/detector.py
Wraps deep learning inference for dental disease detection.
Place your trained model at  models/best.pt
"""

import os
import uuid
import sys
import cv2
import torch
import numpy as np
from pathlib import Path


# ── Disease metadata ─────────────────────────────────────────────────────────
DISEASE_INFO = {
    "caries": {
        "label": "Caries (Tooth Decay)",
        "color": (0, 80, 255),
        "severity": "Moderate",
        "description": "Bacterial infection causing demineralisation and destruction of tooth hard tissues.",
        "recommendation": "Schedule a dental appointment for filling or restoration. Maintain good oral hygiene.",
    },
    "calculus": {
        "label": "Calculus (Tartar)",
        "color": (0, 200, 255),
        "severity": "Mild–Moderate",
        "description": "Hardened plaque deposit on tooth surfaces that cannot be removed by brushing alone.",
        "recommendation": "Professional scaling/cleaning recommended. Improve daily brushing and flossing.",
    },
    "gingivitis": {
        "label": "Gingivitis",
        "color": (0, 0, 220),
        "severity": "Mild",
        "description": "Inflammation of the gums caused by plaque buildup at the gum line.",
        "recommendation": "Improve oral hygiene. Use antibacterial mouthwash. Consult dentist if persistent.",
    },
    "ulcer": {
        "label": "Mouth Ulcer",
        "color": (200, 0, 200),
        "severity": "Mild",
        "description": "Painful sore on the soft tissue inside the mouth.",
        "recommendation": "Use topical anaesthetic gels. Avoid spicy food. See a dentist if lasting > 2 weeks.",
    },
    "hypodontia": {
        "label": "Hypodontia",
        "color": (255, 128, 0),
        "severity": "Structural",
        "description": "Congenital absence of one or more teeth.",
        "recommendation": "Orthodontic / prosthodontic evaluation required for treatment planning.",
    },
    "filling": {
        "label": "Fillings",
        "color": (50, 180, 50),
        "severity": "Existing Restoration",
        "description": "A pre-existing dental filling used to restore a tooth damaged by decay.",
        "recommendation": "Monitor for wear or fracture. Routine dental check-up recommended.",
    },
    "implant": {
        "label": "Dental Implant",
        "color": (255, 180, 0),
        "severity": "Existing Restoration",
        "description": "An artificial tooth root (implant) placed in the jaw to support a crown or bridge.",
        "recommendation": "Maintain good oral hygiene around the implant. Regular check-ups advised.",
    },
    "impacted": {
        "label": "Impacted Tooth",
        "color": (0, 140, 255),
        "severity": "Moderate–Severe",
        "description": "A tooth that has failed to erupt fully into its correct position, often a wisdom tooth.",
        "recommendation": "Consult an oral surgeon. Extraction may be required to prevent crowding or infection.",
    },
    "periapical": {
        "label": "Periapical Lesion",
        "color": (0, 60, 200),
        "severity": "Severe",
        "description": "An infection or cyst at the root tip of a tooth, often visible on X-ray.",
        "recommendation": "Root canal treatment or extraction is typically required. See a dentist promptly.",
    },
    "crown": {
        "label": "Dental Crown",
        "color": (180, 100, 255),
        "severity": "Existing Restoration",
        "description": "A cap placed over a damaged tooth to restore its shape, size and strength.",
        "recommendation": "Check for cracks or looseness during routine visits.",
    },
    "bridge": {
        "label": "Dental Bridge",
        "color": (255, 80, 150),
        "severity": "Existing Restoration",
        "description": "A fixed prosthetic device anchored to adjacent teeth to replace missing teeth.",
        "recommendation": "Floss under the bridge daily. Regular professional cleaning advised.",
    },
    "bone loss": {
        "label": "Bone Loss",
        "color": (30, 30, 200),
        "severity": "Severe",
        "description": "Reduction in jawbone density, commonly associated with periodontal disease.",
        "recommendation": "Periodontal treatment required. Consult a periodontist urgently.",
    },
}

DEFAULT_COLOR = (0, 200, 0)   # green fallback for unlisted classes


def _get_disease_meta(class_name: str) -> dict:
    key = class_name.lower().strip()
    for k, v in DISEASE_INFO.items():
        if k in key or key in k:
            return v
    return {
        "label": class_name.title(),
        "color": DEFAULT_COLOR,
        "severity": "Unknown",
        "description": "Detected dental finding.",
        "recommendation": "Please consult a qualified dentist for proper evaluation.",
    }


# ── Detector class ────────────────────────────────────────────────────────────
class DentalDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.40, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.model_type = None   # "yolov5" | "ultralytics"
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[WARNING] Model not found at '{self.model_path}'. "
                  "Place your best.pt inside the models/ folder. "
                  "A placeholder detector will be used until the model is provided.")
            self.model = None
            return

        # Try torch.hub backend first (handles models trained with ultralytics/yolov5 repo)
        try:
            import pathlib
            # Fix: model saved on Linux uses PosixPath; patch it for Windows
            pathlib.PosixPath = pathlib.WindowsPath
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=self.model_path,
                force_reload=False,
                verbose=False,
            )
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            self.model_type = "yolov5"
            print(f"[INFO] Deep learning model loaded from '{self.model_path}'")
            return
        except Exception as e1:
            print(f"[WARNING] torch.hub load failed ({e1}), trying fallback backend...")

        # Fallback: ultralytics backend
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model_type = "ultralytics"
            print(f"[INFO] Fallback deep learning model loaded from '{self.model_path}'")
        except Exception as e2:
            print(f"[ERROR] Failed to load model: {e2}")
            self.model = None

    # ── Main inference ──────────────────────────────────────────────────────
    def detect(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {"status": "error", "message": "Could not read the uploaded image."}

        if self.model is None:
            # If startup failed (missing deps/model race), retry load at runtime.
            self._load_model()

        if self.model is None:
            return self._placeholder_response(image_path, img)

        try:
            if self.model_type == "yolov5":
                results = self.model(image_path)
            else:
                results = self.model(
                    image_path,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )
            return self._build_response(image_path, img, results)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    # ── Build structured response ──────────────────────────────────────────
    def _build_response(self, image_path: str, img: np.ndarray, results) -> dict:
        annotated = img.copy()
        detections = []
        class_counts: dict[str, int] = {}

        if self.model_type == "yolov5":
            # torch.hub backend: results.pandas().xyxy[0] DataFrame
            pred = results.pandas().xyxy[0]
            for _, row in pred.iterrows():
                cls_name = str(row["name"])
                confidence = float(row["confidence"])
                x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                meta = _get_disease_meta(cls_name)
                color = meta["color"]
                self._draw_box(annotated, x1, y1, x2, y2, meta, confidence)
                detections.append(self._make_detection(cls_name, confidence, x1, y1, x2, y2, meta))
                class_counts[meta["label"]] = class_counts.get(meta["label"], 0) + 1
        else:
            # Ultralytics backend
            for r in results:
                for box in r.boxes:
                    cls_name = r.names[int(box.cls[0].item())]
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    meta = _get_disease_meta(cls_name)
                    color = meta["color"]
                    self._draw_box(annotated, x1, y1, x2, y2, meta, confidence)
                    detections.append(self._make_detection(cls_name, confidence, x1, y1, x2, y2, meta))
                    class_counts[meta["label"]] = class_counts.get(meta["label"], 0) + 1

        # Save annotated image
        annotated_filename = f"annotated_{uuid.uuid4().hex}.jpg"
        annotated_path = os.path.join(os.path.dirname(image_path), annotated_filename)
        cv2.imwrite(annotated_path, annotated)

        return {
            "status": "ok",
            "annotated_filename": annotated_filename,
            "detections": detections,
            "summary": {
                "total": len(detections),
                "class_counts": class_counts,
                "healthy": len(detections) == 0,
            },
        }
    # ── Draw helper ────────────────────────────────────────────────────────────────
    def _draw_box(self, img, x1, y1, x2, y2, meta, confidence):
        color = meta["color"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{meta['label']}  {confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _make_detection(cls_name, confidence, x1, y1, x2, y2, meta):
        color = meta["color"]
        return {
            "class": cls_name,
            "label": meta["label"],
            "confidence": round(confidence, 4),
            "confidence_pct": f"{confidence:.1%}",
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "severity": meta["severity"],
            "description": meta["description"],
            "recommendation": meta["recommendation"],
            "color_hex": "#{:02X}{:02X}{:02X}".format(color[2], color[1], color[0]),
        }
    # ── Placeholder (no model loaded) ──────────────────────────────────────
    def _placeholder_response(self, image_path: str, img: np.ndarray) -> dict:
        annotated = img.copy()
        h, w = annotated.shape[:2]
        msg = "Place best.pt in models/ to enable detection"
        cv2.putText(annotated, msg, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2, cv2.LINE_AA)

        annotated_filename = f"annotated_{uuid.uuid4().hex}.jpg"
        annotated_path = os.path.join(os.path.dirname(image_path), annotated_filename)
        cv2.imwrite(annotated_path, annotated)

        return {
            "status": "ok",
            "annotated_filename": annotated_filename,
            "detections": [],
            "summary": {
                "total": 0,
                "class_counts": {},
                "healthy": True,
                "warning": "Model not loaded. Place your best.pt inside the models/ folder.",
            },
        }
