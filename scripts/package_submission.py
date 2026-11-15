"""Automated packaging and verification utility for UMS university submission."""

import os
import sys
import zipfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

MANDATORY_DELIVERABLES = [
    "PROJECT_LINKS.txt",
    "README.md",
    "requirements.txt",
    "notebooks/weather_clustering_analysis.ipynb",
    "data/processed/processed_weather_data.csv",
    "models/kmeans_champion.joblib",
    "models/cluster_profiles.json",
    "models/training_metrics.json",
    "app/app.py",
    "src/data_loader.py",
    "src/feature_engineering.py",
    "src/train.py",
    "src/evaluate.py",
    "src/utils.py",
    "tests/test_pipeline.py",
    "docs/PRESENTATION.pptx",
    "docs/PROJECT_REPORT.pdf",
    "docs/VIVA_QUESTIONS_AND_ANSWERS.pdf",
    "docs/viva_questions_and_answers.md",
    "docs/architecture_and_metrics.md",
    "docs/report_content.md",
    "docs/ppt_structure.md",
]

EXCLUDE_DIRS = {".git", "venv", "__pycache__", ".pytest_cache", ".github"}
EXCLUDE_SUBSTRINGS = ["data/raw"]


def verify_deliverables() -> bool:
    """Check that all required files exist prior to building ZIP archive."""
    missing = []
    for item in MANDATORY_DELIVERABLES:
        full_path = project_root / item
        if not full_path.exists():
            missing.append(item)
    if missing:
        print(f"[ERROR] Missing mandatory deliverables: {missing}")
        return False
    print(f"[SUCCESS] All {len(MANDATORY_DELIVERABLES)} mandatory deliverables verified.")
    return True


def build_submission_zip(output_name: str = "12523695.zip") -> Path:
    """Build standardized, university-compliant submission ZIP file."""
    zip_path = project_root / output_name
    print(f"Compiling submission archive: {zip_path}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(project_root):
            rel_dir = os.path.relpath(dirpath, project_root)
            parts = Path(rel_dir).parts

            if any(p in EXCLUDE_DIRS for p in parts):
                continue
            if len(parts) >= 2 and parts[0] == "data" and parts[1] == "raw":
                continue

            for f in filenames:
                if f == output_name or f.endswith(".zip") or f == ".DS_Store":
                    continue
                full_file = os.path.join(dirpath, f)
                rel_file = os.path.join(rel_dir, f) if rel_dir != "." else f
                z.write(full_file, arcname=rel_file)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Archive generated: {zip_path.name} ({size_mb:.2f} MB)")
    return zip_path


if __name__ == "__main__":
    if not verify_deliverables():
        sys.exit(1)
    build_submission_zip()
