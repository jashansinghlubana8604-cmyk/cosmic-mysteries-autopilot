"""The four CC BY 4.0 astronomy archives (ESA/Hubble, ESA/Webb, ESO, NOIRLab): where their listings, detail
pages and files live, and how they want to be credited."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "assets" / "index"

ARCHIVES = {
    "esawebb": {
        "name": "ESA/Webb",
        "videos_list": "https://esawebb.org/videos/page/{page}/",
        "images_list": "https://esawebb.org/images/page/{page}/",
        "video_page": "https://esawebb.org/videos/{id}/",
        "image_page": "https://esawebb.org/images/{id}/",
        "image_file": "https://cdn.esawebb.org/archives/images/large/{id}.jpg",
        "credit": "ESA/Webb, NASA & CSA",
        "quality": 1.0,
    },
    "esahubble": {
        "name": "ESA/Hubble",
        "videos_list": "https://esahubble.org/videos/page/{page}/",
        "images_list": "https://esahubble.org/images/page/{page}/",
        "video_page": "https://esahubble.org/videos/{id}/",
        "image_page": "https://esahubble.org/images/{id}/",
        "image_file": "https://cdn.esahubble.org/archives/images/large/{id}.jpg",
        "credit": "ESA/Hubble & NASA",
        "quality": 0.95,
    },
    "eso": {
        "name": "ESO",
        "videos_list": "https://www.eso.org/public/videos/list/{page}/",
        "images_list": "https://www.eso.org/public/images/list/{page}/",
        "video_page": "https://www.eso.org/public/videos/{id}/",
        "image_page": "https://www.eso.org/public/images/{id}/",
        "image_file": "https://cdn.eso.org/images/large/{id}.jpg",
        "credit": "ESO",
        "quality": 0.95,
    },
    "noirlab": {
        "name": "NOIRLab",
        "videos_list": "https://noirlab.edu/public/videos/page/{page}/",
        "images_list": "https://noirlab.edu/public/images/page/{page}/",
        "video_page": "https://noirlab.edu/public/videos/{id}/",
        "image_page": "https://noirlab.edu/public/images/{id}/",
        "image_file": "https://noirlab.edu/public/media/archives/images/large/{id}.jpg",
        "credit": "NOIRLab/NSF/AURA",
        "quality": 0.85,
    },
}

LICENSE = "CC BY 4.0"
