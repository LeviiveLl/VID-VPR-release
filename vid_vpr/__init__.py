"""VID-VPR: VLM injection distillation for image-only place recognition."""

from .models import VIDVPR, VLMConditionedTeacher, load_vid_vpr

__version__ = "1.0.0"
__all__ = ["VIDVPR", "VLMConditionedTeacher", "load_vid_vpr"]
