"""
subtype 분류(2단계: is_lesion=True로 판정된 인스턴스만 대상으로 EDH/IPH/IVH/SAH/SDH 중 하나로 분류) 스키마 + prompt.

vlm_classifier_5cls.ipynb의 build_prompt()에서 병변 정의 부분만 가져오고,
is_lesion 판정(NONE 옵션)은 뺐다 — 이 단계는 이미 병변으로 확정된 인스턴스만 받는다는 전제.
"""

from enum import Enum

from pydantic import BaseModel

from src.vlm.classify.client import pil_to_base64_url


class ICHSubtype(str, Enum):
    edh = "EDH"
    iph = "IPH"
    ivh = "IVH"
    sah = "SAH"
    sdh = "SDH"


class ICHSubtypeResult(BaseModel):
    lesion_type: ICHSubtype


def zeroshot_prompt() -> str:
    return """
    You are Reader #2, providing an independent second read of a brain hemorrhage finding
    from a non-contrast head CT, already confirmed to be a true lesion. Your task is only to
    classify its subtype - do not question whether it is a real hemorrhage.

    You are given two images of the region.
    Image 1: the full axial slice, with the region highlighted, showing its location relative
    to the skull/ventricles/brain surface.
    Image 2: a zoomed-in crop of the region, showing its shape and density pattern in detail.

    Definitions:
    - EDH (epidural): biconvex/lens-shaped, sharply demarcated.
      Limited by cranial sutures (does NOT cross them) - but CAN cross midline/dural reflections (falx, venous sinuses).
      Usually hyperdense, homogeneous.

    - SDH (subdural): crescent-shaped, follows brain surface, CAN cross suture lines,
      but is limited by falx cerebri, tentorium, and falx cerebelli (won't cross midline via falx).
      Shape may become biconvex in chronic stage.
      Density highly variable (hyper/iso/hypodense possible) -
      do NOT rule out SDH based on density alone.
      If isodense/subtle: rely on mass effect / sulcal effacement instead.

    - SAH (subarachnoid): thin, follows sulci/cisterns/fissures - NOT a round/focal mass.
      Often bilateral/diffuse, fragmented; commonly in interhemispheric fissure,
      sylvian fissure, or basal cisterns (around circle of Willis).
      Small-volume SAH can be subtle - check sulcal/cisternal effacement, not density alone.

    - IVH (intraventricular): confined within ventricle boundaries, tends to pool
      dependently (occipital horns). Do NOT confuse with choroid plexus calcification,
      which is a normal finding in the same region.

    - IPH (intraparenchymal): within brain tissue, not confined to a thin space.
      Common locations: basal ganglia/thalamus/pons/cerebellum.
      May show surrounding hypodense edema and can extend into ventricles (IVH) or subarachnoid space (SAH).

    Multiplicity notes (a single slice can contain several separate instances of the same subtype):
    - SAH commonly appears in multiple, separate sulci at once.
    - IVH is often distributed across both lateral ventricles.
    - Traumatic IPH frequently presents as multiple scattered contusions rather than one lesion.
    - SDH and EDH are usually a single lesion, but can appear as separate instances when bilateral, or located at different dural attachment sites (falx, tentorium).

    Classify this confirmed hemorrhage into exactly one of: EDH, SDH, SAH, IVH, IPH.

    STRICT OUTPUT RULES: Do NOT write any analysis, explanation, or text outside the JSON. Output ONLY the JSON object below, with nothing before or after it.
    {"lesion_type": "EDH/SDH/SAH/IVH/IPH"}
    """


def fewshot_prompt() -> str:
    raise NotImplementedError("subtype fewshot prompt 미작성")


def build_messages(full_img, crop_img, prompt_fn=zeroshot_prompt) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Classify the subtype of this confirmed hemorrhage lesion."},
            {"type": "text", "text": "Image 1 (full slice):"},
            {"type": "image_url", "image_url": {"url": pil_to_base64_url(full_img)}},
            {"type": "text", "text": "Image 2 (zoomed crop):"},
            {"type": "image_url", "image_url": {"url": pil_to_base64_url(crop_img)}},
            {"type": "text", "text": prompt_fn()},
        ],
    }]
