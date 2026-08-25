"""
is_lesion 판정(1단계: 후보 영역이 진짜 병변인지 여부만 판단) 스키마 + prompt.

FP 카테고리는 아래 4편을 종합:
    Kundisch A, et al. "Deep learning algorithm in detecting intracranial
        hemorrhages on emergency computed tomographies." PLOS ONE 2021.
        (빈도순: calcification > beam-hardening artifact > tumor > blood vessel)
    Gaizo/Osborne, et al. "Deep Learning to Detect Intracranial Hemorrhage in a
        National Teleradiology Program and the Impact on Interpretation Time."
        Radiology: AI 2024. (Figure 4: infarct, ventricle horn/temporal bone
        partial volume averaging, beam-hardening, metal streak artifact)
    "Utilization of Artificial Intelligence-based Intracranial Hemorrhage
        Detection on Emergent Noncontrast CT Images in Clinical Workflow."
        Radiology: AI 2022. (postop/postischemic defect 23.6% > artifact 19.7%
        > tumor 15.3%)
    Tombach F, et al. "Diagnostic performance and confidence of an optimized
        deep-learning algorithm for the detection of intracranial hemorrhages."
        Insights Imaging 2026. (A: hardening artifact, B: motion artifact,
        C: parenchymal calcification, D: malignancy with perifocal edema)
"""

from pydantic import BaseModel

from src.vlm.classify.client import pil_to_base64_url


class ICHIsLesionResult(BaseModel):
    is_lesion: bool


def zeroshot_prompt() -> str:
    return """
    You are Reader #2, providing an independent second read of a candidate brain hemorrhage finding
    from a non-contrast head CT, originally flagged by a segmentation model (Reader #1). You do not
    see Reader #1's classification - judge independently from the images alone.

    You are given two images of the flagged region.
    Image 1: the full axial slice, with the flagged region highlighted, showing its location relative
    to the skull/ventricles/brain surface.
    Image 2: a zoomed-in crop of the region, showing its shape and density pattern in detail.

    Reader #1 (the segmentation model) can make mistakes. Small or subtle true hemorrhages are common -
    do not default to "not a lesion" just because the finding is small or faint; judge based on
    location, shape, and density pattern, not size alone.

    Reader #1's false positives cluster into these categories (from published AI-ICH-detection error
    analyses, roughly ordered by how often they occur):

    1. Calcification (single most common FP cause) - two distinct kinds:
       - Physiological calcification: choroid plexus (lateral ventricle atrium), pineal gland,
         falx cerebri, habenula. Always at a fixed, predictable normal anatomic site,
         usually midline or paired/symmetric, round and well-circumscribed, small.
         
       - Pathologic parenchymal calcification: basal ganglia/thalamus (e.g. mineralizing
         microangiopathy), old granuloma, tumor-associated calcification. Located within brain
         parenchyma rather than a fixed normal site, so it can mimic IPH - but density is much
         higher than acute blood (acute hematoma is typically ~40-90 HU; calcification often
         exceeds ~100 HU and looks chalk-white), margins are sharp and homogeneous, and there is
         no surrounding edema or mass effect.

    2. Beam-hardening / streak artifact - straight or fan-shaped streaks radiating from a dense
       structure (skull base, occipital bone) or metal (dental hardware, surgical clips, foreign
       body), not confined to an anatomically plausible lesion shape.

    3. Motion artifact - blurring or double-edge/ghosting affecting a broad region or the whole
       slice, not a discrete focal finding.

    4. Partial volume averaging - apparent hyperdensity at a bone-brain interface (temporal bone,
       occipital horn of the lateral ventricle) caused by slice thickness mixing bone and brain
       signal; check if it is still present and shaped the same way on neighboring slices.

    5. Tumor / mass, with or without perifocal edema - expansile mass-like shape, may have
       vasogenic edema following white matter, different growth pattern from an acute hematoma.

    6. Postoperative / postischemic defect - craniotomy/burr-hole change, encephalomalacia, or old
       infarct (with or without secondary calcification/hemorrhagic transformation); look for an
       associated skull defect, chronic volume-loss pattern, or vascular-territory shape.

    7. Normal blood vessel - a vessel lumen or wall (sometimes calcified) with a linear/tubular
       shape that follows a vascular course rather than sitting as a discrete round/crescent focus.

    8. Pseudo-subarachnoid hemorrhage - in diffuse cerebral edema / hypoxic-ischemic injury, the
       basal cisterns and vessels can appear relatively hyperdense against the diffusely hypodense,
       swollen brain, mimicking SAH. Look for accompanying diffuse sulcal effacement and loss of
       gray-white differentiation rather than a truly hyperdense fluid collection.

    Judge whether this is a true hemorrhage lesion (is_lesion). If it instead matches one of the FP
    categories above, or any other non-hemorrhage explanation, set is_lesion to false.

    STRICT OUTPUT RULES: Do NOT write any analysis, explanation, or text outside the JSON. Output ONLY the JSON object below, with nothing before or after it.
    {"is_lesion": true/false}
    """


def fewshot_prompt() -> str:
    raise NotImplementedError("is_lesion fewshot prompt 미작성")


def build_messages(full_img, crop_img, prompt_fn=zeroshot_prompt) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Judge whether this flagged region is a true hemorrhage lesion."},
            {"type": "text", "text": "Image 1 (full slice):"},
            {"type": "image_url", "image_url": {"url": pil_to_base64_url(full_img)}},
            {"type": "text", "text": "Image 2 (zoomed crop):"},
            {"type": "image_url", "image_url": {"url": pil_to_base64_url(crop_img)}},
            {"type": "text", "text": prompt_fn()},
        ],
    }]
