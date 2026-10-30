"""Máscara do céu e composição.

O segmentador é pré-treinado e não é treinado neste projeto: SegFormer-B0 treinado no
ADE20K, uma base com 150 classes, entre elas "sky". Os modelos do PASCAL VOC (como o
DeepLab do torchvision) não servem, porque não têm a classe céu.
"""
import torch
import torch.nn.functional as nnf
from torchvision.transforms.functional import gaussian_blur

SEGMENTER = "nvidia/segformer-b0-finetuned-ade-512-512"

_cache = {}


def _load_segmenter(device: torch.device):
    if "model" not in _cache:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        _cache["processor"] = AutoImageProcessor.from_pretrained(SEGMENTER)
        model = SegformerForSemanticSegmentation.from_pretrained(SEGMENTER).to(device).eval()
        labels = {v: int(k) for k, v in model.config.id2label.items()}
        _cache["model"], _cache["sky_id"] = model, labels["sky"]
    return _cache["processor"], _cache["model"], _cache["sky_id"]


@torch.no_grad()
def sky_mask(img: torch.Tensor, device: torch.device, blur: float = 0.01) -> torch.Tensor:
    """Imagem uint8 (3, H, W) -> máscara float (1, H, W): 1 = céu, 0 = resto.

    Usa a probabilidade de "céu" (não um 0/1 seco) e ainda suaviza a borda com um
    desfoque gaussiano, para a emenda entre céu pintado e foto não aparecer.
    `blur` é o tamanho do desfoque como fração do maior lado da imagem.
    """
    processor, model, sky_id = _load_segmenter(device)
    inputs = processor(images=img.permute(1, 2, 0).cpu().numpy(), return_tensors="pt").to(device)
    logits = model(**inputs).logits                       # (1, 150, h/4, w/4)
    logits = nnf.interpolate(logits, size=img.shape[1:], mode="bilinear", align_corners=False)
    mask = logits.softmax(1)[:, sky_id:sky_id + 1][0]     # (1, H, W)

    k = int(blur * max(img.shape[1:])) // 2 * 2 + 1      # tamanho do kernel, sempre ímpar
    if k > 1:
        mask = gaussian_blur(mask, kernel_size=k)
    return mask


def composite(original: torch.Tensor, stylized: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Estilizada onde é céu, original no resto. Onde a máscara vale 0.5, mistura meio a meio."""
    return mask * stylized + (1 - mask) * original
