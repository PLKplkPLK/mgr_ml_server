import torch
from megadetector.detection.pytorch_detector import PTDetector
from torchvision import transforms
from PIL import Image
from torch.nn import Module


class Models():
    detector: PTDetector
    classifier: Module
    transforms: transforms.Compose


def crop_normalized_bbox_square(img: Image.Image, bbox: list[float]) -> Image.Image:
    """
    img: PIL.Image opened image
    bbox: list [x, y, w, h], normalized 0-1
    returns cropped PIL.Image as square
    """
    W, H = img.size
    x, y, w, h = bbox

    # Convert normalized bbox to pixel coords
    left = int(x * W)
    top = int(y * H)
    right = int((x + w) * W)
    bottom = int((y + h) * H)

    # Original width/height in pixels
    bw = right - left
    bh = bottom - top

    # Determine square side
    side = max(bw, bh)

    # Compute center of bbox
    cx = left + bw // 2
    cy = top + bh // 2

    # Recompute square boundaries
    half = side // 2
    new_left = cx - half
    new_top = cy - half
    new_right = new_left + side
    new_bottom = new_top + side

    # Clamp to image boundaries
    new_left = max(0, new_left)
    new_top = max(0, new_top)
    new_right = min(W, new_right)
    new_bottom = min(H, new_bottom)

    return img.crop((new_left, new_top, new_right, new_bottom))

def predict_batch(model, pil_images: list[Image.Image], transform: transforms.Compose,
                  class_names: list[str], top_k: int=5):
    """
    pil_images: list of PIL.Image
    returns: list of list of (classname, prob)
    """
    model.eval()

    # Transform images → stack into a batch
    xs = [transform(im) for im in pil_images]  # list of tensors (3, 480, 480)
    x = torch.stack(xs).to('cuda')                   # (B, 3, 480, 480)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    top_probs, top_idxs = probs.topk(top_k, dim=1)

    results = []
    for i in range(len(pil_images)):
        r = []
        for p, idx in zip(top_probs[i], top_idxs[i]):
            r.append((class_names[idx], float(p.item())))
        results.append(r)

    return results