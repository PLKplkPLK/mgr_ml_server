import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


CROP_SIZE = 476
BACKBONE = "vit_large_patch14_dinov2.lvd142m"
class_names = [
    'badger', 'bear', 'beaver', 'bird', 'bison', 'cat', 'dog',
    'fallow deer', 'fox', 'hare', 'lynx', 'marten', 'mink',
    'moose', 'otter', 'polecat', 'raccoon', 'raccoon dog',
    'red deer', 'roe deer', 'squirrel', 'stoat', 'weasel',
    'wild boar', 'wildcat', 'wolf'
]
class_to_idx = {name: i for i, name in enumerate(class_names)}

class Deepfaune:
    def __init__(self, dfvit_weights: str):
        self.model: Model = Model()
        self.model.loadWeights(dfvit_weights)
        # transform image to form usable by network
        self.transforms = transforms.Compose([
            transforms.Resize(
                size=(CROP_SIZE, CROP_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
                max_size=None, antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                                 std=torch.tensor([0.2290, 0.2240, 0.2250]))])

    def predictOnBatch(self, batchtensor, withsoftmax=True):
        return self.model.predict(batchtensor, withsoftmax)

    # croppedimage loaded by PIL
    def preprocessImage(self, croppedimage):
        preprocessimage = self.transforms(croppedimage)
        return preprocessimage.unsqueeze(dim=0)


class Model(nn.Module):
    def __init__(self):
        """
        Constructor of model classifier
        """
        super().__init__()
        self.base_model = timm.create_model(BACKBONE, pretrained=False,
                                            num_classes=len(class_names),
                                            dynamic_img_size=True)
        print(f"Using model in resolution {CROP_SIZE}x{CROP_SIZE}")
        self.backbone = BACKBONE
        self.nbclasses = len(class_names)

    def forward(self, input):
        x = self.base_model(input)
        return x

    def predict(self, data, withsoftmax=True):
        """
        Predict on test DataLoader
        :param test_loader: test dataloader: torch.utils.data.DataLoader
        :return: numpy array of predictions without soft max
        """
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        total_output = []
        with torch.no_grad():
            x = data.to(device)
            if withsoftmax:
                output = self.forward(x).softmax(dim=1)
            else:
                output = self.forward(x)
            total_output += output.tolist()

        return np.array(total_output)

    def loadWeights(self, path: str):
        """
        :param path: path of .pt save of model
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("CUDA available" if torch.cuda.is_available()
              else "CUDA unavailable. Using CPU")

        try:
            params = torch.load(path, map_location=device, weights_only=False)
            self.base_model.load_state_dict(params['state_dict'])
        except Exception as e:
            print("Can't load checkpoint model because :\n\n " + str(e))
            raise e

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