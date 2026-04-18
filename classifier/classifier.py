import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import transforms


CROP_SIZE = 476
BACKBONE = "vit_large_patch14_dinov2.lvd142m"
class_names = [
    "badger",
    "bear",
    "beaver",
    "bird",
    "bison",
    "cat",
    "dog",
    "fallow deer",
    "fox",
    "hare",
    "lynx",
    "marten",
    "mink",
    "moose",
    "otter",
    "polecat",
    "raccoon",
    "raccoon dog",
    "red deer",
    "roe deer",
    "squirrel",
    "stoat",
    "weasel",
    "wild boar",
    "wildcat",
    "wolf",
]
class_to_idx = {name: i for i, name in enumerate(class_names)}


class Deepfaune:
    """
    Wrapper class for the Deepfaune animal classification model.
    """
    def __init__(
        self, dfvit_weights: str = "deepfaune_polish_lr4_checkpoint.pt"
    ):
        """
        Initialize the Deepfaune model.

        Args:
            dfvit_weights: Path to the model checkpoint file.
        """
        self.model: Model = Model()
        self.model.loadWeights(dfvit_weights)
        # transform image to form usable by network
        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    size=(CROP_SIZE, CROP_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    max_size=None,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                    std=torch.tensor([0.2290, 0.2240, 0.2250]),
                ),
            ]
        )

    def predictOnBatch(self, batchtensor, withsoftmax=True):
        """
        Predict on a batch of tensors.

        Args:
            batchtensor: Batch of input tensors.
            withsoftmax: Whether to apply softmax to outputs.

        Returns:
            Model predictions.
        """
        return self.model.predict(batchtensor, withsoftmax)

    # croppedimage loaded by PIL
    def preprocessImage(self, croppedimage):
        """
        Preprocess a cropped PIL image for model input.

        Args:
            croppedimage: PIL Image to preprocess.

        Returns:
            Preprocessed tensor.
        """
        preprocessimage = self.transforms(croppedimage)
        return preprocessimage.unsqueeze(dim=0)


class Model(nn.Module):
    """
    Neural network model for animal classification using ViT backbone.
    """
    def __init__(self):
        """
        Constructor of model classifier
        """
        super().__init__()
        self.base_model = timm.create_model(
            BACKBONE,
            pretrained=False,
            num_classes=len(class_names),
            dynamic_img_size=True,
        )
        print(f"Using model in resolution {CROP_SIZE}x{CROP_SIZE}")
        self.backbone = BACKBONE
        self.nbclasses = len(class_names)

    def forward(self, input):
        """
        Forward pass through the model.

        Args:
            input: Input tensor.

        Returns:
            Model output logits.
        """
        x = self.base_model(input)
        return x

    def predict(self, data, withsoftmax=True):
        """
        Perform prediction on input data.

        Args:
            data: Input tensor.
            withsoftmax: Whether to apply softmax.

        Returns:
            Numpy array of predictions.
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
        Load model weights from checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "CUDA available"
            if torch.cuda.is_available()
            else "CUDA unavailable. Using CPU"
        )

        try:
            params = torch.load(path, map_location=device, weights_only=False)
            self.base_model.load_state_dict(params["state_dict"])
        except Exception as e:
            print("Can't load checkpoint model because :\n\n " + str(e))
            raise e
