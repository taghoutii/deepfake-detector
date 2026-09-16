import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class BinaryOutputTarget:
    """
    Custom Grad-CAM target for single-neuron binary classifiers.
    If predicting fake (prob < 0.5): negate the output so gradients
    flow toward the fake direction.
    If predicting real (prob >= 0.5): use output as-is.
    """
    def __init__(self, is_fake: bool):
        self.is_fake = is_fake

    def __call__(self, model_output):
        output = model_output[0]   # single neuron → scalar
        return -output if self.is_fake else output

def get_gradcam_image(model, pil_image: Image.Image, device: str) -> Image.Image:
    tensor        = TRANSFORM(pil_image).unsqueeze(0).to(device)
    img_resized   = pil_image.resize((224, 224))
    img_array     = np.array(img_resized).astype(np.float32) / 255.0
    target_layers = [model.features[-1]]

    # Get prediction first
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()

    is_fake = prob < 0.5
    targets = [BinaryOutputTarget(is_fake)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)