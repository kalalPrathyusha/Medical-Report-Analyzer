# gradcam_utils1.py

import torch
import numpy as np
import cv2
from PIL import Image


def generate_gradcam(model, image_tensor, target_layer_name="layer4", target_class=None):

    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Print available layers for user reference
    print("Available layers for Grad-CAM:")
    for name, _ in model.named_modules():
        print(name)

    # Hook the target layer
    found = False
    for name, module in model.named_modules():
        if name == target_layer_name:
            module.register_forward_hook(forward_hook)
            # Use register_full_backward_hook for future compatibility
            module.register_full_backward_hook(backward_hook)
            found = True
            break
    if not found:
        raise ValueError(f"Target layer '{target_layer_name}' not found in model. See printed layer names above.")

    # Forward pass
    output = model(image_tensor)
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()

    # Backward pass
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    # Extract gradients and activations
    grad = gradients[0][0].cpu().numpy()
    act = activations[0][0].detach().cpu().numpy()

    # Compute weights and CAM
    weights = np.mean(grad, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (224, 224))

    # Create color heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Convert tensor back to original image format
    img_np = image_tensor.squeeze().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
    img_np = np.clip(img_np, 0, 1)
    img_np = np.uint8(img_np * 255)

    # Overlay heatmap
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    return Image.fromarray(overlay)



