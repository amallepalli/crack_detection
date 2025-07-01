from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import torch.nn.functional as F

class YOLOGradCAM:
    def __init__(self, model, target_layer_name):
        self.yolo_model = model  # Keep reference to original YOLO model
        self.model = model.model  # Get the actual PyTorch model from YOLO wrapper
        self.model.eval()
        
        print(f"Model type: {type(self.model)}")
        
        # Get target layer
        available_modules = dict(self.model.named_modules())
        if target_layer_name in available_modules:
            self.target_layer = available_modules[target_layer_name]
            print(f"Using target layer: {target_layer_name} - {type(self.target_layer)}")
        else:
            print(f"Layer '{target_layer_name}' not found. Available layers:")
            for name in list(available_modules.keys())[:20]:  # Show first 20 layers
                print(f"  {name}")
            print("  ...")
            raise ValueError(f"Target layer '{target_layer_name}' not found")
            
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def __del__(self):
        """Clean up hooks when object is destroyed"""
        try:
            if hasattr(self, 'forward_hook'):
                self.forward_hook.remove()
            if hasattr(self, 'backward_hook'):
                self.backward_hook.remove()
        except:
            pass
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        # Ensure input tensor requires grad
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Clear any existing gradients
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        # Reset stored gradients and activations
        self.gradients = None
        self.activations = None
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle YOLO output structure
        print(f"Output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"Output tuple length: {len(output)}")
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"Output[{i}] shape: {o.shape}")
                else:
                    print(f"Output[{i}] type: {type(o)}")
        
        # For YOLO segmentation models, we need to create a meaningful loss
        # The first output is typically the detection output [batch, classes+coords, anchors]
        if isinstance(output, tuple) and len(output) >= 1:
            detection_output = output[0]  # Shape: [1, 37, 8400]
            if isinstance(detection_output, torch.Tensor):
                # For segmentation, we want to maximize the detection confidence
                # Sum over all detections to get a scalar loss
                total_score = detection_output.sum()
                print(f"Detection output shape: {detection_output.shape}")
                print(f"Total score: {total_score.item()}")
            else:
                print("First output is not a tensor")
                return None
        else:
            print("Unexpected output structure")
            return None
        
        # Backward pass
        try:
            total_score.backward(retain_graph=True)
            print("Backward pass successful")
        except Exception as e:
            print(f"Backward pass error: {e}")
            return None
        
        # Check if we captured gradients and activations
        print(f"Gradients captured: {self.gradients is not None}")
        print(f"Activations captured: {self.activations is not None}")
        
        if self.gradients is None or self.activations is None:
            print("Warning: No gradients or activations captured")
            return None
        
        gradients = self.gradients
        activations = self.activations
        
        print(f"Gradients shape: {gradients.shape}")
        print(f"Activations shape: {activations.shape}")
        
        # Global average pooling of gradients
        if gradients.dim() == 4:  # [batch, channels, height, width]
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        elif gradients.dim() == 3:  # [batch, channels, spatial]
            weights = torch.mean(gradients, dim=2, keepdim=True)
        else:
            print(f"Unexpected gradient dimensions: {gradients.dim()}")
            weights = gradients
        
        print(f"Weights shape: {weights.shape}")
        
        # Weighted combination of activation maps
        if activations.dim() == 4 and weights.dim() == 4:
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
        elif activations.dim() == 3 and weights.dim() == 3:
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
        else:
            print(f"Dimension mismatch - activations: {activations.dim()}, weights: {weights.dim()}")
            # Try element-wise multiplication anyway
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        print(f"CAM shape before processing: {cam.shape}")
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze()
        
        print(f"CAM shape after squeeze: {cam.shape}")
        
        # Handle different CAM dimensions
        if cam.dim() == 0:  # Scalar
            print("Warning: CAM is a scalar, creating uniform heatmap")
            cam = torch.ones((20, 20)) * cam.item()
        elif cam.dim() == 1:  # 1D tensor
            print("Warning: CAM is 1D, reshaping to 2D")
            size = int(cam.shape[0] ** 0.5)
            if size * size == cam.shape[0]:
                cam = cam.view(size, size)
            else:
                # Create a square tensor
                side = int(cam.shape[0] ** 0.5) + 1
                padded = torch.zeros(side * side)
                padded[:cam.shape[0]] = cam
                cam = padded.view(side, side)
        
        # Resize to input size if needed
        if cam.dim() >= 2:
            target_size = (input_tensor.shape[2], input_tensor.shape[3])
            if cam.shape != target_size:
                cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                                   size=target_size, 
                                   mode='bilinear', align_corners=False)
                cam = cam.squeeze()
        
        print(f"Final CAM shape: {cam.shape}")
        
        # Normalize to 0-1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        return cam.detach().cpu().numpy()

def preprocess_image(image_path, input_size=640):
    """Preprocess image for YOLO model"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize while maintaining aspect ratio (YOLO style)
    h, w = image_rgb.shape[:2]
    scale = input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image_rgb, (new_w, new_h))
    
    # Pad to square
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # Convert to tensor and normalize
    tensor = torch.from_numpy(padded).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor, image_rgb, (scale, new_h, new_w)

def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on image"""
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to color
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    return overlayed

def detect_crack_and_spall_with_gradcam(image_path, crack_model, spalling_model, layer_name="model.22"):
    # Original detection code
    image_og = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_og, cv2.COLOR_BGR2RGB)
    H, W = image_og.shape[:2] 

    # Predict with crack model and get mask
    crack_mask = np.zeros((H, W), dtype=np.uint8)
    crack_result = crack_model.predict(image_og, verbose=False)[0]
    if crack_result.masks is not None:
        crack_mask = np.zeros(image_og.shape[:2], dtype=np.uint8)
        for m in crack_result.masks.data:
            mask = m.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            crack_mask[mask_resized > 127] = 1

    # Predict with spalling model and get mask
    spall_mask = np.zeros((H, W), dtype=np.uint8)
    spalling_result = spalling_model.predict(image_og, verbose=False)[0]
    if spalling_result.masks is not None:
        spall_mask = np.zeros(image_og.shape[:2], dtype=np.uint8)
        for m in spalling_result.masks.data:
            mask = m.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            spall_mask[mask_resized > 127] = 2

    # Combine masks
    final_mask = np.zeros((H, W), dtype=np.uint8)
    final_mask[crack_mask == 1] = 1
    final_mask[spall_mask == 2] = 2
    final_mask[(crack_mask == 1) & (spall_mask == 2)] = 3
     
    # Create mask overlay
    output_image = image_rgb.copy()
    mask_overlay = np.zeros_like(image_rgb)
    mask_overlay[final_mask == 1] = [255, 0, 0]     # Red for cracks
    mask_overlay[final_mask == 2] = [0, 255, 0]     # Green for spalling
    mask_overlay[final_mask == 3] = [255, 0, 255]   # Magenta for both
    output_image = cv2.addWeighted(image_rgb, 1.0, mask_overlay, 0.6, 0)

    # Generate Grad-CAM visualizations - try multiple layers if needed
    input_tensor, _, _ = preprocess_image(image_path)
    
    crack_cam_overlay = image_rgb
    
    print("Generating Grad-CAM for crack model...")

    crack_gradcam = YOLOGradCAM(crack_model, target_layer_name=layer_name)
    crack_cam = crack_gradcam.generate_cam(input_tensor)
    
    if crack_cam is not None:
        crack_cam_overlay = overlay_heatmap(image_rgb, crack_cam, alpha=0.5, colormap=cv2.COLORMAP_HOT)
        print(f"Successfully generated Grad-CAM with layer: {layer_name}")
    else:
        print(f"Failed to generate Grad-CAM with layer: {layer_name}")
    
    # Try different layers for spalling model
    spall_cam_overlay = image_rgb
    print("\nGenerating Grad-CAM for spalling model...")

    spall_gradcam = YOLOGradCAM(spalling_model, target_layer_name=layer_name)
    spall_cam = spall_gradcam.generate_cam(input_tensor)
    
    if spall_cam is not None:
        spall_cam_overlay = overlay_heatmap(image_rgb, spall_cam, alpha=0.5, colormap=cv2.COLORMAP_HOT)
        print(f"Successfully generated Grad-CAM with layer: {layer_name}")
    else:
        print(f"Failed to generate Grad-CAM with layer: {layer_name}")

    # Create comprehensive visualization
    plt.figure(figsize=(12, 12))
    
    # Row 1: Original and segmentation results

    plt.subplot(2, 2, 1)
    plt.title("Segmentation Results", fontsize=14, fontweight='bold')
    plt.imshow(output_image)
    plt.axis('off')
    
    # Add legend for segmentation
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', label='Crack'),
        Patch(facecolor='green', edgecolor='green', label='Spalling'),
        Patch(facecolor='magenta', edgecolor='magenta', label='Both Detected'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize='small')

    # Row 2: Grad-CAM visualizations
    plt.subplot(2, 2, 3)
    plt.title("Crack Model - Grad-CAM", fontsize=14, fontweight='bold')
    plt.imshow(crack_cam_overlay)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Spalling Model - Grad-CAM", fontsize=14, fontweight='bold')
    plt.imshow(spall_cam_overlay)
    plt.axis('off')

    # Combined heatmap visualization
    plt.subplot(2, 2, 2)
    plt.title("Attention Comparison", fontsize=14, fontweight='bold')
    if 'crack_cam' in locals() and 'spall_cam' in locals() and crack_cam is not None and spall_cam is not None:
        # Create a combined attention map
        combined_attention = np.zeros_like(image_rgb)
        crack_resized = cv2.resize(crack_cam, (W, H))
        spall_resized = cv2.resize(spall_cam, (W, H))
        
        combined_attention[:,:,0] = (crack_resized * 255).astype(np.uint8)  # Red channel for cracks
        combined_attention[:,:,1] = (spall_resized * 255).astype(np.uint8)  # Green channel for spalling
        
        combined_overlay = cv2.addWeighted(image_rgb, 0.7, combined_attention, 0.3, 0)
        plt.imshow(combined_overlay)
    else:
        plt.imshow(image_rgb)
        plt.text(0.5, 0.5, 'Grad-CAM\nNot Available', 
                transform=plt.gca().transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Usage
if __name__ == "__main__":
    # Load models
    crack_model = YOLO("C:\Programming\crack_detection\models\crack_segmentation_model_02.pt")      
    spalling_model = YOLO("C:\Programming\crack_detection\models\spalling_segmentation_model_01.pt")  
    
    image_path = r"C:\Programming\crack_detection\crack.jpg"
    
    # Run detection with Grad-CAM
    detect_crack_and_spall_with_gradcam(image_path, crack_model, spalling_model, layer_name="model.21")