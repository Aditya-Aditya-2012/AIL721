import os
import sys
import csv
import torch
import cv2
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from utility.class_names import class_names
from utility.initialize import initialize
from models.resnet import resnet
initialize(seed=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def graphcut_segmentation(image_bgr, saliency):
    H, W = saliency.shape
    flat_vals = saliency.flatten()

    fg_thresh = np.percentile(flat_vals, 95)
    bg_thresh = np.percentile(flat_vals, 30)
    mask = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)

    mask[saliency > fg_thresh] = cv2.GC_FGD
    mask[saliency < bg_thresh] = cv2.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    grabcut_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grabcut_mask, connectivity=8)
    if num_labels <= 1:
        return grabcut_mask

    max_area = 0
    max_label = 1
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = label_id

    final_mask = (labels == max_label).astype(np.uint8)
    return final_mask

def threshold_otsu(image):
    arr = image.flatten()
    arr_scaled = (arr * 255).astype(np.uint8)
    hist, _ = np.histogram(arr_scaled, bins=256, range=(0, 256))
    total = arr_scaled.size
    sum_all = np.dot(np.arange(256), hist)
    
    sumB = 0.0   
    wB = 0.0     
    max_between = 0.0
    best_thresh = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        
        sumB += t * hist[t]
        mB = sumB / wB 
        mF = (sum_all - sumB) / wF
        
        var_between = wB * wF * (mB - mF) * (mB - mF)
        if var_between > max_between:
            max_between = var_between
            best_thresh = t

    return best_thresh / 255.0

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_image, target_class=None):
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min != 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam.zero_()
        return cam

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <model_ckpt path> <test-imgs dir>")
        sys.exit(1)
    
    model_ckpt_path = sys.argv[1]
    test_imgs_dir = sys.argv[2]
    
    seg_maps_dir = os.path.join(os.getcwd(), "seg_maps")
    os.makedirs(seg_maps_dir, exist_ok=True)

    n = 2 
    num_classes = 100
    model = resnet(n, num_classes)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    target_layer = model.layer3[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(test_imgs_dir) if f.lower().endswith(('.jpg'))]

    submission_data = []

    for image_file in image_files:
        image_path = os.path.join(test_imgs_dir, image_file)
        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        output = model(input_tensor)
        predicted_class_idx = output.argmax(dim=1).item()
        predicted_class_name = class_names[predicted_class_idx]
       
        cam = grad_cam(input_tensor, target_class=predicted_class_idx)
        cam_np = cam.cpu().numpy()[0, 0]

        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        final_mask = graphcut_segmentation(img_bgr, cam_np)
        seg_mask_uint8 = (final_mask * 255).astype(np.uint8)

        otsu_threshold = threshold_otsu(cam_np)
        seg_mask_otsu = (cam_np > otsu_threshold).astype(np.uint8) * 255

        seg_mask_img = Image.fromarray(seg_mask_otsu + seg_mask_uint8)
        seg_mask_img.save(os.path.join(seg_maps_dir, image_file))
        submission_data.append([image_file, predicted_class_name])

    csv_file = "submission.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "label"])
        writer.writerows(submission_data)

    print("Evaluation complete.")
    print(f"Submission CSV saved as {csv_file}.")
    print(f"Segmentation maps saved in folder: {seg_maps_dir}")

if __name__ == "__main__":
    main()
