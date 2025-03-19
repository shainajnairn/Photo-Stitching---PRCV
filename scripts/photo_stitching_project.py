import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image

# Adjust paths
current_dir = os.path.dirname(os.path.abspath(__file__))
r2d2_parent_dir = r"E:/NEU/Academics/PRCV/PRCV_FINAL_PROJECT/photo_stitching_prcv/r2d2"
dataset_path = r"E:/NEU/Academics/PRCV/PRCV_FINAL_PROJECT/photo_stitching_prcv/UDIS-D"
input1_path = os.path.join(dataset_path, 'testing', 'input1')
input2_path = os.path.join(dataset_path, 'testing', 'input2')
output_dir = os.path.join(current_dir, '..', 'stitched_results')

# Add necessary paths for R2D2
sys.path.append(r2d2_parent_dir)
sys.path.append(os.path.join(r2d2_parent_dir, 'r2d2'))
sys.path.append(os.path.join(r2d2_parent_dir, 'r2d2', 'tools'))

from extract import load_network, extract_multiscale, NonMaxSuppression
from tools.dataloader import norm_RGB

# Load the R2D2 model
model_path = r"E:/NEU/Academics/PRCV/PRCV_FINAL_PROJECT/photo_stitching_prcv/r2d2/models/custom_UDIS-D_trained_prcv.pt"
model = load_network(model_path)

def extract_features(img, model):
    """
    Extract keypoints and descriptors from the given image using R2D2 model.
    """
    detector = NonMaxSuppression(rel_thr=0.7, rep_thr=0.7)
    img_tensor = norm_RGB(Image.fromarray(img)).unsqueeze(0)

    if torch.cuda.is_available():
        model = model.cuda()
        img_tensor = img_tensor.cuda()

    keypoints, descriptors, _ = extract_multiscale(model, img_tensor, detector)
    return keypoints.cpu().numpy(), descriptors.cpu().numpy()

def normalize_brightness(img1, img2):
    """
    Normalize brightness between two images to reduce visible differences.
    """
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mean1, mean2 = np.mean(img1_gray), np.mean(img2_gray)
    scale = mean2 / mean1
    img1 = cv2.convertScaleAbs(img1, alpha=scale, beta=0)
    return img1, img2

def crop_black_borders(image):
    """
    Crop black borders from the stitched image by finding the bounding box of valid pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def stitch_images(img1, img2, model):
    """
    Stitch two images using their extracted keypoints and descriptors.
    """
    # Normalize brightness
    img1, img2 = normalize_brightness(img1, img2)

    # Extract features
    img1_kps, img1_desc = extract_features(img1, model)
    img2_kps, img2_desc = extract_features(img2, model)

    # Match features using BFMatcher with Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(img1_desc, img2_desc, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Lower ratio for stricter matching
            good_matches.append(m)

    print(f"Good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < 4:
        raise ValueError("Not enough good matches to calculate homography.")

    # Extract matching keypoints
    src_pts = np.float32([img1_kps[m.queryIdx][:2] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([img2_kps[m.trainIdx][:2] for m in good_matches]).reshape(-1, 1, 2)

    # Calculate homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the first image to align with the second
    warped_img1 = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

    # Blend images at the seam
    overlap_width = img2.shape[1] // 10
    blended_img = blend_images(warped_img1, img2, overlap_width)

    # Crop black borders from the stitched result
    cropped_img = crop_black_borders(blended_img)

    return cropped_img

def blend_images(img1, img2, overlap_width):
    """
    Blend the overlapping region of two images using weighted average.
    """
    height, width = img2.shape[:2]
    blend = np.zeros_like(img1)
    blend[:height, :width] = img2

    for i in range(overlap_width):
        alpha = i / overlap_width
        blend[:height, width - overlap_width + i] = cv2.addWeighted(
            img2[:, width - overlap_width + i],
            1 - alpha,
            img1[:height, width - overlap_width + i],
            alpha,
            0
        )

    # Fill remaining region with warped image
    blend[:height, width:] = img1[:height, width:]
    return blend

def process_dataset(input1_path, input2_path, output_dir, model):
    """
    Process the dataset, stitch images with matching filenames, and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    img1_files = sorted(os.listdir(input1_path))
    img2_files = sorted(os.listdir(input2_path))

    # Find common filenames in both folders
    common_files = set(img1_files) & set(img2_files)

    for img_file in sorted(common_files):
        img1 = cv2.imread(os.path.join(input1_path, img_file))
        img2 = cv2.imread(os.path.join(input2_path, img_file))

        if img1 is None or img2 is None:
            print(f"Error reading {img_file}. Skipping...")
            continue

        try:
            stitched_img = stitch_images(img1, img2, model)
            output_path = os.path.join(output_dir, f'stitched_{img_file}')
            cv2.imwrite(output_path, stitched_img)
            print(f"Stitched image saved to {output_path}")
        except ValueError as e:
            print(f"Error stitching {img_file}: {e}")

if __name__ == "__main__":
    process_dataset(input1_path, input2_path, output_dir, model)
