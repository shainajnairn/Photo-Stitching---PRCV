import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from photo_stitching_project import stitch_images, extract_features, load_network, normalize_brightness

# Paths for the demo
input1_demo = r"E:/NEU/Academics/PRCV/PRCV_FINAL_PROJECT/photo_stitching_prcv/UDIS-D/testing/input1"
input2_demo = r"E:/NEU/Academics/PRCV/PRCV_FINAL_PROJECT/photo_stitching_prcv/UDIS-D/testing/input2"
output_demo_dir = r"E:/NEU/Academics/PRCV/PRCV_FINAL_PROJECT/photo_stitching_prcv/demo_results"
os.makedirs(output_demo_dir, exist_ok=True)

# Load the R2D2 model
model_path = r"E:/NEU/Academics/PRCV/PRCV_FINAL_PROJECT/photo_stitching_prcv/r2d2/models/custom_UDIS-D_trained_prcv.pt"
model = load_network(model_path)

# Demo pair
demo_file = "demo_image.jpg"  # Replace with a valid image name in the input folders
img1_path = os.path.join(input1_demo, demo_file)
img2_path = os.path.join(input2_demo, demo_file)

# Read images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    print("Demo images not found. Ensure the specified images exist.")
    exit()

# Normalize brightness
img1, img2 = normalize_brightness(img1, img2)

# Display input images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Image 1")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title("Image 2")
plt.show()

# Extract and display features
kps1, desc1 = extract_features(img1, model)
kps2, desc2 = extract_features(img2, model)

# Visualize keypoints
img1_kp = cv2.drawKeypoints(img1, [cv2.KeyPoint(kp[0], kp[1], 1) for kp in kps1], None, color=(0, 255, 0))
img2_kp = cv2.drawKeypoints(img2, [cv2.KeyPoint(kp[0], kp[1], 1) for kp in kps2], None, color=(0, 255, 0))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
plt.title("Image 1 Keypoints")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
plt.title("Image 2 Keypoints")
plt.show()

# Stitch the images
try:
    stitched_image = stitch_images(img1, img2, model)
    output_path = os.path.join(output_demo_dir, f'stitched_{demo_file}')
    cv2.imwrite(output_path, stitched_image)

    # Display stitched result
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
    plt.title("Stitched Image")
    plt.axis("off")
    plt.show()

    print(f"Stitched image saved at {output_path}")
except ValueError as e:
    print(f"Error during stitching: {e}")
