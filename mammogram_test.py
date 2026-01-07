import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import flood
from skimage.transform import resize
import os                # <<< ADD THIS
import matplotlib.pyplot as plt  # <<< ADD THIS if you want to show images

#######################################################
 #Noise Removal (Median Filter 3×3)
def noise_removal_median(image):
    """
    Apply 3×3 median filtering to remove impulsive and Gaussian noise
    while preserving edges.
    """
    denoised = cv2.medianBlur(image, 3)
    return denoised



#Artifact Suppression & Background Separation
#Global Thresholding
def global_threshold(image, threshold=18):
    """
    Apply global thresholding to separate foreground from background.
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

#Connected Component Labeling
def connected_components(binary_image):
    """
    Label connected components in the binary image.
    """
    labels = measure.label(binary_image, connectivity=2)
    return labels

#Largest Object Selection (Breast Region)
def keep_largest_component(labels):
    """
    Keep only the largest connected component (assumed to be the breast).
    """
    regions = measure.regionprops(labels)

    largest_region = max(regions, key=lambda r: r.area)
    breast_mask = np.zeros_like(labels, dtype=np.uint8)
    breast_mask[labels == largest_region.label] = 255

    return breast_mask


#Mask Original Image
def apply_mask(image, mask):
    """
    Remove background and artifacts by masking the original image.
    """
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

#Pectoral Muscle Suppression
#Flip Right MLO Images

def flip_right_mlo(image, is_right_mlo):
    """
    Flip right MLO images horizontally for consistent orientation.
    """
    if is_right_mlo:
        return cv2.flip(image, 1)
    return image

#Quadrant Division
def extract_upper_left_quadrant(image):
    """
    Extract upper-left quadrant (512×512) from a 1024×1024 image.
    """
    h, w = image.shape
    return image[:h//2, :w//2]

    # Upper-Left Triangular Region Localization (LTRPM)
    
def upper_left_triangle(quadrant):
    """
    Keep only the upper-left triangular region.
    """
    h, w = quadrant.shape
    triangle_mask = np.zeros_like(quadrant, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if j <= (w - i):
                triangle_mask[i, j] = 255

    triangle = cv2.bitwise_and(quadrant, quadrant, mask=triangle_mask)
    return triangle, triangle_mask


#Seeded Region Growing (SRG)
def seeded_region_growing(triangle):
    """
    Segment the pectoral muscle using seeded region growing.
    """
    # Automatically select brightest pixel as seed
    seed_point = np.unravel_index(np.argmax(triangle), triangle.shape)

    # Normalize image for region growing
    norm_triangle = triangle / 255.0

    # Region growing using flood
    muscle_mask = flood(norm_triangle, seed_point, tolerance=0.15)
    muscle_mask = muscle_mask.astype(np.uint8) * 255

    return muscle_mask


#Remove Pectoral Muscle

def remove_pectoral_muscle(image, muscle_mask):
    """
    Remove pectoral muscle from breast image.
    """
    # Ensure proper type
    image = image.astype(np.uint8)
    muscle_mask = muscle_mask.astype(np.uint8)

    # Invert mask
    muscle_mask_inv = cv2.bitwise_not(muscle_mask)

    # Ensure same size
    if muscle_mask_inv.shape != image.shape:
        muscle_mask_inv = cv2.resize(muscle_mask_inv, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mask
    clean_image = cv2.bitwise_and(image, image, mask=muscle_mask_inv)
    return clean_image


#Crop Region of Interest (ROI)
def crop_roi(image):
    """
    Crop the bounding box around the breast region.
    """
    coords = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(coords)
    roi = image[y:y+h, x:x+w]
    return roi


#Resize All ROIs to Same Size
def resize_roi(roi, size=(512, 512)):
    """
    Resize ROI to fixed dimensions.
    """
    resized = resize(roi, size, preserve_range=True).astype(np.uint8)
    return resized

#FULL PIPELINE FUNCTION
def preprocess_mammogram(image_path, is_right_mlo=False):
    """
    Complete mammogram preprocessing pipeline.
    """
    # Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = normalize_to_uint8(image)


    # STEP 1
    image = noise_removal_median(image)

    # STEP 2
    binary = global_threshold(image)
    labels = connected_components(binary)
    breast_mask = keep_largest_component(labels)
    breast_image = apply_mask(image, breast_mask)

    # STEP 3
    breast_image = flip_right_mlo(breast_image, is_right_mlo)
    quadrant = extract_upper_left_quadrant(breast_image)
    triangle, _ = upper_left_triangle(quadrant)
    muscle_mask = seeded_region_growing(triangle)
    breast_no_muscle = remove_pectoral_muscle(breast_image, muscle_mask)

    roi = crop_roi(breast_no_muscle)
    final_image = resize_roi(roi)

    return final_image

def normalize_to_uint8(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    return image



if __name__ == "__main__":

    raw_dir = "raw"
    output_dir = "processed"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(raw_dir):

        if filename.lower().endswith(".pgm"):
            image_path = os.path.join(raw_dir, filename)

            # Set manually if images are right MLO
            is_right_mlo = True

            final_image = preprocess_mammogram(image_path, is_right_mlo)

            # Save result
            out_path = os.path.join(output_dir, filename.replace(".pgm", ".png"))
            cv2.imwrite(out_path, final_image)

            # Optional visualization (first image only)
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.title("Original")
            plt.imshow(original, cmap="gray")
            plt.axis("off")

            plt.subplot(1,2,2)
            plt.title("Preprocessed")
            plt.imshow(final_image, cmap="gray")
            plt.axis("off")

            plt.show()

            print(f"Processed: {filename}")




