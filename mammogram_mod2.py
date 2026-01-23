import cv2
import numpy as np
from skimage import measure
from skimage.transform import resize
import os
import matplotlib.pyplot as plt



def normalize_to_uint8(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    return image

# Noise Removal (Median Filter)

def noise_removal_median(image):
    """
    Remove noise using 3×3 median filtering.
    """
    return cv2.medianBlur(image, 3)



#Artifact Suppression & Background Separation

def global_threshold(image, threshold=18):
    """
    Separate foreground from background using global thresholding.
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


def connected_components(binary_image):
    """
    Label connected components.
    """
    return measure.label(binary_image, connectivity=2)


def keep_largest_component(labels):
    """
    Keep the largest connected component (breast region).
    """
    regions = measure.regionprops(labels)
    largest = max(regions, key=lambda r: r.area)

    mask = np.zeros_like(labels, dtype=np.uint8)
    mask[labels == largest.label] = 255
    return mask


def apply_mask(image, mask):
    """
    Apply mask to remove background and artifacts.
    """
    return cv2.bitwise_and(image, image, mask=mask)





def flip_right_mlo(image, is_right_mlo):
    """
    Flip right MLO images for consistent orientation.
    """
    if is_right_mlo:
        return cv2.flip(image, 1)
    return image


def extract_upper_left_quadrant(image):
    """
    Divide image into four parts and extract upper-left quadrant.
    """
    h, w = image.shape
    return image[:h//2, :w//2]


def pectoral_triangle_mask(quadrant):
    """
    Create triangular mask representing pectoral muscle
    in the upper-left quadrant.
    """
    h, w = quadrant.shape
    mask = np.zeros_like(quadrant, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if j <= (w - i):
                mask[i, j] = 255

    return mask


def remove_pectoral_muscle_geometric(image):

    h, w = image.shape

    quadrant = image[:h//2, :w//2]
    triangle_mask = pectoral_triangle_mask(quadrant)

    # Invert mask (muscle → 0)
    inv_mask = cv2.bitwise_not(triangle_mask)

    cleaned_quadrant = cv2.bitwise_and(
        quadrant, quadrant, mask=inv_mask
    )

    result = image.copy()
    result[:h//2, :w//2] = cleaned_quadrant

    return result


#  ROI Extraction & Normalization

def crop_roi(image):
    """
    Crop bounding box of breast region.
    """
    coords = cv2.findNonZero(image)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]


def resize_roi(roi, size=(512, 512)):
    """
    Resize ROI to standard size.
    """
    return resize(roi, size, preserve_range=True).astype(np.uint8)



# FULL PREPROCESSING PIPELINE

def preprocess_mammogram(image_path, is_right_mlo=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = normalize_to_uint8(image)

    # Noise removal
    image = noise_removal_median(image)

    # Background & artifact suppression
    binary = global_threshold(image)
    labels = connected_components(binary)
    breast_mask = keep_largest_component(labels)
    breast_image = apply_mask(image, breast_mask)

    #  Orientation
    breast_image = flip_right_mlo(breast_image, is_right_mlo)

    # Pectoral muscle suppression (LTRPM)
    breast_no_muscle = remove_pectoral_muscle_geometric(breast_image)

    #  ROI extraction & resizing
    roi = crop_roi(breast_no_muscle)
    final_image = resize_roi(roi)

    return final_image


# MAIN

if __name__ == "__main__":

    raw_dir = "raw"
    output_dir = "processed"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(raw_dir):
        if filename.lower().endswith(".pgm"):

            image_path = os.path.join(raw_dir, filename)
            is_right_mlo = True  # set manually if needed

            final_image = preprocess_mammogram(image_path, is_right_mlo)

            out_path = os.path.join(
                output_dir, filename.replace(".pgm", ".png")
            )
            cv2.imwrite(out_path, final_image)

            # Visualization
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(original, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Preprocessed")
            plt.imshow(final_image, cmap="gray")
            plt.axis("off")

            plt.show()

            print(f"Processed: {filename}")
