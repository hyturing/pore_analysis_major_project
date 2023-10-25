import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, use_adaptive_thresh=False, use_morphology=False, use_canny=False):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional
    # gray_img = cv2.equalizeHist(gray_img)
    
    # 1. Gaussian Blur
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # 2. Adaptive Thresholding
    if use_adaptive_thresh:
        binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    else:
        _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Morphological Operations
    if use_morphology:
        kernel = np.ones((3, 3), np.uint8)
        binary_img = cv2.erode(binary_img, kernel, iterations=1)
        binary_img = cv2.dilate(binary_img, kernel, iterations=1)
    
    # 4. Canny Edge Detection
    if use_canny:
        binary_img = cv2.Canny(gray_img, 50, 150)
    
    return img, binary_img


def detect_pores(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter Small Contours
    min_pore_area = 1
    contours = [c for c in contours if cv2.contourArea(c) > min_pore_area]
    return contours

def calculate_density(img, contours):
    total_area = img.shape[0] * img.shape[1]
    pore_area = sum([cv2.contourArea(c) for c in contours])
    return pore_area / total_area

def density_variation(binary_img, num_strips):
    strip_width = binary_img.shape[1] // num_strips
    densities = []

    for i in range(num_strips):
        strip = binary_img[:, i*strip_width:(i+1)*strip_width]
        densities.append(np.sum(strip) / (255 * strip_width * binary_img.shape[0]))
    
    return densities

def convert_density_to_real_world(density, pixel_size):
    # Converts the pore density from "pores per pixel squared" to "pores per micrometer squared" 
    return density / (pixel_size ** 2)

def fill_pores(binary_img, contours):
    # Make a copy of the binary image
    filled_img = binary_img.copy()

    for contour in contours:
        # Get a point inside the contour; using the first point
        seed_point = tuple(contour[0][0])
        
        flags = 4
        flags |= cv2.FLOODFILL_MASK_ONLY
        flags |= (255 << 8)

        # Create a mask used for flood filling. 
        # Note: the size needs to be 2 pixels more than the image size.
        mask = np.zeros((binary_img.shape[0] + 2, binary_img.shape[1] + 2), np.uint8)
        
        cv2.floodFill(filled_img, mask, seed_point, 255, flags=flags)

    return filled_img


def plot_graph(densities):
    x = list(range(1, len(densities)+1))
    plt.plot(x, densities, marker='o', linestyle='-')
    plt.xlabel('Strip Number')
    plt.ylabel('Pore Density (per pixel^2)')
    plt.title('Variation of Pore Density from Left to Right')
    plt.show()

if __name__ == '__main__':
    image_path = 'image_path'  
    pixel_size = 1 
    
    img, binary_img = preprocess_image(image_path)
    contours = detect_pores(binary_img)
    
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)  # Draw in green
    # cv2.imshow('Detected Pores', img)
    
    pore_img = np.zeros_like(binary_img)
    cv2.drawContours(pore_img, contours, -1, (255), -1)  # Draw filled in white
    # cv2.imshow('Pores on Black Background', pore_img)

    filled_pores_img = fill_pores(binary_img, contours)
    cv2.imshow('Filled Pores', filled_pores_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite("result.jpg", filled_pores_img)
    
    overall_density = calculate_density(img, contours)
    overall_density_micrometers = convert_density_to_real_world(overall_density, pixel_size)
    print(f"Overall Pore Density: {overall_density_micrometers} pores/pixel^2")
    
    densities = density_variation(binary_img, num_strips=10)
    densities_micrometers = [convert_density_to_real_world(d, pixel_size) for d in densities]
    plot_graph(densities_micrometers)
