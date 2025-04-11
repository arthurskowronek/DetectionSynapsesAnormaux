import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage import io, filters, feature
from skimage.filters import frangi
from skimage.measure import label
from skimage.morphology import remove_small_holes, binary_closing, disk

def remove_small_objects(image, option=1, min_size_value=30):
    if option == 1:
        bool_img = image.astype(bool)
        temp_result = ski.morphology.remove_small_objects(bool_img, min_size=min_size_value)
        image = temp_result
        return image
    elif option == 2:
        # Label connected components and remove small objects (intestines likely being smaller than synapses)
        labeled_image = ski.measure.label(image) 
        regions = ski.measure.regionprops(labeled_image)
        # Filter based on region properties (e.g., area) to keep only large regions (synapse chain)
        large_regions = np.zeros_like(labeled_image)

        for region in regions:
            if region.area > min_size_value:
                large_regions[labeled_image == region.label] = 1       
        # The final processed image should only contain the chain-like synapse regions
        return large_regions.astype(np.uint16)
    else:
        print("Option not available")
        return image

def worm_segmentation(img):
    
    # Apply Gaussian filter to smooth the image
    img = ski.filters.meijering(img, sigmas=range(8, 14, 2), black_ridges=False) # quand on baisse le sigma max, on garde seulement les vaisseaux fins
    
    # plot
    """plt.imshow(img, cmap='gray')
    plt.title('Meijering Filter Response')
    plt.show()"""
    
    # Thresholding to create a binary mask
    threshold = filters.threshold_otsu(img)
    """print("Threshold value:", threshold)
    print("Mean value:", np.mean(img))"""
    binary_mask = img > np.mean(img)
    
    # plot
    """plt.imshow(binary_mask, cmap='gray')
    plt.title('Binary Mask')
    plt.show()"""
    
    # Remove small objects from the binary mask
    cleaned_mask = remove_small_objects(binary_mask, option=2, min_size_value=30)
    
    # transform it in boolean
    cleaned_mask = cleaned_mask.astype(bool)
    
    # Fill small holes inside the worm
    worm_mask = remove_small_holes(cleaned_mask, area_threshold=50)

    # plot 
    """plt.imshow(worm_mask, cmap='gray')
    plt.title('After removing small holes : Mask')
    plt.show()"""
    
    # Close small gaps in the worm mask
    worm_mask = binary_closing(worm_mask, disk(20))
    
    # keep the largest connected component (the worm)
    labeled_mask = label(worm_mask)
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1  # +1 to skip background label
    worm_mask = (labeled_mask == largest_component).astype(np.uint8)
    
    
    # plot cleaned mask
    """plt.imshow(worm_mask, cmap='gray')
    plt.title('Cleaned Mask')
    plt.show()"""
    
    
    return worm_mask

def preprocessing(img, threshold_percentile=95):
    """
    Segment synapses in C. elegans fluorescent images.
    
    Parameters:
    -----------
    image_path : str
        Path to the fluorescent image
    frangi_scale_range : tuple
        Range of scales for Frangi filter (min, max)
    frangi_scale_step : float
        Step size for scales in Frangi filter
    threshold_percentile : int
        Percentile for thresholding Frangi response
    connectivity_angle_threshold : float
        Maximum angle deviation for considering two maxima connected (in degrees)
    max_distance : int
        Maximum distance between connected maxima
    min_connections : int
        Minimum number of connections for a maximum to be considered part of a synapse cord
    
    Returns:
    --------
    tuple
        (original image, frangi filter response, mask, detected maxima, filtered maxima)
    """
    
    """# Apply Frangi filter to enhance tubular structures (synapses)
    frangi_response = frangi(img ,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    # show image
    plt.imshow(frangi_response, cmap='gray')
    plt.title('Frangi Filter Response')
    plt.show()
    frangi_response = apply_hysteresis_threshold(frangi_response, 0.01, 0.2)
    plt.imshow(frangi_response, cmap='gray')
    plt.title('Hysteresis Thresholding')
    plt.show()
    
    frangi_response = remove_small_objects(frangi_response, option=2, min_size_value=25)
    plt.imshow(frangi_response, cmap='gray')
    plt.title('Remove Small Objects')
    plt.show()
    
    # keep only components that are more like a line than a blob
    labeled_image = ski.measure.label(frangi_response)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        # if components is more like a line than a blob, keep it
        if component.major_axis_length/component.minor_axis_length > 4:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    frangi_response = label_components
    
    
    # plot
    plt.imshow(frangi_response, cmap='gray')
    plt.title('Filtered 4/1')
    plt.show()
    
    # Normalize Frangi response to 0-1
    frangi_response = (frangi_response - frangi_response.min()) / (frangi_response.max() - frangi_response.min())
    
    # Create mask from Frangi response
    threshold = np.percentile(frangi_response, threshold_percentile)
    mask = frangi_response > threshold
    
    # show mask
    plt.imshow(mask, cmap='gray')
    plt.title('Binary Mask')
    plt.show()"""
    
    # Apply mask to the original image
    worm_mask = worm_segmentation(img)    
    masked_img = img.copy()
    masked_img = masked_img * worm_mask
    
    # Show image
    plt.imshow(masked_img, cmap='gray')
    plt.title('Masked Image')
    plt.show()
    
    frangi_image = frangi(masked_img ,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    
    # show image with mask
    plt.imshow(masked_img, cmap='gray')
    plt.title('Masked Image after Frangi')
    plt.show()
    

    # Find local maxima in the masked image
    local_max = feature.peak_local_max(masked_img, 
                                    min_distance=5, # minimum distance between maxima
                                    threshold_abs=0, # absolute threshold, means that we only consider maxima above this value
                                    exclude_border=False)
    
    # show local maxima on image
    plt.imshow(masked_img, cmap='gray')
    plt.plot(local_max[:, 1], local_max[:, 0], 'r.', markersize=1)
    plt.title(f'Local Maxima ({len(local_max)})')
    plt.show()
    
    print("Number of local maxima detected:", len(local_max))

    return masked_img, local_max, frangi_image

# Example usage
if __name__ == "__main__":
    # image_path is a random image of the directory "data/WildType 2023_12_22"
    path_directory = "data/XWildType 2025_04_02"
    image_path = os.path.join(path_directory, "15.tif") 
    image = io.imread(image_path)
    
    # cut the image to keep only the 45% of the right part without the first 5%
    image_right = image[:, int(image.shape[1]*0.52):int(image.shape[1]*1)] 
    # cut the image to keep only the 45% of the left part without the last 5%
    image_left = image[:, int(image.shape[1]*0):int(image.shape[1]*0.48)]
    
    # Segment synapses
    img_right, local_max_right, frangi_right = preprocessing(image_right)
    img_left, local_max_left, frangi_left = preprocessing(image_left)
    
    # plot both images superposed with transparency
    plt.figure(figsize=(8, 8))
    plt.imshow(img_right, cmap='gray', alpha=0.5)
    plt.imshow(img_left, cmap='gray', alpha=0.5)
    plt.title("Left and Right Images")
    plt.show()
    
    print("Length of local_max_right:", len(local_max_right))
    print("Length of local_max_left:", len(local_max_left))
    
    # for each local_max_right, keep it only if there is no local_max_left in the same position
    local_max = []
    for i in range(len(local_max_right)):
        if local_max_right[i] not in local_max_left:
            local_max.append(local_max_right[i])
    
    local_max = []
    for p1 in local_max_right:
        keep = True
        for p2 in local_max_left:
            if np.linalg.norm(np.array(p1) - np.array(p2)) <= 10:
                keep = False
                break
        if keep:
            local_max.append(p1)
    
    print("Length of local_max:", len(local_max))
    
    local_max = np.array(local_max)
    
    # plot local maxima on the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image_right, cmap='gray')
    plt.plot(local_max[:, 1], local_max[:, 0], 'r.', markersize=1)
    plt.title(f'Local Maxima ({len(local_max)})')
    plt.show()
    