import numpy as np
import skimage as ski
import skan
import datetime
import joblib
import shutil
from pathlib import Path
import matplotlib.pyplot as plt


# Constants
DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_PKL_NAME = f'dataset_{DATE}.pkl'
DATASET_PKL_DIR = Path('./dataset_pkl')
# Features
MIN_AREA_FEATURE = 10
MAX_LENGTH_OF_FEATURES = 20000


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

def close_gap_between_edges(image, max_distance=5):
    # Morphological operation: Dilation followed by Erosion to close the chain gaps
    selem = ski.morphology.disk(max_distance)  # Use a disk-shaped structuring element
    dilated = ski.morphology.dilation(image, selem)
    closed = ski.morphology.erosion(dilated, selem)
    return closed

def create_mask_synapse(image):
    
    image_copy = image.copy()
    
    #display_image(image)
    
    # ------ MASK 1 ------
    image = ski.filters.frangi(image_copy,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    image = ski.filters.apply_hysteresis_threshold(image, 0.01, 0.2)
    # Remove small objects
    image = remove_small_objects(image, option=2, min_size_value=25)
    # keep only components that are more like a line than a blob
    labeled_image = ski.measure.label(image)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        # if components is more like a line than a blob, keep it
        if component.major_axis_length/component.minor_axis_length > 4:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    mask1 = label_components
    
    
    # ------ MASK 2 ------
    """image = ski.filters.frangi(image_copy,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    image = ski.filters.apply_hysteresis_threshold(image, 0.02, 0.15)
    # get skeleton
    skeleton = ski.morphology.skeletonize(image)
    # display_image(skeleton)
    # keep only components of skeleton that are longer than 10 pixels 
    labeled_image = ski.measure.label(skeleton)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        if component.major_axis_length > 50:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    image = label_components  
    # dilate image
    selem = ski.morphology.disk(1)
    mask2 = ski.morphology.dilation(image, selem)"""
    
    
    image = mask1 #| mask2 # combine masks
    
    #display_image(image)
    
    # ----- ADJUST CONTRAST ----- 
    #image = anisotropic_diffusion(image) # remove noise and enhance edges
    #image = exposure.adjust_gamma(image, gamma=3) 
    #image = exposure.adjust_log(image, gain=2, inv=False) 
    #image = ski.exposure.equalize_hist(image) # not a good idea
        
        
    # ----- TUBNESS FILTERS -----
    # Meijering filter
    #meij_image = ski.filters.meijering(image, sigmas=range(1, 8, 2), black_ridges=False) # quand on baisse le sigma max, on garde seulement les vaisseaux fins
    # Sato filter
    #image_sato = ski.filters.sato(image, sigmas=range(1, 3, 1), black_ridges=False)
    # Hessian filter
    #image = ski.filters.hessian(image,black_ridges=False,sigmas=range(1, 5, 1), alpha=2, beta=0.5, gamma=15)
    # Franji filter
    #image = ski.filters.frangi(image,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    #image = ski.filters.frangi(image,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=15)
        
        
    #display_image(0.9 * image + 0.1 * image_sato)
    #display_image(image)
    
    #image =  0.9 * image + 0.1 * meij_image
    
    #display_image(image)
        
    # gabors filter
    #real, imag = ski.filters.gabor(image, frequency=0.5)
    #image = real 
        
    # hysterisis thresholding
    #image = ski.filters.apply_hysteresis_threshold(image, 0.01, 0.2)
      
        
    # ----- DENOISE -----
    #image = ski.restoration.denoise_nl_means(image, h=0.7)
    #image = ski.restoration.denoise_bilateral(image)
    #image = ski.restoration.denoise_tv_chambolle(image, weight=0.1)
    #image = ski.restoration.denoise_bilateral(image)
        

    # ----- REMOVE SMALL OBJECTS -----
    #image = remove_small_objects(image, option=2, min_size_value=25)
        
    
    # keep only components that are more like a line than a blob
    """labeled_image = ski.measure.label(image)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        # if components is more like a line than a blob, keep it
        if component.major_axis_length/component.minor_axis_length > 4:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    image = label_components"""
        
        
    """# get skeleton
    skeleton = ski.morphology.skeletonize(image)

    #display_image(skeleton)
    
    # keep only components of skeleton that are longer than 10 pixels
    labeled_image = ski.measure.label(skeleton)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        if component.major_axis_length > 50:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    image = label_components"""
    
    #display_image(image)
        
    # ----- THRESHOLDING -----
    # threshold otsu
    #threshold_value = ski.filters.threshold_otsu(image)
    # threshold local
    #threshold_value = ski.filters.threshold_local(image, block_size=3)
    # threshold mean
    #threshold_value = ski.filters.threshold_mean(image)
    # threshold triangle
    #threshold_value = ski.filters.threshold_triangle(image)
    # threshold yen
    #threshold_value = ski.filters.threshold_yen(image)
    # threshold li
    #threshold_value = ski.filters.threshold_li(image)
    #fig, ax = ski.filters.try_all_threshold(image, figsize=(8, 5), verbose=True) 
    #plt.show()
    #image = image  > threshold_value
        
    # ----- EDGE DETECTION -----
    # canny edge detector
    #image = ski.feature.canny(image, sigma=1)
    # sobel filter - edge detection
    #image = ski.filters.sobel(image)
    # prewitt filter - edge detection
    #image = ski.filters.prewitt(image)
    # scharr filter
    #image = ski.filters.scharr(image)
    # roberts filter
    #image = ski.filters.roberts(image)
    # laplace filter
    #image = ski.filters.laplace(image, ksize=3) # doesn't work
   
    
    # Hough Transform to detect long edges
    #lines = ski.transform.probabilistic_hough_line(mask_synapses, threshold=10, line_length=5, line_gap=3)
    #for line in lines:
        #p0, p1 = line
        #mask_synapses[p0[0]:p1[0], p0[1]:p1[1]] = 1
        
    
    # dilate image
    #selem = ski.morphology.disk(1)
    #image = ski.morphology.dilation(image, selem)

    
    # ----- CLOSE GAP BETWEEN EDGES -----
    #image = close_gap_between_edges(image, max_distance=10)
    
    #display_image(image)
    
    return image
     
def fill_black_pixels(image): # INUTILE ?
    # for each pixel in the image that is black, we will fill it with the mean of the 5x5 pixels around it
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 0:
                count_number_black_pixels = 0
                for k in range(-2,3):
                    for l in range(-2,3):
                        if i+k >= 0 and i+k < image.shape[0] and j+l >= 0 and j+l < image.shape[1]:
                            if image[i+k,j+l] == 0:
                                count_number_black_pixels += 1
                if count_number_black_pixels == 25:
                    image[i,j] = 0
                else:
                    image[i,j] = np.sum(image[i-2:i+2,j-2:j+2])/(25-count_number_black_pixels)
    return image    
 
def order_skeleton_points_skan(skeleton):
    # Create the Skeleton object
    skel_obj = skan.csr.Skeleton(skeleton)
    
    # Get the summary with branch information
    summary = skan.summarize(skel_obj, separator='-')
    
    # Create a flat list of all points from all paths
    all_points = []
    
    for i in range(len(summary)):
        # Get coordinates for each path
        path_coords = skel_obj.path_coordinates(i)
        # Add all points from this path to the flat list
        for coord in path_coords:
            all_points.append(tuple(coord))
    
    return all_points

def get_high_intensity_pixels (mask, image):
    skeleton = ski.morphology.skeletonize(mask)
        
    ordered_skeleton_points = order_skeleton_points_skan(skeleton)
    intensities = []
    for x, y in ordered_skeleton_points:
        intensities.append(image[x, y])
        
    # smooth intensities
    smoothed_intensities = intensities
    #window_size = 1
    #smoothed_intensities = np.convolve(intensities, np.ones(window_size), 'valid') / window_size 
    
    # get the index of the local maxima. A maxima is a point where the intensity is greater than its neighbors (2 left and 2 right)
    maxima = []
    for i in range(2, len(smoothed_intensities)-2):
        if smoothed_intensities[i] > smoothed_intensities[i-1] and smoothed_intensities[i] > smoothed_intensities[i-2] and smoothed_intensities[i] > smoothed_intensities[i+1] and smoothed_intensities[i] > smoothed_intensities[i+2]:
            maxima.append(i)
    
    # plot the maxima
    #plt.plot(smoothed_intensities)
    #plt.plot(maxima, [smoothed_intensities[i] for i in maxima], 'ro')
    #plt.show() 
    
    # x is a vector from 1 to the length of the smoothed intensities
    x = np.arange(len(smoothed_intensities))
    
    # get the plot to the derive of the smoothed intensities
    derive = np.gradient(smoothed_intensities, x)
    #plt.plot(derive)
    #plt.show()
    
    # get pixel coordinates of the maxima
    maxima_coords = []
    for i in maxima:
        maxima_coords.append(ordered_skeleton_points[i])
        
    # plot the maxima on the image
    for x, y in maxima_coords:
        for i in range(-1,1):
            for j in range(-1,1):
                if x+i >= 0 and x+i < image.shape[0] and y+j >= 0 and y+j < image.shape[1]:
                    image[x+i, y+j] = 65535
        
    #display_image(image)
    
    # complete smoothed intensities until have MAX_LENGTH_OF_FEATURES values
    while len(smoothed_intensities) < MAX_LENGTH_OF_FEATURES:
        smoothed_intensities.append(0)
   
    derive = list(derive)
    while len(derive) < MAX_LENGTH_OF_FEATURES:
        derive.append(0)

    """# get the distance map
    distance_map = scipy.ndimage.distance_transform_edt(mask)  
    # get the local maxima of the distance map
    def detect_local_maxima(image):
        # get the boolean mask of the local maxima
        peaks_mask = ski.feature.peak_local_max(image, min_distance=4, threshold_abs=0)
        # get the coordinates of the local maxima
        coords = np.transpose(np.nonzero(peaks_mask))
        return coords
    maxima_coords = detect_local_maxima(distance_map)"""
    
    return smoothed_intensities, derive, maxima_coords
    
def get_preprocess_images(recompute=False, X=None, pkl_name=DEFAULT_PKL_NAME):
    """
    Apply preprocessing to images.
    
    Parameters
    ----------
    recompute : bool
        If True, recompute preprocessing. If False, load from file.
    X : ndarray
        Array of images to preprocess
    pkl_name : str
        Base name for the preprocessing pkl file
        
    Returns
    -------
    ndarray
        Preprocessed images
    """
    preprocess_file = f'{Path(pkl_name).stem}_preprocessing.pkl'
    
    # if name already exists, add a number to the name
    i = 1
    while (DATASET_PKL_DIR / preprocess_file).exists():
        preprocess_file = f'{Path(pkl_name).stem}_preprocessing_{i}.pkl'
        i += 1
        
    
    # Try to load existing preprocessing
    if not recompute:
        try:
            dict_preprocess = joblib.load(DATASET_PKL_DIR / preprocess_file)
            X_preprocessed = dict_preprocess['X_preprocessed']
            X_intensity = dict_preprocess['X_intensity']    
            X_derivative_intensity = dict_preprocess['X_derivative_intensity']
            maxima_coords = dict_preprocess['maxima_coords']
            mask_synapses = dict_preprocess['mask_synapses']
            print('Preprocessing loaded from file.')
            return X_preprocessed, X_intensity, X_derivative_intensity, maxima_coords, mask_synapses
        except FileNotFoundError:
            print('Preprocessing file not found. Recomputing...')
            recompute = True
    
    # Validate input
    if recompute and X is None:
        raise ValueError("Input images (X) must be provided when recomputing preprocessing")
    
    print('Preprocessing images...')
    X_preprocessed = np.zeros_like(X, dtype=np.float64)
    X_intensity = np.zeros((len(X), MAX_LENGTH_OF_FEATURES), dtype=np.float64)
    X_derivative_intensity = np.zeros((len(X), MAX_LENGTH_OF_FEATURES), dtype=np.float64)
    maxima_coords = [None] * len(X)
    mask_synapses = [None] * len(X)
    
    
    for im_num, image in enumerate(X):
        print(f'Processing image {im_num+1}/{len(X)}')
        
        original_image = image.copy()
        
        # Create mask for synapses
        mask_synapses[im_num] = create_mask_synapse(image)
        
        X_intensity[im_num], X_derivative_intensity[im_num], maxima_coords[im_num] = get_high_intensity_pixels(mask_synapses[im_num], image)
        
        # apply mask to original image
        X_preprocessed[im_num] = original_image * mask_synapses[im_num]
    
    
        
    # Save preprocessing results
    DATASET_PKL_DIR.mkdir(exist_ok=True)
    dict_preprocess = {'X_preprocessed': X_preprocessed, 'X_intensity': X_intensity, 'X_derivative_intensity': X_derivative_intensity, 'maxima_coords': maxima_coords, 'mask_synapses': mask_synapses}
    joblib.dump(dict_preprocess, preprocess_file)
    
    #joblib.dump(X_preprocessed, preprocess_file)
    shutil.move(preprocess_file, DATASET_PKL_DIR)
    print(f'Preprocessing done and saved to {DATASET_PKL_DIR / preprocess_file}')
    
    # return le dictionnaire ? 
    
    return X_preprocessed, X_intensity, X_derivative_intensity, maxima_coords, mask_synapses

