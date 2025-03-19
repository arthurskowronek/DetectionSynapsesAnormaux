import os
import shutil
import joblib
import datetime
import skan
import numpy as np
import pandas as pd
from numpy.random import RandomState, MT19937, SeedSequence
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
from boruta import BorutaPy
import skimage as ski
from skimage.io import imread
from skimage.transform import resize
from skimage.color import label2rgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Constants
DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_PKL_NAME = f'dataset_{DATE}.pkl'
DATA_DIR = Path('./data')
DATASET_PKL_DIR = Path('./dataset_pkl')
MUTANT_DIR = DATA_DIR / '_Mutant'
WT_DIR = DATA_DIR / '_WT'
N_FEAT = 12
N_BINS_FEAT = 20
NUMBER_OF_PIXELS = 1024
IMAGE_SIZE = (NUMBER_OF_PIXELS, NUMBER_OF_PIXELS)
MIN_AREA_COMPO = 0

# Features
MIN_AREA_FEATURE = 10
MAX_LENGTH_OF_FEATURES = 10000

# Training parameters
N_RUNS = 100
MAX_BINS = 255
LEARN_RATE = 0.1
MAX_ITER = 1000
IN_PARAM = np.array([MAX_BINS, LEARN_RATE, MAX_ITER], dtype='float')
SEED = RandomState(MT19937(SeedSequence(753))) # Set random seed for reproducibility



### ------------------------------- UTILS ------------------------------- ###
### --------------------------------------------------------------------- ###

# Dataset functions
def show_dataset_properties(data):
    """
    Display properties of the dataset.
    
    Parameters
    ----------
    data : dict
        Dataset dictionary
    """
    print('')
    print('------------------------------------------')
    print('| Number of samples:', len(data['data']))
    print('| Keys:', list(data.keys()))
    print('| Description:', data['description'])
    print('| Image shape:', data['data'][0].shape if data['data'] else 'No data')
    print('| Labels:', np.unique(data['label']) if data['label'] else 'No labels')
    print('| Label counts:')
    for label in np.unique(data['label']):
        count = sum(1 for x in data['label'] if x == label)
        print(f'|   {label}: {count}')
    print('------------------------------------------')
    print('')

def create_dataset(reimport_images=False, pkl_name=DEFAULT_PKL_NAME):
    """
    Create a dataset from images in directory "data" and save it as a pkl file.
    
    Parameters
    ----------
    reimport_images : bool
        If True, reimport images from directory "data" and save them as a pkl file.
        If False, load the existing pkl file.
    pkl_name : str
        Name of the pkl file to save/load the dataset.
        
    Returns
    ------- 
    data : dict
        Dictionary containing the dataset.
    """
    
    # Ensure the pkl directory exists
    DATASET_PKL_DIR.mkdir(exist_ok=True)
    
    # Load existing dataset if not reimporting
    if not reimport_images:
        try:
            data = joblib.load(DATASET_PKL_DIR / pkl_name)
            print('Data loaded successfully')
            return data
        except FileNotFoundError:
            print(f"Dataset file {pkl_name} not found. Reimporting images...")
            reimport_images = True
    
    if reimport_images:
        print("Importing images...")
        
        # Ensure directories exist and are empty
        for directory in [MUTANT_DIR, WT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            # Clear directory
            for file in directory.glob('*'):
                file.unlink()
        
        # Setup data dictionary
        data = {
            "description": "original (1024x1024) C elegans images in grayscale",
            "label": [],
            "filename": [],
            "data": []
        }
        
        # Process subdirectories
        count_mutant = 0
        count_wildtype = 0
        
        # Find all subdirectories
        subdirectories = [x[0] for x in os.walk(DATA_DIR)][1:]
        
        # Copy and process files from subdirectories
        for subdirectory in subdirectories:
            subdir_path = Path(subdirectory)
            subdir_name = subdir_path.name
            
            if subdir_name.startswith('Mut'):
                target_dir = MUTANT_DIR
                prefix = "Mut"
                counter = count_mutant
                count_mutant = copy_and_rename_files(subdir_path, target_dir, prefix, counter)
                
            elif subdir_name.startswith('WildType'):
                target_dir = WT_DIR
                prefix = "WT"
                counter = count_wildtype
                count_wildtype = copy_and_rename_files(subdir_path, target_dir, prefix, counter)
        
        print(f"Images imported. Mutant files: {count_mutant}, WildType files: {count_wildtype}")
        
        # Load images into data dictionary
        for label, directory in [("Mutant", MUTANT_DIR), ("WildType", WT_DIR)]:
            for file in directory.glob('*.tif'):
                try:
                    im = imread(file)
                    im = process_image_format(im)
                    
                    data["label"].append(label)
                    data["filename"].append(file.name)
                    data["data"].append(im)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        # Save dataset
        joblib.dump(data, pkl_name)
        shutil.move(pkl_name, DATASET_PKL_DIR)
        print(f"Dataset saved as {DATASET_PKL_DIR / pkl_name}")
        
        return data

def copy_and_rename_files(source_dir, target_dir, prefix, counter):
    """
    Process files in a directory by copying and renaming them.
    
    Parameters
    ----------
    source_dir : Path
        Source directory containing files to process
    target_dir : Path
        Target directory to copy files to
    prefix : str
        Prefix for renamed files
    counter : int
        Starting counter for file naming
        
    Returns
    -------
    int
        Updated counter
    """
    for file in source_dir.glob('*.tif'):
        try:
            shutil.copy(file, target_dir)
            new_name = f"{prefix}{counter}.tif"
            os.rename(target_dir / file.name, target_dir / new_name)
            counter += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return counter 

def process_image_format(image):
    """
    Process an image to ensure consistent format and size.
    
    Parameters
    ----------
    image : ndarray
        Input image
        
    Returns
    -------
    ndarray
        Processed image with consistent size and format
    """
    # Handle multi-channel images
    if len(image.shape) > 2:
        image = image[1, :, :]
    
    # Resize if necessary
    if image.shape != IMAGE_SIZE:
        image = resize(image, IMAGE_SIZE, preserve_range=True)
        
    # Ensure consistent data type
    return image.astype(np.uint16)

# Image functions
def display_image(image, number=None, image_type=''):
    """
    Display an image with matplotlib.
    
    Parameters
    ----------
    image : ndarray
        Image to display
    number : int or str, optional
        Identifier for the image
    image_type : str, optional
        Type of the image (e.g., 'original', 'Frangi')
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    
    title = "Image"
    if number is not None:
        title += f" {number}"
    if image_type:
        title += f" ({image_type})"
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_2_images(image1, image2, title1="Image 1", title2="Image 2", cmap="gray"):
    """
    Displays two images side by side using matplotlib.

    Parameters
    ----------
    image1 : ndarray
        The first image to display.
    image2 : ndarray
        The second image to display.
    title1 : str, optional
        Title for the first image.
    title2 : str, optional
        Title for the second image.
    cmap : str, optional
        Colormap to use for displaying the images (e.g., 'gray', 'viridis').
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Create a figure with 1 row and 2 columns

    axes[0].imshow(image1, cmap=cmap)
    axes[0].set_title(title1)
    axes[0].axis('off')

    axes[1].imshow(image2, cmap=cmap)
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def display_4_images(image1, image2, image3, image4, titles=None, cmap="gray"):
    """
    Displays 4 images in a 2x2 grid using matplotlib.

    Parameters
    ----------
    image1 : ndarray
        The first image to display.
    image2 : ndarray
        The second image to display.
    image3 : ndarray
        The third image to display.
    image4 : ndarray
        The fourth image to display.
    titles : list of str, optional
        A list of titles for each image. If None, default titles are used.
    cmap : str, optional
        Colormap to use for displaying the images (e.g., 'gray', 'viridis').
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # 2 rows, 2 columns

    images = [image1, image2, image3, image4]
    if titles is None:
        titles = [f"Image {i+1}" for i in range(4)] #Default titles if none are passed in.

    for i, ax in enumerate(axes.flatten()): #Iterate through each subplot. flatten() makes it easier to iterate.

        ax.imshow(images[i], cmap=cmap)
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
def display_6_images(image1, image2, image3, image4, image5, image6, titles=None, cmap="gray"):
    """
    Displays 6 images in a 2x3 grid using matplotlib.

    Parameters
    ----------
    image1 : ndarray
        The first image to display.
    image2 : ndarray
        The second image to display.
    image3 : ndarray
        The third image to display.
    image4 : ndarray
        The fourth image to display.
    image5 : ndarray
        The fifth image to display.
    image6 : ndarray
        The sixth image to display.
    titles : list of str, optional
        A list of titles for each image. If None, default titles are used.
    cmap : str, optional
        Colormap to use for displaying the images (e.g., 'gray', 'viridis').
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    images = [image1, image2, image3, image4, image5, image6]

    for i, ax in enumerate(axes):
        if i % 3 == 2:  # Histogram plot
            ax.plot(images[i]) #plot the histogram as a lineplot
            ax.set_title(titles[i])
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
        else: #Image plot
            ax.imshow(images[i], cmap=cmap)
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()

def colorize_image(X_original, features):
    colored_image = np.zeros((len(X_original), X_original.shape[1], X_original.shape[2], 3)) 
    for i, image in enumerate(X_original):
        colored_image[i] = get_image_with_color_features(X_original[i], features['components'][i], features['label_components'][i])
    return colored_image

def get_image_with_color_features(X_original, components, label_components):
    
    for component in components:
        if component.area < MIN_AREA_FEATURE:
            for x_p, y_p in component.coords:
                label_components[x_p, y_p] = 0
                
    # Normalize original image for overlay
    if np.max(X_original) > np.min(X_original):
        normalized_im = (X_original - np.min(X_original)) / (np.max(X_original) - np.min(X_original))
    else:
        normalized_im = np.zeros_like(X_original)
             
    # Create overlay
    colored_image = label2rgb(
        label_components, 
        image=normalized_im, 
        bg_label=0
    )
            
    return colored_image

# Analysis functions
def get_histogram_vector(X):
    """
    Create histograms for each image in the dataset.
    
    Parameters
    ----------
    X : ndarray
        Image dataset
    
    Returns
    -------
    ndarray
        Histogram dataset
    """
    
    # calculate maximum pixel value in X 
    if np.issubdtype(X.dtype, np.integer):
        max_pixel = np.max([np.max(image) for image in X])
    else:
        max_pixel = int(np.max(X)) + 1 # Add 1 to include the max pixel value as a bin.
    
    X_hist = np.zeros((len(X), max_pixel))
    for im_num, image in enumerate(X):
        hist, _ = np.histogram(image.flatten(), bins=max_pixel, range=(0, max_pixel))
        # put the first bin (0) to 0
        hist[0] = 0
        X_hist[im_num] = hist
    return X_hist
        
def display_histogram(image, max_pixel, number=None, image_type=''):
    """
    Display a histogram of the image with matplotlib.
    
    Parameters
    ----------
    image : ndarray
        Image to display
    number : int or str, optional
        Identifier for the image
    image_type : str, optional
        Type of the image (e.g., 'original', 'Frangi')
    """
    
    if image.dtype != np.uint16:
        print("Image is not in uint16 format.")
    else:
        plt.figure(figsize=(8, 8))
        plt.hist(image.ravel(), bins=max_pixel, range=(0, max_pixel), density=True, color='black', alpha=0.75)
        
        title = "Histogram"
        if number is not None:
            title += f" {number}"
        if image_type:
            title += f" ({image_type})"
        
        plt.title(title)
        plt.xlabel('Pixel intensity')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show() 

def show_errors(X, y, X_feat, X_preprocessed): # A REFAIRE SANS ENTRAINER DE NOUVEAU UN MODEL
    """
    Display misclassified images with labeled components overlaid.
    
    Parameters
    ----------
    X_feat : ndarray
        Feature vectors
    y : array-like
        Labels
    features : dict
        Features dictionary with filenames
    X : ndarray
        Original images
    X_frangi : ndarray
        Frangi-filtered images
    min_area : int, optional
        Minimum component area to keep
    random_state : int or None, optional
        Random state for splitting data
    test_size : float, optional
        Proportion of data to use for testing
    
    Returns
    -------
    dict
        Classification results summary
    """
    
    # Convert inputs to pandas for easier handling
    X_feat_df = pd.DataFrame(X_feat)
    y_series = pd.Series(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
            X_feat_df, y_series, test_size=0.2, random_state=None
    )
    
    # Train classifier
    print("Training classifier...")
    clf = HistGradientBoostingClassifier()
    
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    # Track results
    correct = 0
    errors = []
    
    # Process each prediction
    print("\nAnalyzing misclassifications:")
    for im_num in range(len(predictions)):
        true_label = y_test.iloc[im_num]
        predicted_label = predictions[im_num]
        original_im_index = y_test.index[im_num]
        
        if predicted_label != true_label:
            # Get filename for the misclassified image
            file_name = f"image_{original_im_index}"
            
            print(f'Error on image {original_im_index}: predicted {predicted_label}, actual {true_label}, file {file_name}')
            errors.append((original_im_index, predicted_label, true_label))
            
            image_label_overlay = get_image_with_color_features(X[original_im_index], X_preprocessed[original_im_index])
                
            # Display image with overlay
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title(f'File: {file_name}\nPredicted: {predicted_label}, Actual: {true_label}')
            ax.imshow(image_label_overlay)
            ax.set_axis_off()
            plt.tight_layout()
            plt.show()
   
        else:
            correct += 1
    
    # Print results summary
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    print(f'\nResults: {correct}/{total} correct predictions ({accuracy:.1%})')
    
    if len(errors) > 0:
        print(f'Total misclassifications: {len(errors)}')
    
    # Return results dictionary
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'errors': errors,
        'classifier': clf
    }

def show_distribution_features(features, mutant_label="Mutant", wt_label="WildType"): # A REFAIRE ? 
    distrib_WT = np.zeros_like(features['data'][0])
    distrib_mutants = np.zeros_like(features['data'][0])
    
    # Get counts to avoid division by zero
    mutant_count = sum(1 for label in features['label'] if label == mutant_label)
    wt_count = sum(1 for label in features['label'] if label == wt_label)
    
    # Check if counts are valid
    if mutant_count == 0 or wt_count == 0:
        print(f"Warning: Found {mutant_count} {mutant_label} samples and {wt_count} {wt_label} samples")
        print(f"Available labels: {np.unique(features['label'])}")
        return
    
    for im_num, feature in enumerate(features['data']):
        if features['label'][im_num] == mutant_label:
            distrib_mutants = np.add(feature, distrib_mutants)
        elif features['label'][im_num] == wt_label:
            distrib_WT = np.add(feature, distrib_WT)
    
    distrib_mutants = distrib_mutants / mutant_count
    distrib_WT = distrib_WT / wt_count
    
    _, ax = plt.subplots()
    ax.plot(np.arange(len(features['data'][0])), distrib_mutants, 'bo-', label=mutant_label)
    ax.plot(np.arange(len(features['data'][0])), distrib_WT, 'ro-', label=wt_label)
    ax.legend()
    plt.title("Feature Distribution Comparison")
    plt.xlabel("Feature Index")
    plt.ylabel("Average Feature Value")
    plt.show()



### --------------------------- PREPROCESSING --------------------------- ###
### --------------------------------------------------------------------- ###

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

def create_mask_synapse(image): # A AMELIORER
    # ----- ADJUST CONTRAST ----- 
    #image = anisotropic_diffusion(image) # remove noise and enhance edges
    #image = exposure.adjust_gamma(image, gamma=3) 
    #image = exposure.adjust_log(image, gain=2, inv=False) 
    #image = ski.exposure.equalize_hist(image) # not a good idea
        
        
    # ----- TUBNESS FILTERS -----
    # Meijering filter
    #image = ski.filters.meijering(image, sigmas=range(1, 8, 2), black_ridges=False) # quand on baisse le sigma max, on garde seulement les vaisseaux fins
    # Sato filter
    image_sato = ski.filters.sato(image, sigmas=range(1, 3, 1), black_ridges=False)
    # Hessian filter
    #image = ski.filters.hessian(image,black_ridges=False,sigmas=range(1, 5, 1), alpha=2, beta=0.5, gamma=15)
    # Franji filter
    image = ski.filters.frangi(image,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    #image = ski.filters.frangi(image,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=15)
        
        
    #display_image(0.9 * image + 0.1 * image_sato)
    display_image(image)
    
    #image =  0.9 * image + 0.1 * image_sato
        
    # gabors filter
    #real, imag = ski.filters.gabor(image, frequency=0.5)
    #image = real 
        
    # hysterisis thresholding
    image = ski.filters.apply_hysteresis_threshold(image, 0.02, 0.15)
      
    display_image(image) 
        
    # ----- DENOISE -----
    #image = ski.restoration.denoise_nl_means(image, h=0.7)
    #image = ski.restoration.denoise_bilateral(image)
    #image = ski.restoration.denoise_tv_chambolle(image, weight=0.1)
    #image = ski.restoration.denoise_bilateral(image)
        

    # ----- REMOVE SMALL OBJECTS -----
    """image = remove_small_objects(image, option=2, min_size_value=25)
        
    
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
    image = label_components"""
        
        
    # get skeleton
    skeleton = ski.morphology.skeletonize(image)

    display_image(skeleton)
    
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
    
    display_image(image)
        
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
    selem = ski.morphology.disk(1)
    image = ski.morphology.dilation(image, selem)

    
    # ----- CLOSE GAP BETWEEN EDGES -----
    #image = close_gap_between_edges(image, max_distance=10)
    
    display_image(image)
    
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


### ----------------------------- FEATURES ------------------------------ ###
### --------------------------------------------------------------------- ###

def first_neighbor_distance_histogram(positions, bins):
    """
    Compute the histogram of the distance to first neighbor of the centroids.
    
    Parameters
    ----------
    positions : ndarray
        Array of centroid positions
    bins : ndarray
        Bins for histogram
        
    Returns
    -------
    ndarray
        Normalized histogram
    """
    if len(positions) <= 1:
        return np.zeros(len(bins)-1)
        
    min_dist = np.zeros(len(positions))
    
    # Take each blob position in turn
    for indx in range(len(positions)):
        curr_pos = positions[indx, :]
        # Square of the differences of the positions with the other centroids
        sq_diff = np.square(np.array(positions) - curr_pos)
        # Distances with other centroids
        dist = np.sqrt(np.sum(sq_diff, axis=1))
        # Remove the zero distance of curr_pos with itself
        dist = dist[dist > 0]
        
        if len(dist) > 0:
            # Keep the smallest distance
            min_dist[indx] = dist.min()
        else:
            min_dist[indx] = 0

    histo, _ = np.histogram(min_dist, bins)
    sum_histo = np.sum(histo)
    
    if sum_histo > 0:
        return histo / sum_histo
    return histo

def create_feature_vector(image, component_props, n_bins=N_BINS_FEAT): # A AMELIORER
    """
    Create a feature vector from a preprocessed image.
    
    Parameters
    ----------
    image : ndarray
        Preprocessed image
    n_features : int
        Number of features
    n_bins : int
        Number of bins for histograms
        
    Returns
    -------
    ndarray
        Feature vector
    """
      
    
    if not component_props:
        print("Warning: No components found in image.")
        return np.zeros(N_FEAT * (n_bins - 1) + 1)
    
    # Number of detected synapses
    num_synapses = len(component_props)

    # Define feature extraction mapping
    feature_dict = {
        "major_axis_length": [x.axis_major_length for x in component_props],
        "axis_ratio": [x.axis_minor_length / x.axis_major_length if x.axis_major_length != 0 else 0 for x in component_props],
        "extent": [x.extent for x in component_props],
        "area": [x.area for x in component_props],
        "perimeter": [x.perimeter for x in component_props],
        "convex_area": [x.convex_area for x in component_props],
        "eccentricity": [x.eccentricity for x in component_props],
        "solidity": [x.solidity for x in component_props],
        "mean_intensity": [x.mean_intensity for x in component_props],
        "max_intensity": [x.max_intensity for x in component_props],
        "min_intensity": [x.min_intensity for x in component_props]
    }

    # Define histogram bin ranges for each feature
    bin_ranges = {
        "major_axis_length": (0, 20),
        "axis_ratio": (0, 1),
        "extent": (0, 1),
        "area": (0, 50),
        "perimeter": (0, 50),
        "convex_area": (0, 50),
        "eccentricity": (0, 1),
        "solidity": (0, 1),
        "mean_intensity": (0, 3000),
        "max_intensity": (0, 3000),
        "min_intensity": (0, 3000)
    }

    # Compute histograms for all features
    feat_vector = [num_synapses]

    for feature, values in feature_dict.items():
        bin_range = bin_ranges[feature]
        hist, bins = np.histogram(values, bins=np.linspace(*bin_range, num=n_bins), density=True)
        feat_vector.extend(hist * (bins[1] - bins[0]))

    # Distance to nearest neighbor histogram
    bins = np.linspace(3, 30, num=n_bins)
    centroid_positions = np.array([x.centroid for x in component_props])
    hist_nn = first_neighbor_distance_histogram(centroid_positions, bins)
    feat_vector.extend(hist_nn * (bins[1] - bins[0]))

    return np.array(feat_vector)

def get_regions_of_interest(coord, image_original, binary_mask):
    # Initialize markers
    image_markers = np.zeros_like(image_original, dtype=np.int32)

    # Assign unique labels to each centroid
    for i, (x, y) in enumerate(coord, 1):  # Start at 1 (0 is background)
        image_markers[int(x), int(y)] = i + 100 # Add 100 to avoid overlap with binary mask

    # Expand markers to avoid single-pixel problems
    #image_markers = ski.morphology.dilation(image_markers, ski.morphology.disk(2))  # Dilation to help watershed grow

    # Apply watershed segmentation
    segmented = ski.segmentation.watershed(-image_original, connectivity=1, markers=image_markers, mask=binary_mask)

    # Visualize segmentation
    """colored_labels = label2rgb(segmented, image=image_original, bg_label=0)
    plt.figure(figsize=(8, 6))
    plt.imshow(colored_labels)
    plt.title("Refined Watershed Segmentation")
    plt.show()"""
    
    # Calculate region properties
    region_props = ski.measure.regionprops(segmented, intensity_image=image_original)
    

    return region_props, segmented

def get_feature_vector(X, y, X_orig, max_images, mask_images, recompute=False, pkl_name=DEFAULT_PKL_NAME, n_features=N_FEAT, n_bins=N_BINS_FEAT):
    """
    Create feature vectors from preprocessed images.
    
    Parameters
    ----------
    X : ndarray
        Preprocessed images
    y : ndarray
        Labels
    recompute : bool
        If True, recompute features. If False, load from file.
    pkl_name : str
        Base name for the features pkl file
    n_features : int
        Number of features
    n_bins : int
        Number of bins for histograms
        
    Returns
    -------
    ndarray
        Feature vectors
    dict
        Features dictionary
    """
    DATASET_PKL_DIR.mkdir(exist_ok=True)
    features_file_name = f'{Path(pkl_name).stem}_features.pkl'
    
    # Try to load existing features
    if not recompute:
        try:
            features = joblib.load(DATASET_PKL_DIR / features_file_name)
            print('Features loaded from file.')
            
            X_feat = np.zeros((len(features['data']), n_features * (n_bins - 1)))
            for im_num, feat in enumerate(features['data']):
                X_feat[im_num, :] = feat
                
            return X_feat, features
        except FileNotFoundError:
            print('Features file not found. Recomputing...')
            recompute = True
    
    # If recompute is True or features not found : compute features
    if recompute:
        print('Computing features...')
        
        # Initialize feature matrix
        X_feat = np.zeros((len(X), n_features * (n_bins - 1) + 1)) 
        features = {
            'description': 'C elegans images features',
            'label': [],
            'data': [],
            'filename': [], 
            'components': [],
            'label_components': []
        }

        for im_num, image in enumerate(X):
            print(f'Extracting features for image {im_num+1}/{len(X)}')
            image_original = X_orig[im_num]
            maxima = max_images[im_num]
            mask = mask_images[im_num]
        
            # Get regions of interest
            component, label = get_regions_of_interest(maxima, image_original, mask)
            
            # Compute feature vector
            feat = create_feature_vector(image, component, n_bins)
            
            # Save features in dictionary
            features['label'].append(y[im_num])
            features['data'].append(feat)
            features['filename'].append(f"image_{im_num}")  # Fallback filename
            features['components'].append(component)
            features['label_components'].append(label)
            X_feat[im_num, :] = feat

        # Save features in pkl file
        print('Features computed.')
        """joblib.dump(features, features_file_name)
        shutil.move(features_file_name, DATASET_PKL_DIR)
        print(f'Features computed and saved to {DATASET_PKL_DIR / features_file_name}')"""

        return X_feat, features


### ----------------------------- LEARNING ------------------------------ ###
### --------------------------------------------------------------------- ###

def train_model(X_features, y, seed=SEED, n_runs=N_RUNS, params=IN_PARAM):
    """Train a model on the feature vectors."""

    print("Training model...")

    # Convert labels to numeric
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])

    # Initialize results
    correct_estimations = []

    # Run multiple training iterations
    for run in range(n_runs):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_numeric, test_size=0.3, random_state=seed
        )

        # Apply Boruta feature selection
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=seed
        )
        """boruta_selector = BorutaPy(clf, n_estimators='auto', verbose=2, random_state=seed)
        boruta_selector.fit(X_train, y_train)

        X_train_transformed = boruta_selector.transform(X_train)
        X_test_transformed = boruta_selector.transform(X_test)

        # Train model with selected features
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=seed
        )"""
        X_train_transformed = X_train
        X_test_transformed = X_test
        
        
        clf.fit(X_train_transformed, y_train)

        # Evaluate
        y_pred = clf.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        correct_estimations.append(accuracy)

        # Update seed for next iteration
        seed = RandomState(seed.randint(0, 2**32 - 1))

        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{n_runs} runs. Current mean accuracy: {np.mean(correct_estimations):.4f}")

    mean_correct_estim = np.mean(correct_estimations)
    return mean_correct_estim

