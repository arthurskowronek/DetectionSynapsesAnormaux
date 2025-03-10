import os
import shutil
import joblib
import datetime
from pathlib import Path
import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence
import matplotlib.pyplot as plt
import skimage as ski
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Constants
DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_PKL_NAME = f'dataset_{DATE}.pkl'
DATA_DIR = Path('./data')
DATASET_PKL_DIR = Path('./dataset_pkl')
MUTANT_DIR = DATA_DIR / '_Mutant'
WT_DIR = DATA_DIR / '_WT'
N_FEAT = 4
N_BINS_FEAT = 20
IMAGE_SIZE = (1024, 1024)
MIN_AREA_COMPO = 5

# Training parameters
N_RUNS = 100
MAX_BINS = 255
LEARN_RATE = 0.1
MAX_ITER = 1000
IN_PARAM = np.array([MAX_BINS, LEARN_RATE, MAX_ITER], dtype='float')

# Set random seed for reproducibility
SEED = RandomState(MT19937(SeedSequence(753)))


### ------------------------------- UTILS ------------------------------- ###
### --------------------------------------------------------------------- ###

# Dataset functions
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

def show_dataset_properties(data):
    """
    Display properties of the dataset.
    
    Parameters
    ----------
    data : dict
        Dataset dictionary
    """
    print('Number of samples:', len(data['data']))
    print('Keys:', list(data.keys()))
    print('Description:', data['description'])
    print('Image shape:', data['data'][0].shape if data['data'] else 'No data')
    print('Labels:', np.unique(data['label']) if data['label'] else 'No labels')
    print('Label counts:')
    for label in np.unique(data['label']):
        count = sum(1 for x in data['label'] if x == label)
        print(f'  {label}: {count}')

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


# Image functions
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
    return image.astype(np.uint8)

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


### --------------------------- PREPROCESSING --------------------------- ###
### --------------------------------------------------------------------- ###

def preprocess_images(recompute=False, X=None, pkl_name=DEFAULT_PKL_NAME):
    """
    Apply preprocessing (Frangi filter) to images.
    
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
            X_preprocessed = joblib.load(DATASET_PKL_DIR / preprocess_file)
            print('Preprocessing loaded from file.')
            return X_preprocessed
        except FileNotFoundError:
            print('Preprocessing file not found. Recomputing...')
            recompute = True
    
    # Validate input
    if recompute and X is None:
        raise ValueError("Input images (X) must be provided when recomputing preprocessing")
    
    print('Preprocessing images with Frangi filter...')
    X_preprocessed = np.zeros_like(X, dtype=np.float64)
    
    for im_num, image in enumerate(X):
        print(f'Processing image {im_num+1}/{len(X)}')
        X_preprocessed[im_num] = ski.filters.frangi(
            image, 
            black_ridges=False,
            sigmas=range(1, 5, 1),
            gamma=70
        )
    
    # Save preprocessing results
    DATASET_PKL_DIR.mkdir(exist_ok=True)
    joblib.dump(X_preprocessed, preprocess_file)
    shutil.move(preprocess_file, DATASET_PKL_DIR)
    print(f'Preprocessing done and saved to {DATASET_PKL_DIR / preprocess_file}')
    
    return X_preprocessed


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

def create_feature_vector(X, y, recompute=False, pkl_name=DEFAULT_PKL_NAME, n_features=N_FEAT, n_bins=N_BINS_FEAT):
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
                
            return X_feat
        except FileNotFoundError:
            print('Features file not found. Recomputing...')
            recompute = True
    
    if recompute:
        print('Computing features...')
        
        X_feat = np.zeros((len(X), n_features * (n_bins - 1)))
        features = {
            'description': 'C elegans images features from frangi blobs',
            'label': [],
            'data': [],
            'filename': []
        }

        for im_num, image in enumerate(X):
            print(f'Extracting features for image {im_num+1}/{len(X)}')
            
            # Threshold and label image
            threshold = threshold_otsu(image)
            binary_image = image > threshold
            labeled_components = label(binary_image)
            component_props = regionprops(labeled_components, intensity_image=image)

            # Filter components by size
            sel_component_props = [x for x in component_props if x.area > MIN_AREA_COMPO]
            
            if not sel_component_props:
                print(f"Warning: No components found in image {im_num}")
                feat = np.zeros(n_features * (n_bins - 1))
            else:
                # Extract properties
                axis_M_ls = [x.axis_major_length for x in sel_component_props]
                ratio_axis = [x.axis_minor_length/x.axis_major_length for x in sel_component_props]
                centroids = [x.centroid for x in sel_component_props]
                extents = [x.extent for x in sel_component_props]
                
                # Compute feature histograms
                feat = []

                # Major axis length histogram
                feat1, bins = np.histogram(
                    axis_M_ls,
                    bins=np.linspace(start=3, stop=20, num=n_bins),
                    density=True
                )
                feat = np.append(feat, feat1 * (bins[1] - bins[0]))

                # Axis ratio histogram
                feat1, bins = np.histogram(
                    ratio_axis,
                    bins=np.linspace(start=0, stop=1, num=n_bins),
                    density=True
                )
                feat = np.append(feat, feat1 * (bins[1] - bins[0]))

                # Extent histogram
                feat1, bins = np.histogram(
                    extents,
                    bins=np.linspace(start=0, stop=1, num=n_bins),
                    density=True
                )
                feat = np.append(feat, feat1 * (bins[1] - bins[0]))

                # Distance to nearest neighbor histogram
                bins = np.linspace(start=3, stop=30, num=n_bins)
                feat1 = first_neighbor_distance_histogram(np.array(centroids), bins)
                feat = np.append(feat, feat1 * (bins[1] - bins[0]))

            features['label'].append(y[im_num])
            features['data'].append(feat)
            features['filename'].append(f"image_{im_num}")  # Fallback filename
            X_feat[im_num, :] = feat

        # Save features
        joblib.dump(features, features_file_name)
        shutil.move(features_file_name, DATASET_PKL_DIR)
        print(f'Features computed and saved to {DATASET_PKL_DIR / features_file_name}')

        return X_feat


### ----------------------------- LEARNING ------------------------------ ###
### --------------------------------------------------------------------- ###

def train_model(X_features, y, seed=SEED, n_runs=N_RUNS, params=IN_PARAM):
    """
    Train a model on the feature vectors.
    
    Parameters
    ----------
    X_features : ndarray
        Feature vectors
    y : ndarray
        Labels
    seed : RandomState
        Random state for reproducibility
    n_runs : int
        Number of runs
    params : ndarray
        Training parameters
        
    Returns
    -------
    float
        Mean correct estimation
    """
    
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
        
        # Train model
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=seed
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        accuracy = clf.score(X_test, y_test)
        correct_estimations.append(accuracy)
        
        # Update seed for next iteration
        seed = RandomState(seed.randint(0, 2**32 - 1))
        
        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{n_runs} runs. Current mean accuracy: {np.mean(correct_estimations):.4f}")
    
    mean_correct_estim = np.mean(correct_estimations)
    return mean_correct_estim




if __name__ == "__main__":
    # Load dataset
    filename_pkl_dataset = 'dataset_2025-03-10_08-05-48'
    data = create_dataset(reimport_images=False, pkl_name=filename_pkl_dataset + '.pkl')
    
    # Display dataset properties
    show_dataset_properties(data)
    
    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    # Preprocessing
    X_preprocessed = preprocess_images(recompute=False, X=X, pkl_name=filename_pkl_dataset)
    
    # Display sample images
    if len(X) > 0:
        sample_idx = min(50, len(X) - 1)
        display_image(X[sample_idx], sample_idx, 'original')
        display_histogram(X[sample_idx], X[sample_idx].max(), sample_idx, 'original')
        display_image(X_preprocessed[sample_idx], sample_idx, 'Frangi')
    
    # Compute features
    X_features = create_feature_vector(X_preprocessed, y, recompute=False, pkl_name=filename_pkl_dataset)
    
    # Training
    print('Training model...')
    mean_corr_estim = train_model(X_features, y, SEED, N_RUNS, IN_PARAM)
    print(f'Mean accuracy: {100*mean_corr_estim:.1f}%')