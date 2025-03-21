import os
import shutil
import joblib
import datetime
import random
import numpy as np
import pandas as pd
from numpy.random import RandomState, MT19937, SeedSequence
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.color import label2rgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import seaborn as sns

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
MIN_AREA_FEATURE = 10

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

def create_dataset(reimport_images=False, test_random = False, pkl_name=DEFAULT_PKL_NAME):
    """
    Create a dataset from images in directory "data" and save it as a pkl file.
    
    Parameters
    ----------
    reimport_images : bool
        If True, reimport images from directory "data" and save them as a pkl file.
        If False, load the existing pkl file.
    pkl_name : str
        Name of the pkl file to save/load the dataset.
    test : bool
        If True, select only 1 image of each type (Mutant and WildType) randomly.
        If False, include all images.
        
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
        
                    
        # Temporary file lists for potential random selection
        mutant_files = list(MUTANT_DIR.glob('*.tif'))
        wildtype_files = list(WT_DIR.glob('*.tif'))
        
        # If test mode, select only 1 random image of each type
        if test_random:
            selected_files = []
            
            if mutant_files:
                selected_files.append(("Mutant", random.choice(mutant_files)))
            if wildtype_files:
                selected_files.append(("WildType", random.choice(wildtype_files)))
                
            #print(f"Test mode: Selected {len(selected_files)} images (max 1 per type)")
            
            # Load only selected images
            for label, file in selected_files:
                try:
                    im = imread(file)
                    im = process_image_format(im)
                    data["label"].append(label)
                    data["filename"].append(file.name)
                    data["data"].append(im)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        else:
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

def plot_probability_histograms(y_pred_proba, class_names=None):
    """Plots histograms of predicted probabilities for each class."""

    if y_pred_proba is None:
        print("y_pred_proba is None. Cannot plot histograms.")
        return

    n_classes = y_pred_proba.shape[1]
    plt.figure(figsize=(12, 6))

    for i in range(n_classes):
        plt.subplot(1, n_classes, i + 1)
        sns.histplot(y_pred_proba[:, i], kde=True)
        if  class_names is not None:
            plt.title(f"Probability of Class: {class_names[i]}")
        else:
            plt.title(f"Probability of Class: {i}")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
     
def plot_probability_boxplots(y_pred_proba, class_names=None):
    """Plots box plots of predicted probabilities for each class."""

    if y_pred_proba is None:
        print("y_pred_proba is None. Cannot plot box plots.")
        return

    n_classes = y_pred_proba.shape[1]
    df_proba = pd.DataFrame(y_pred_proba, columns=class_names if class_names else [f"Class {i}" for i in range(n_classes)])

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_proba)
    plt.title("Predicted Probabilities by Class")
    plt.ylabel("Probability")
    plt.show()

def plot_probability_scatter(y_pred_proba, y_test, y_pred):
    """Scatter plots of predicted probabilities vs. true labels for both classes, highlighting incorrect predictions."""
    if y_pred_proba is None:
        print("y_pred_proba is None. Cannot plot probability scatter.")
        return

    errors = y_test != y_pred  # Boolean array indicating errors

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create 1 row, 2 columns of subplots

    # Plot for Class 1 (Probability of Class 1)
    correct_indices = np.where(~errors)[0]
    axes[0].scatter(y_test[correct_indices], y_pred_proba[correct_indices, 1], c='blue', label='Correct')
    incorrect_indices = np.where(errors)[0]
    axes[0].scatter(y_test[incorrect_indices], y_pred_proba[incorrect_indices, 1], c='red', label='Incorrect')
    axes[0].set_xlabel("True Labels")
    axes[0].set_ylabel("Predicted Probability (Class 1)")
    axes[0].set_title("Predicted Probabilities vs. True Labels (Class 1)")
    axes[0].legend()

    # Plot for Class 2 (Probability of Class 2)
    axes[1].scatter(y_test[correct_indices], y_pred_proba[correct_indices, 0], c='blue', label='Correct') # class 0 instead of 2.
    incorrect_indices = np.where(errors)[0]
    axes[1].scatter(y_test[incorrect_indices], y_pred_proba[incorrect_indices, 0], c='red', label='Incorrect') #class 0 instead of 2.
    axes[1].set_xlabel("True Labels")
    axes[1].set_ylabel("Predicted Probability (Class 0)") #class 0 instead of 2.
    axes[1].set_title("Predicted Probabilities vs. True Labels (Class 0)") #class 0 instead of 2.
    axes[1].legend()

    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()