import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb


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
 
def create_histogram(X):
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


def show_distribution_features(X_feat, features, mutant_label="Mutant", wt_label="WildType"):
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


