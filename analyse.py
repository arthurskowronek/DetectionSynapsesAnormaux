import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb

MIN_AREA_COMPO = 5


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


def show_errors(X_feat, y, X, X_preprocessed, min_area=MIN_AREA_COMPO, random_state=None, test_size=0.2):
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
            X_feat_df, y_series, test_size=test_size, random_state=random_state
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
            
            # Get original and processed images
            test_im = X[original_im_index]
            test_im_frangi = X_preprocessed[original_im_index]
            
            # Process image to show components
            # Threshold and label image
            threshold = threshold_otsu(test_im_frangi)
            binary_image = test_im_frangi > threshold
            labeled_components = label(binary_image)
                
            # Filter small components
            component_props = regionprops(labeled_components)
            for component in component_props:
                if component.area < min_area:
                    for x_p, y_p in component.coords:
                        labeled_components[x_p, y_p] = 0
                
            # Normalize original image for overlay
            if np.max(test_im) > np.min(test_im):
                normalized_im = (test_im - np.min(test_im)) / (np.max(test_im) - np.min(test_im))
            else:
               normalized_im = np.zeros_like(test_im)
                
            # Create overlay
            image_label_overlay = label2rgb(
                labeled_components, 
                image=normalized_im, 
                bg_label=0
            )
                
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


