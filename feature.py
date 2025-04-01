import numpy as np
import skimage as ski
import datetime
import joblib
import shutil
import pandas as pd
import pyfeats
import pointpats
from pathlib import Path
from boruta import BorutaPy  
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_engine.selection import MRMR

import matplotlib.pyplot as plt

from scipy import stats
from scipy.signal import savgol_filter

from skimage.feature import graycomatrix, graycoprops, hog
from skimage.color import label2rgb



from constants import *


# Constants
N_FEAT = 12
N_BINS_FEAT = 20
DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_PKL_NAME = f'dataset_{DATE}.pkl'
DATASET_PKL_DIR = Path('./dataset_pkl')

# ---------- Utility functions ----------

def first_neighbor_distance_histogram(positions, bins):
    """
    Compute the histogram of the distance to first neighbor of the centroids.
    
    Parameters
    ----------
    positions : ndarray
        Array of centroid positions.
    bins : ndarray
        Bins for histogram.
        
    Returns
    -------
    ndarray
        Normalized histogram.
    """
    if len(positions) <= 1:
        return np.zeros(len(bins)-1)
        
    min_dist = np.zeros(len(positions))
    
    for indx in range(len(positions)):
        curr_pos = positions[indx, :]
        sq_diff = np.square(np.array(positions) - curr_pos)
        dist = np.sqrt(np.sum(sq_diff, axis=1))
        dist = dist[dist > 0]
        min_dist[indx] = dist.min() if len(dist) > 0 else 0

    histo, _ = np.histogram(min_dist, bins)
    sum_histo = np.sum(histo)
    if sum_histo > 0:
        return histo / sum_histo
    return histo

def compute_zernike_features(roi_mask, radius=10):
    """
    Compute Zernike moments from a binary ROI.
    
    Parameters
    ----------
    roi_mask : ndarray
        Binary mask of the ROI.
    degree : int
        Degree for Zernike moments.
    radius : int
        Radius parameter.
        
    Returns
    -------
    ndarray
        Zernike moments vector.
    """
    # The library function expects the ROI and parameters as given
    # (Assumes that the ROI is centered and the radius is appropriate.)
    roi_mask = roi_mask.astype(np.float32)
    zernike =  pyfeats.zernikes_moments(roi_mask, radius=radius)
    return zernike[0]

def compute_hu_features(roi_mask):
    roi_mask = roi_mask.astype(np.float32)
    hu = pyfeats.hu_moments(roi_mask)
    return hu[0]

def compute_hog_features(roi_intensity, pixels_per_cell=(2, 2), cells_per_block=(2, 2), orientations=9):
    """
    Compute HOG (Histogram of Oriented Gradients) features for an ROI.
    
    Parameters
    ----------
    roi_intensity : ndarray
        Intensity image of the ROI.
    pixels_per_cell : tuple of int
        Size of a cell.
    cells_per_block : tuple of int
        Block size.
    orientations : int
        Number of orientation bins.
        
    Returns
    -------
    ndarray
        HOG feature vector.
    """
    # Resize ROI to a fixed size for uniformity (e.g., 12x12)
    roi_resized = ski.transform.resize(roi_intensity, (12, 12), anti_aliasing=True)
    hog_desc = hog(roi_resized, orientations=orientations, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, feature_vector=True)
    return hog_desc

def compute_ripley_k(centroid_positions, support=50):
    """
    Compute Ripley's K-function for the given centroid positions.
    
    Parameters
    ----------
    centroid_positions : ndarray
        Array of (row, col) coordinates.
    support : float
        Distance at which to evaluate K.
        
    Returns
    -------
    float
        Ripley K-function value at the support distance.
    """
    if len(centroid_positions) < 2:
        return 0.0
    # Unpack the tuple returned by RipleyK: (radii, k_values)
    radii, k_values = pointpats.k(centroid_positions, support=support)
    # Get the K-value at the maximum support (or use an alternative aggregation)
    K_value = k_values[-1]
    return K_value

def rms_roughness(curve, window_length=11, polyorder=2):
    """Calculates the RMS roughness of a 1D curve (intensity values)."""
    curve = np.array(curve)
    # Smooth the curve using Savitzky-Golay filter
    smoothed_curve = savgol_filter(curve, window_length, polyorder)
    # Calculate the deviations
    deviations = curve - smoothed_curve
    # Calculate the RMS roughness
    roughness = np.sqrt(np.mean(deviations**2))
    return roughness

def calculate_glcm_features(image, distances=[1], angles=[0]):
    """
    Calculate multiple GLCM texture features for an image
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale)
    distances : list, optional
        List of pixel pair distances
    angles : list, optional
        List of angles to consider
    
    Returns:
    --------
    dict
        Dictionary of GLCM texture features
    """
    # Ensure image is 2D and has appropriate data type
    image = np.atleast_2d(image).astype(np.uint8)
    
    # Handle empty or zero-sized images
    if image.size == 0:
        return {
            'contrast': 0,
            'dissimilarity': 0,
            'homogeneity': 0,
            'energy': 0,
            'correlation': 0
        }
    
    # Normalize image to 8-bit range if needed
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(
        image, 
        distances=distances, 
        angles=angles, 
        levels=256,  # Full 8-bit range
        symmetric=True,  # Symmetric to capture all directions
        normed=True  # Normalize the matrix
    )
    
    # Calculate features
    features = {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0]
    }
    
    return features

def calculate_glds_features(image):
    """
    Calculate Gray Level Difference Statistics (GLDS) features for an image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale)
    
    Returns:
    --------
    dict
        Dictionary of GLDS features
    """
    # Ensure image is 2D and has appropriate data type
    #image = np.atleast_2d(image).astype(np.uint8)
    
    # Handle empty or zero-sized images
    if image.size == 0:
        return {
            'contrast': 0,
            'entropy': 0,
            'homogeneity': 0,
            'energy': 0,
            'mean': 0
        }
    
    # Normalize image to 8-bit range if needed
    #image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Compute GLDS features
    glds = pyfeats.glds_features(image, mask=None)
    
    features = {
        'contrast': glds[0],
        'entropy': glds[1],
        'homogeneity': glds[2],
        'energy': glds[3],
        'mean': glds[4]
    }
    
    return features

def calculate_ngtdm_features(image):
    """
    Calculate Neighborhood Gray Tone Difference Matrix (NGTDM) features for an image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale)
    
    Returns:
    --------
    dict
        Dictionary of NGTDM features
    """
    # Ensure image is 2D and has appropriate data type
    #image = np.atleast_2d(image).astype(np.uint8)
    
    # Handle empty or zero-sized images
    if image.size == 0:
        return {
            'coarseness': 0,
            'contrast': 0,
            'busyness': 0,
            'complexity': 0,
            'strength': 0
        }
    
    # Compute NGTDM features
    ngtdm = pyfeats.ngtdm_features(image)
    
    features = {
        'coarseness': ngtdm[0],
        'contrast': ngtdm[1],
        'busyness': ngtdm[2],
        'complexity': ngtdm[3],
        'strength': ngtdm[4]
    }
    
    return features

# ---------- Feature extraction ----------

def create_feature_vector(image, component_props, intensity=None, n_bins=N_BINS_FEAT, 
                         basic_features=None, include_neighborhood=True,
                         include_intensity_deriv=True, include_zernike=True, 
                         include_hog=True, include_hu=True):
    """
    Create a modular feature vector from a preprocessed image.
    
    The feature vector can include any combination of:
      - Histogram features of geometric and intensity properties (original features)
      - Neighborhood features (nearest neighbor distances)
      - Mean intensity derivative along the skeleton for each ROI
      - Zernike moments (aggregated as mean over ROIs)
      - HOG features (aggregated as mean over ROIs)
    
    Parameters
    ----------
    image : ndarray
        Preprocessed intensity image.
    component_props : list
        List of region properties (from skimage.measure.regionprops).
    intensity : ndarray, optional
        Intensity image, if different from the image used for segmentation.
    n_bins : int, default=10
        Number of bins for histograms.
    basic_features : list or None, default=None
        List of basic features to include. If None, all basic features are included.
        Options: ["major_axis_length", "axis_ratio", "extent", "area", "perimeter",
                 "convex_area", "eccentricity", "solidity", "mean_intensity",
                 "max_intensity", "min_intensity"]
    include_neighborhood : bool, default=True
        Whether to include neighborhood features (nearest neighbor distances).
    include_intensity_deriv : bool, default=True
        Whether to include intensity derivative features.
    include_zernike : bool, default=True
        Whether to include Zernike moment features.
    include_hog : bool, default=True
        Whether to include HOG features.
        
    Returns
    -------
    ndarray
        Feature vector containing only the requested features.
    dict
        Dictionary with information about feature vector structure.
    """
    # Initialize empty feature vector and feature info dictionary
    feat_vector = []

    if not component_props:
        print("Warning: No components found in image.")
        # Return an empty feature vector and info dictionary
        return np.array([0])
    
    # Always include number of detected components
    num_synapses = len(component_props)
    #print(f"Number of detected components: {num_synapses}")
    feat_vector = [num_synapses]
    
    
    # ---------- Define all possible basic features ----------
    all_basic_features = {
        "major_axis_length": {
            "values": [prop.axis_major_length for prop in component_props],
            "range": (0, np.inf)
        },
        "axis_ratio": {
            "values": [prop.axis_minor_length / prop.axis_major_length if prop.axis_major_length != 0 else 0 for prop in component_props],
            "range": (0, 1)
        },
        "extent": {
            "values": [prop.extent for prop in component_props],
            "range": (0, 1)
        },
        "area": {
            "values": [prop.area for prop in component_props],
            "range": (0, np.inf)
        },
        "perimeter": {
            "values": [prop.perimeter for prop in component_props],
            "range": (0, np.inf)
        },
        "convex_area": {
            "values": [prop.convex_area for prop in component_props],
            "range": (0, 50)
        },
        "eccentricity": {
            "values": [prop.eccentricity for prop in component_props],
            "range": (0, 1)
        },
        "solidity": {
            "values": [prop.solidity for prop in component_props],
            "range": (0, 1)
        },
        "mean_intensity": {
            "values": [prop.mean_intensity for prop in component_props],
            "range": (0, np.inf)
        },
        "max_intensity": {
            "values": [prop.max_intensity for prop in component_props],
            "range": (0, np.inf)
        },
        "min_intensity": {
            "values": [prop.min_intensity for prop in component_props],
            "range": (0, np.inf)
        },
        "gvsd": {
            "values": [prop.intensity_image.std() for prop in component_props],
            "range": (0, np.inf) 
        },
        "skewness": {
            "values": [stats.skew(prop.intensity_image.flatten()) if np.std(prop.intensity_image) > 1e-8 else 0 for prop in component_props],
            "range": (-np.inf, np.inf)
        },
        "kurtosis": {
            "values": [stats.kurtosis(prop.intensity_image.flatten(), fisher=True) if np.std(prop.intensity_image) > 1e-8 else 0 for prop in component_props],
            "range": (-np.inf, np.inf)
        },
        "contrast_GLCM": {
            "values": [calculate_glcm_features(prop.intensity_image)['contrast'] for prop in component_props],
            "range": (0, np.inf)
        },
        "dissimilarity_GLCM": {
            "values": [calculate_glcm_features(prop.intensity_image)['dissimilarity'] for prop in component_props],
            "range": (0, np.inf)
        },
        "homogeneity_GLCM": {
            "values": [calculate_glcm_features(prop.intensity_image)['homogeneity'] for prop in component_props],
            "range": (0, np.inf)
        },
        "energy_GLCM": {
            "values": [calculate_glcm_features(prop.intensity_image)['energy'] for prop in component_props],
            "range": (0, np.inf)
        },
        "correlation_GLCM": {
            "values": [calculate_glcm_features(prop.intensity_image)['correlation'] for prop in component_props],
            "range": (0, np.inf)
        },
        "Gray_Level_Difference_Statistics_contrast": {
            "values": [calculate_glds_features(prop.intensity_image)['contrast'] for prop in component_props],
            "range": (0, np.inf)
        },
        "Gray_Level_Difference_Statistics_entropy": {
            "values": [calculate_glds_features(prop.intensity_image)['entropy'] for prop in component_props],
            "range": (0, np.inf)
        },
        "Gray_Level_Difference_Statistics_homogeneity": {
            "values": [calculate_glds_features(prop.intensity_image)['homogeneity'] for prop in component_props],
            "range": (0, np.inf)
        },
        "Gray_Level_Difference_Statistics_energy": {
            "values": [calculate_glds_features(prop.intensity_image)['energy'] for prop in component_props],
            "range": (0, np.inf)
        },
        "Gray_Level_Difference_Statistics_mean": {
            "values": [calculate_glds_features(prop.intensity_image)['mean'] for prop in component_props],
            "range": (0, np.inf)
        },
        "ngtdm_features_coarseness": {
            "values": [calculate_ngtdm_features(prop.intensity_image)['coarseness'] for prop in component_props],
            "range": (0, np.inf)
        },
        "ngtdm_features_contrast": {
            "values": [calculate_ngtdm_features(prop.intensity_image)['contrast'] for prop in component_props],
            "range": (0, np.inf)
        },
        "ngtdm_features_busyness": {
            "values": [calculate_ngtdm_features(prop.intensity_image)['busyness'] for prop in component_props],
            "range": (0, np.inf)
        },
        "ngtdm_features_complexity": {
            "values": [calculate_ngtdm_features(prop.intensity_image)['complexity'] for prop in component_props],
            "range": (0, np.inf)
        },
        "ngtdm_features_strength": {
            "values": [calculate_ngtdm_features(prop.intensity_image)['strength'] for prop in component_props],
            "range": (0, np.inf)
        },
        "ellipticity": {
            "values": [
                1 - (prop.minor_axis_length / prop.major_axis_length) if prop.major_axis_length > 0 else 0
                for prop in component_props
            ],
            "range": (0, 1)
        },
        "circularity": {
            "values": [
                (4 * np.pi * prop.area / (prop.perimeter ** 2)) if prop.perimeter > 0 else 0
                for prop in component_props
            ],
            "range": (0, 1)
        },
        "first_order_statistics_std": {
            "values": [np.std(prop.intensity_image.flatten()) for prop in component_props],
            "range": (0, np.inf)
        },
        "first_order_statistics_median": {
            "values": [np.median(prop.intensity_image.flatten()) for prop in component_props],
            "range": (0, np.inf)
        },
        "first_order_statistics_mode": {
            "values": [stats.mode(prop.intensity_image.flatten())[0][0] for prop in component_props],
            "range": (0, np.inf)
        },
        "first_order_statistics_energy": {
            "values": [np.sum(np.square(prop.intensity_image.flatten())) for prop in component_props],
            "range": (0, np.inf)
        },
        "first_order_statistics_entropy": {
            "values": [stats.entropy(prop.intensity_image.flatten()) for prop in component_props],
            "range": (0, np.inf)
        },
        "first_order_statistics_histogram_width": {
            "values": [np.max(prop.intensity_image.flatten()) - np.min(prop.intensity_image.flatten()) for prop in component_props],
            "range": (0, np.inf)
        }
    }
    
    # ---------- Basic features : geometric and intensity properties ----------
    # If basic_features is None, include all basic features
    if basic_features is None:
        basic_features = list(all_basic_features.keys())
        #print(f"Using all basic features: {basic_features}")
    
    # Only include the specified basic features
    for feature in basic_features:
        if feature in all_basic_features:
            #print(f"----- Computing feature: {feature} -----")
            values = all_basic_features[feature]["values"]
            bin_range = all_basic_features[feature]["range"]
            
            # change bin range to the min and max of the values
            bin_range = (min(values), max(values))    
            
            hist, bins = np.histogram(values, bins=np.linspace(*bin_range, num=n_bins), density=True)
            
            # get mean of the histogram
            mean_hist = np.mean(hist)
            
            # Multiply by bin width to get an approximation of the density integrated over each bin
            hist_norm = hist * (bins[1] - bins[0])
            
            # Add to feature vector
            feat_vector.append(mean_hist)
            #feat_vector.extend(hist_norm)
        else:
            print(f"Warning: Unknown basic feature '{feature}'. Skipping.")
        
    # ---------- Neighborhood features ----------
    if include_neighborhood:
        # Compute nearest neighbor histogram (spatial proximity of centroids)
        bins_nn = np.linspace(0, 30, num=n_bins)
        centroid_positions = np.array([prop.centroid for prop in component_props])
        
        try:
            hist_nn = first_neighbor_distance_histogram(centroid_positions, bins_nn)
    
            # get the mean of the histogram
            mean_hist_nn = np.mean(hist_nn)
            
            
            # Add to feature vector
            feat_vector.append(mean_hist_nn)
            #feat_vector.extend(hist_nn * (bins_nn[1] - bins_nn[0]))
        except Exception as e:
            print(f"Warning: Could not compute nearest neighbor features: {e}")
    
    # For ROI-based features, extract all ROIs first
    roi_data = []
    for prop in component_props:
        # Extract ROI using bounding box from regionprops
        minr, minc, maxr, maxc = prop.bbox
        
        roi_mask = prop.image  # binary mask of the region
        intensity_roi = image[minr:maxr, minc:maxc]  # intensity values
        
        roi_data.append({
            'mask': roi_mask,
            'intensity': intensity_roi,
            'bbox': (minr, minc, maxr, maxc)
        })
    
    # ---------- Intensity derivative ----------
    if include_intensity_deriv and roi_data:
        
        # cut intensity to keep only non zero values
        intensity_cut = intensity[intensity > 0]
        # compute the roughness
        rough = rms_roughness(intensity_cut)
        # Add to feature vector
        feat_vector.append(rough)
    
    # ---------- Zernike moments ----------
    if include_zernike and roi_data:
        zernike_radius = 10  
        zernike_list = []
        
        for roi in roi_data:
            try:
                zernike_vals = compute_zernike_features(roi['mask'], radius=zernike_radius)
                zernike_list.append(zernike_vals)
            except Exception as e:
                print(f"Warning: Could not compute Zernike moments: {e}")
                # Append zeros for this ROI
                zernike_list.append(np.zeros(len(roi_data)))  
        
        #print(f"Length of zernike_list: {len(zernike_list)}")
        
        
        # Standardize the data 
        scaler = StandardScaler()
        zernike_list_scaled = scaler.fit_transform(zernike_list)
        
        # use PCA to reduce the number of features of zernike moments
        pca = PCA(n_components=1)
        zernike_list_scaled = pca.fit_transform(zernike_list_scaled)
        
        #print(pca.explained_variance_ratio_)  # Shows how much each component explains
        #print(sum(pca.explained_variance_ratio_))  # Total variance retained
        
        #print(f"Shape of zernike_list after PCA: {zernike_list_scaled.shape}")

        
        if zernike_list_scaled is not None:
            mean_zernike = np.mean(np.vstack(zernike_list_scaled), axis=0)
            std_zernike = np.std(np.vstack(zernike_list_scaled), axis=0)
            median_zernike = np.median(np.vstack(zernike_list_scaled), axis=0)
        else:
            mean_zernike = np.zeros(3)  
            
        
        
        # Add to feature vector
        feat_vector.extend(mean_zernike)
        feat_vector.extend(std_zernike)
        #feat_vector.extend(median_zernike)
       
    # ---------- Hu moments ----------
    if include_hu and roi_data:
        hu_list = []
        
        for roi in roi_data:
            try:
                hu_vals = compute_hu_features(roi['mask'])
                hu_list.append(hu_vals)
            except Exception as e:
                print(f"Warning: Could not compute Hu moments: {e}")
                # Append zeros for this ROI
                hu_list.append(0)  
        
        # Standardize the data 
        scaler = StandardScaler()
        hu_list_scaled = scaler.fit_transform(hu_list)
        
        # use PCA to reduce the number of features of zernike moments
        pca = PCA(n_components=1)
        hu_list_scaled = pca.fit_transform(hu_list_scaled)
        
        #print(pca.explained_variance_ratio_)  # Shows how much each component explains
        #print(sum(pca.explained_variance_ratio_))  # Total variance retained
        
        
        if hu_list:
            mean_hu = np.mean(np.vstack(hu_list_scaled), axis=0)
            std_hu = np.std(np.vstack(hu_list_scaled), axis=0)
            median_hu = np.median(np.vstack(hu_list_scaled), axis=0)
        else:
            mean_hu = np.zeros(3)
            
        
        # Add to feature vector
        feat_vector.extend(mean_hu)
        feat_vector.extend(std_hu)
        #feat_vector.extend(median_hu)
    
    # ---------- HOG features ----------
    if include_hog and roi_data:
        hog_list = []
        expected_hog_length = None
        
        for roi in roi_data:
            try:
                hog_desc = compute_hog_features(roi['intensity'])
                
                # Set expected length based on first successful computation
                if expected_hog_length is None:
                    expected_hog_length = len(hog_desc)
                
                hog_list.append(hog_desc)
            except Exception as e:
                print(f"Warning: Could not compute HOG features: {e}")
        
        if hog_list:
            # Make sure all HOG descriptors have the same length
            hog_list = [h for h in hog_list if len(h) == expected_hog_length]
            if hog_list:
                mean_hog = np.mean(np.vstack(hog_list), axis=0)
            else:
                mean_hog = np.zeros(36)  # Default expected length for 64x64 image
        else:
            mean_hog = np.zeros(36)  # Default expected length for 64x64 image
        
        # Add to feature vector
        feat_vector.extend(mean_hog)
    
    
    return np.array(feat_vector)

def get_synapse_centers_using_hessian(region, image, sigma=2):
    """
    Detects synapse centers in a region using the Hessian matrix.
    
    Parameters:
    - region: Binary mask of the region.
    - image: Original grayscale image.
    - sigma: Scale for Hessian filter.
    
    Returns:
    - List of (x, y) coordinates for synapse centers.
    """
    # Compute Hessian matrix of the region
    hessian_elems = ski.feature.hessian_matrix(image, sigma=sigma, order='rc')
    hessian_eigenvals = ski.feature.hessian_matrix_eigvals(hessian_elems)
    
    # Step 2: Threshold negative eigenvalues
    eigenvalue1, eigenvalue2 = hessian_eigenvals[0], hessian_eigenvals[1]
        
    # Keep points where both eigenvalues are negative
    negative_eigenvalue_mask = (eigenvalue1 < 0) & (eigenvalue2 < 0)
        
    # Step 3: Find local maxima in the negative eigenvalue mask (intensity peaks)
    hessian_response = np.abs(eigenvalue1)  # or use the largest eigenvalue for intensity peaks
    local_maxima = ski.feature.peak_local_max(hessian_response, min_distance=3, exclude_border=False)
        
    # Step 4: Filter maxima based on the mask and thresholding condition
    synapse_centers = [(x, y) for x, y in local_maxima if region[x, y] > 0 and negative_eigenvalue_mask[x, y]]
    
    # Plot original vs. Hessian response
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(hessian_response, cmap='inferno')
    axes[1].set_title('Hessian Response')
    plt.show()
        

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    ax.scatter([y for x, y in synapse_centers], [x for x, y in synapse_centers], 
            color='red', s=20, label="Detected Centers")
    ax.set_title("Synapse Centers Overlaid on Image")
    ax.legend()
    plt.show()
    
    return synapse_centers

def get_regions_of_interest(coord, image_original, binary_mask):
    
    #print(f"Detected {len(coord)} synapse centers")
    
    # Step 1: Initial rough segmentation
    image_markers = np.zeros_like(image_original, dtype=np.int32)
    for i, (x, y) in enumerate(coord, 1):  # Start at 1 (0 is background)
        image_markers[int(x), int(y)] = i + 100  # Offset markers to avoid overlap
    
    rough_segmented = ski.segmentation.watershed(-image_original, connectivity=1, markers=image_markers, mask=binary_mask)
    refined_segmented = np.zeros_like(rough_segmented)
 
    for region in ski.measure.regionprops(rough_segmented, intensity_image=image_original):
        
        minr, minc, maxr, maxc = region.bbox  # Get bounding box of region
        mask = (rough_segmented[minr:maxr, minc:maxc] == region.label)  # Extract the region
        
        
        # Step 3: Compute mean boundary intensity
        boundary = ski.morphology.dilation(mask, ski.morphology.disk(1)) ^ mask  # Find boundary pixels
        if image_original[minr:maxr, minc:maxc][boundary].size > 0:
            mean_boundary_intensity = np.mean(image_original[minr:maxr, minc:maxc][boundary])
            #print(f"Mean boundary intensity for region {region.label}: {mean_boundary_intensity}")
        else:
            mean_boundary_intensity = 0
            #print(f"Warning: No boundary pixels found for region {region.label}")
            continue

        # Step 4: Construct new window
        new_window = image_original[minr:maxr, minc:maxc].copy()
        new_window[~mask] = mean_boundary_intensity  # Replace background with mean boundary intensity
        
        local_maxima = ski.feature.peak_local_max(new_window, min_distance=2, exclude_border=False)
            
        # Step 4: Filter maxima based on the mask and thresholding condition
        synapse_centers = [(x, y) for x, y in local_maxima if new_window[x, y] > mean_boundary_intensity]
        
        # add synapse_center to coord
        coord.extend(synapse_centers)
        
        
        """print(f"Detected {len(synapse_centers)} synapse centers in region {region.label}")
        print(synapse_centers)
        
        # plot the centers
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(new_window, cmap='gray')
        ax.scatter([y for x, y in synapse_centers], [x for x, y in synapse_centers],
                   color='red', s=20, label="Detected Centers")
        ax.set_title("Synapse Centers Overlaid on Image")
        ax.legend()
        plt.show()"""
        
    # erase from coord duplicate
    coord = list(set(coord)) 
    
    #print(f"Detected {len(coord)} synapse centers after second pass")
        
    # Step 1: Initial rough segmentation
    image_markers = np.zeros_like(image_original, dtype=np.int32)
    for i, (x, y) in enumerate(coord, 1):  # Start at 1 (0 is background)
        image_markers[int(x), int(y)] = i + 100  # Offset markers to avoid overlap
    
    rough_segmented = ski.segmentation.watershed(-image_original, connectivity=1, markers=image_markers, mask=binary_mask)
    refined_segmented = np.zeros_like(rough_segmented) 
    
    """# plot the centers
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_original, cmap='gray')
    ax.scatter([y for x, y in coord], [x for x, y in coord],
               color='red', s=2, label="Detected Centers")
    ax.set_title("Synapse Centers Overlaid on Image")
    ax.legend()
    plt.show()
    
    # plot rough_segmented
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rough_segmented, cmap='gray')
    ax.set_title("Rough Segmented")
    plt.show()"""

 
    # Step 2: Process each region
    for region in ski.measure.regionprops(rough_segmented, intensity_image=image_original):
        
        minr, minc, maxr, maxc = region.bbox  # Get bounding box of region
        mask = (rough_segmented[minr:maxr, minc:maxc] == region.label)  # Extract the region
        
        
        # Step 3: Compute mean boundary intensity
        boundary = ski.morphology.dilation(mask, ski.morphology.disk(1)) ^ mask  # Find boundary pixels
        if image_original[minr:maxr, minc:maxc][boundary].size > 0:
            mean_boundary_intensity = np.mean(image_original[minr:maxr, minc:maxc][boundary])
            #print(f"Mean boundary intensity for region {region.label}: {mean_boundary_intensity}")
        else:
            mean_boundary_intensity = 0
            #print(f"Warning: No boundary pixels found for region {region.label}")
            continue

        # Step 4: Construct new window
        new_window = image_original[minr:maxr, minc:maxc].copy()
        new_window[~mask] = mean_boundary_intensity  # Replace background with mean boundary intensity

        # Step 5: Apply K-means
        I = new_window / new_window.max()  # Normalize intensities
        B = mask.astype(float) * 0  # Binary weight
        features = np.column_stack((I.flatten(), B.flatten()))  # 2D feature space
        
        # Show feature space
        """plt.figure(figsize=(8, 6))
        plt.scatter(features[:, 0], features[:, 1], c=features[:, 0], cmap='viridis')
        plt.title(f"Feature space for region {region.label}")
        plt.xlabel("Intensity")
        plt.ylabel("Binary weight")
        plt.show()"""
        
        if features.shape[0] < 2:
            #print(f"Warning: Not enough features for region {region.label}. Skipping.")
            labels = np.zeros_like(features)
            continue
        
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        labels = kmeans.fit_predict(features)

        # Step 6: Determine foreground and update segmentation
        refined_region = labels.reshape(mask.shape)
        foreground_label = np.argmax([np.mean(I[refined_region == 0]), np.mean(I[refined_region == 1])])

        refined_mask = np.zeros_like(mask, dtype=np.int32)
        refined_mask[refined_region == foreground_label] = region.label  # Keep the correct label

        # Place refined region in final segmented image
        refined_segmented[minr:maxr, minc:maxc][refined_mask > 0] = region.label
        
        
        """# Create a figure with a specified size for the combined image
        fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # 1 row, 3 columns
        # Visualize region (first subplot)
        axes[0].imshow(mask, cmap='gray')
        axes[0].set_title(f"Region {region.label}")
        axes[0].axis('off') # Turn off axis labels and ticks
        # Visualize new window (second subplot)
        axes[1].imshow(new_window, cmap='gray')
        axes[1].set_title(f"New window for region {region.label} with mean boundary intensity for background")
        axes[1].axis('off')
        # Visualize refined region (third subplot)
        axes[2].imshow(refined_mask, cmap='gray')
        axes[2].set_title(f"Refined region {region.label}")
        axes[2].axis('off')
        # Adjust layout to prevent overlapping titles
        plt.tight_layout()
        # Show the combined image
        plt.show()"""
        

    # Step 7: Compute refined region properties
    region_props = ski.measure.regionprops(refined_segmented, intensity_image=image_original)
    
    #print(f"Detected {len(region_props)} synapses")

    return region_props, refined_segmented

def get_feature_vector(X, y, X_orig, max_images, mask_images, intensity, recompute=False, pkl_name=DEFAULT_PKL_NAME, n_features=N_FEAT, n_bins=N_BINS_FEAT):
    """
    Create feature vectors from preprocessed images.
    
    Parameters
    ----------
    X : ndarray
        Preprocessed images.
    y : ndarray
        Labels.
    recompute : bool
        If True, recompute features. Otherwise, try loading from file.
    pkl_name : str
        Base name for the features pkl file.
    n_features : int
        Number of original features.
    n_bins : int
        Number of bins for histograms.
        
    Returns
    -------
    ndarray
        Feature vectors.
    dict
        Features dictionary.
    """
    DATASET_PKL_DIR.mkdir(exist_ok=True)
    features_file_name = f'{Path(pkl_name).stem}_features.pkl'
    
    # if name already exists, add a number to the name
    i = 1
    while (DATASET_PKL_DIR / features_file_name).exists():
        features_file_name = f'{Path(pkl_name).stem}_{i}_features.pkl'
        i += 1
    
    if not recompute:
        try:
            features = joblib.load(DATASET_PKL_DIR / features_file_name)
            print('Features loaded from file.')
            X_feat = np.array(features['data'])
            return X_feat, features
        except FileNotFoundError:
            print('Features file not found. Recomputing...')
            recompute = True
    
    if recompute:
        print('Computing features...')
        X_feat = []
        features = {
            'description': 'C elegans images features (extended)',
            'label': [],
            'data': [],
            'filename': [],
            'components': [],
            'label_components': []
        }
        BASIC_FEATURES = []
        if INCLUDE_MAJOR_AXIS_LENGTH:
            BASIC_FEATURES.append('major_axis_length')
        if INCLUDE_AXIS_RATIO:
            BASIC_FEATURES.append('axis_ratio')
        if INCLUDE_EXTENT:
            BASIC_FEATURES.append('extent')
        if INCLUDE_AREA:
            BASIC_FEATURES.append('area')
        if INCLUDE_PERIMETER:
            BASIC_FEATURES.append('perimeter')      
        if INCLUDE_CONVEX_AREA:
            BASIC_FEATURES.append('convex_area')
        if INCLUDE_SOLIDITY:
            BASIC_FEATURES.append('solidity')
        if INCLUDE_ECCENTRICITY:
            BASIC_FEATURES.append('eccentricity')
        if INCLUDE_MEAN_INTENSITY:  
            BASIC_FEATURES.append('mean_intensity')
        if INCLUDE_MAX_INTENSITY:
            BASIC_FEATURES.append('max_intensity')
        if INCLUDE_MIN_INTENSITY:
            BASIC_FEATURES.append('min_intensity')
        if INCLUDE_GVSD:
            BASIC_FEATURES.append('gvsd')
        if INCLUDE_SKEWNESS:
            BASIC_FEATURES.append('skewness')
        if INCLUDE_KURTOSIS:
            BASIC_FEATURES.append('kurtosis')
        if INCLUDE_CIRCULARITY:
            BASIC_FEATURES.append('circularity')
        if INCLUDE_ELLIPTICITY:
            BASIC_FEATURES.append('ellipticity')
        if INCLUDE_CONTRAST_GLCM:
            BASIC_FEATURES.append('contrast_GLCM')
        if INCLUDE_DISSIMILARITY_GLCM:
            BASIC_FEATURES.append('dissimilarity_GLCM')
        if INCLUDE_HOMOGENEITY_GLCM:
            BASIC_FEATURES.append('homogeneity_GLCM')
        if INCLUDE_ENERGY_GLCM:
            BASIC_FEATURES.append('energy_GLCM')
        if INCLUDE_CORRELATION_GLCM:
            BASIC_FEATURES.append('correlation_GLCM')


        
        for im_num, image in enumerate(X):
            print(f'Extracting features for image {im_num+1}/{len(X)}')
            image_original = X_orig[im_num]
            maxima = max_images[im_num]
            mask = mask_images[im_num]
        
            component, label_seg = get_regions_of_interest(maxima, image_original, mask)
            feat = create_feature_vector(image, component, intensity[im_num], n_bins, 
                                         basic_features=BASIC_FEATURES, include_neighborhood=INCLUDE_NEIGHBORHOOD,
                                         include_intensity_deriv=INCLUDE_INTENSITY, include_zernike=INCLUDE_ZERNIKE, 
                                         include_hog=INCLUDE_HOG, include_hu=INCLUDE_HU)
            
            features['label'].append(y[im_num])
            features['data'].append(feat)
            features['filename'].append(f"image_{im_num}")
            features['components'].append(component)
            features['label_components'].append(label_seg)
            
            X_feat.append(feat)
        
        X_feat = np.array(X_feat)
        print('Features computed.')
    
        # Save features in pkl file
        """joblib.dump(features, features_file_name)
        shutil.move(features_file_name, DATASET_PKL_DIR)
        print(f'Features computed and saved to {DATASET_PKL_DIR / features_file_name}')"""
        
        return X_feat, features

# ---------- Feature selection ----------

def select_features(X, y, k=10, method='mRMR', verbose_features_selected=False, feature_names=None):
    """
    Select the top k features using specified method and show selected features.

    Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features).
        y (array-like): Target vector.
        k (int): Number of top features to select (for 'kbest').
        method (str): Feature selection method ('kbest', 'boruta', 'lasso').
        feature_names (list, optional): List of feature names.

    Returns:
        X_new (array-like): The reduced feature matrix.
        selector (object): The fitted feature selector.
        selected_feature_names (list, optional): List of selected feature names.
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    if method == 'kbest':
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_indices]
        # save indices of selected features in a file "selected_features.txt" in 'models' folder
        with open('models/selected_features.txt', 'w') as f:
            counter = 0
            for item in selected_indices:
                if item == True:
                    # write the indices of the selected features
                    f.write("%s\n" % counter)
                counter += 1
        if verbose_features_selected : print(f"Selected features (kbest): {selected_feature_names}")
        return X_new, selector

    elif method == 'boruta':
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=42) 
        boruta_selector.fit(X, y)
        selected_indices = boruta_selector.support_
        selected_feature_names = [feature_names[i] for i, selected in enumerate(selected_indices) if selected]
        if verbose_features_selected : print(f"Selected features (boruta): {selected_feature_names}")
        # save indices of selected features in a file "selected_features.txt" in 'models' folder
        with open('models/selected_features.txt', 'w') as f:
            counter = 0
            for item in selected_indices:
                if item == True:
                    # write the indices of the selected features
                    f.write("%s\n" % counter)
                counter += 1
        return X[:, selected_indices], boruta_selector

    elif method == 'lasso':
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)
        mask = lasso.coef_ != 0
        selected_indices = np.where(mask)[0]
        selected_feature_names = [feature_names[i] for i in selected_indices]
        # save indices of selected features in a file "selected_features.txt" in 'models' folder
        with open('models/selected_features.txt', 'w') as f:
            counter = 0
            for item in selected_indices:
                if item == True:
                    # write the indices of the selected features
                    f.write("%s\n" % counter)
                counter += 1
        if verbose_features_selected :
            print(f"Selected {np.sum(mask)} features out of {X.shape[1]}")
            print(f"Selected features (lasso): {selected_feature_names}")
        return X[:, mask], lasso

    elif method == 'mRMR':
        
        # minimumRedundancyMaximumRelevance(mRMR) feature selection
        
        # Convert numpy array to pandas DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=feature_names if feature_names else [f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
            
        print(X_df)
        
        # Initialize MRMR selector
        # method 1 : F-Statistic
        #mrmr_selector = MRMR(method="FCQ", max_features=8, regression=False)
        # method 2 : Random forest
        mrmr_selector = MRMR(method="RFCQ",max_features=None, scoring="roc_auc",param_grid = {"n_estimators": [5, 30, 100], "max_depth":[1,2,3]},cv=3,regression=False, random_state=42)
        # method 3 : Mutual information
        #mrmr_selector = MRMR(method="MIQ", max_features=11, regression=False) 
        
        # Fit and transform the data
        mrmr_selector.fit(X_df, y)

        
        # plot the relevance
        pd.Series(mrmr_selector.relevance_, index=mrmr_selector.variables_).sort_values(
            ascending=False).plot.bar(figsize=(15, 6))
        plt.title("Relevance")
        plt.show()
        
        # transform the data to keep only the selected features 
        X_df = mrmr_selector.transform(X_df) 
        
        print(f"Selected features (mRMR): {X_df.columns}")
   
      
        # Get indices of selected features
        selected_indices = mrmr_selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
        # Save selected feature indices
        with open('models/selected_features.txt', 'w') as f:
            counter = 0
            for item in selected_indices:
                if item == True:
                    # write the indices of the selected features
                    f.write("%s\n" % counter)
                counter += 1
        
        # Print selected features if verbose mode is on
        if verbose_features_selected:
            print(f"Selected features (mRMR): {selected_feature_names}")
        
        # Return selected features and their indices
        return X[:, selected_indices], selected_indices
           
    else:
        print("Method must be 'kbest', 'boruta', 'mRMR' or 'lasso'.")
        return X, None

