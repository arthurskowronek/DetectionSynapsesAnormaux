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
from scipy.spatial import distance_matrix

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

def create_feature_vector(image, component_props, intensity=None, n_bins=20,
                         include_texture=True, include_morphological=True,
                         include_histogram=True, include_multiscale=True,
                         include_other=True, verbose_warning=False):
    """
    Create a comprehensive feature vector from a preprocessed image.
    
    The feature vector includes:
      - Texture features (FOS, GLCM, GLDS, NGTDM, SFM, LTE, FDTA, GLRLM, FPS, Shape, HOS, LBP, GLSZM)
      - Morphological features (Grayscale, Binary)
      - Histogram-based features (Histogram, MultiregionHistogram, Correlogram)
      - Multi-scale features (DWT, SWT, WP, GT, AMFM)
      - Other features (HOG, Hu Moments, TAS, Zernikes Moments)
    
    Parameters
    ----------
    image : ndarray
        Preprocessed intensity image.
    component_props : list
        List of region properties (from skimage.measure.regionprops).
    intensity : ndarray, optional
        Intensity image, if different from the image used for segmentation.
    n_bins : int, default=20
        Number of bins for histograms.
    include_texture : bool, default=True
        Whether to include texture features.
    include_morphological : bool, default=True
        Whether to include morphological features.
    include_histogram : bool, default=True
        Whether to include histogram-based features.
    include_multiscale : bool, default=True
        Whether to include multi-scale features.
    include_other : bool, default=True
        Whether to include other features (HOG, Hu Moments, etc.).
        
    Returns
    -------
    ndarray
        Comprehensive feature vector.
    """
    # Initialize empty feature vector
    feat_vector = []

    if not component_props:
        print("Warning: No components found in image.")
        # Return an empty feature vector
        return np.array([0])
    
    # Always include number of detected components
    num_components = len(component_props)
    feat_vector = [num_components]
    
    # Get intensity image if not provided
    if intensity is None:
        intensity = image
    
    # For component-based processing, we'll compute features for each component
    # and then aggregate them (mean, std, etc.)
    component_features = []
    
    for prop in component_props:
        # Extract ROI using bounding box from regionprops
        minr, minc, maxr, maxc = prop.bbox
        roi_mask = prop.image  # binary mask of the region
        roi_intensity = image[minr:maxr, minc:maxc]  # intensity values
        
        # Create dictionary to store features for this component
        roi_features = {}
        
        # ---------- A. Texture Features ----------
        if include_texture:
            try:
                # First Order Statistics
                roi_features['A_FOS'], name = pyfeats.fos(roi_intensity, roi_mask)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing FOS features: {e}")
                roi_features['A_FOS'] = np.zeros(16)
                
            try:    
                # Gray Level Co-occurrence Matrix
                roi_features['A_GLCM'], name = pyfeats.glcm_features(roi_intensity, ignore_zeros=True)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing GLCM features: {e}")
                roi_features['A_GLCM'] = np.zeros(13)
                
            try:     
                # Gray Level Difference Statistics
                roi_features['A_GLDS'], name = pyfeats.glds_features(roi_intensity, roi_mask, 
                                                              Dx=[0,1,1,1], Dy=[1,1,0,-1])
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing GLDS features: {e}")
                roi_features['A_GLDS'] = np.zeros(5)
                
            try:     
                # Neighborhood Gray Tone Difference Matrix
                roi_features['A_NGTDM'], name = pyfeats.ngtdm_features(roi_intensity, roi_mask, d=1)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing NGTDM features: {e}")
                roi_features['A_NGTDM'] = np.zeros(5)
                
            try:     
                # Statistical Feature Matrix
                roi_features['A_SFM'], name = pyfeats.sfm_features(roi_intensity, roi_mask, Lr=4, Lc=4)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing SFM features: {e}")
                roi_features['A_SFM'] = np.zeros(4)
                
            try:     
                # Laws Texture Energy
                roi_features['A_LTE'], name = pyfeats.lte_measures(roi_intensity, roi_mask, l=7)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing LTE features: {e}")
                roi_features['A_LTE'] = np.zeros(6)
                
            try:     
                # Fractal Dimension Texture Analysis
                roi_features['A_FDTA'], name = pyfeats.fdta(roi_intensity, roi_mask, s=3)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing FDTA features: {e}")
                roi_features['A_FDTA'] = np.zeros(4)
                
            try:     
                # Gray Level Run Length Matrix
                roi_features['A_GLRLM'], name = pyfeats.glrlm_features(roi_intensity, roi_mask, Ng=256)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing GLRLM features: {e}")
                roi_features['A_GLRLM'] = np.zeros(11)
                
            try:     
                # Fourier Power Spectrum
                roi_features['A_FPS'], name = pyfeats.fps(roi_intensity, roi_mask)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing FPS features: {e}")
                roi_features['A_FPS'] = np.zeros(2)
                
            try:     
                # Shape Parameters
                # Calculate perimeter first
                perimeter = prop.perimeter
                roi_features['A_Shape_Parameters'], name = pyfeats.shape_parameters(
                    roi_intensity, roi_mask, perimeter, pixels_per_mm2=1)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing shape parameters features: {e}")
                roi_features['A_Shape_Parameters'] = np.zeros(5)
                
            try:     
                # Higher Order Spectra
                # Using adaptive thresholds based on intensity range
                intensity_min = np.min(roi_intensity)
                intensity_max = np.max(roi_intensity)
                th_low = intensity_min + 0.4 * (intensity_max - intensity_min)
                th_high = intensity_min + 0.6 * (intensity_max - intensity_min)
                roi_features['A_HOS'], name = pyfeats.hos_features(roi_intensity, th=[th_low, th_high])
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing HOS features: {e}")
                roi_features['A_HOS'] = np.zeros(2)
                
            try:     
                # Local Binary Patterns
                roi_features['A_LBP'], name = pyfeats.lbp_features(roi_intensity, roi_intensity, 
                                                            P=[8,16,24], R=[1,2,3])
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing LPB features: {e}")
                roi_features['A_LBP'] = np.zeros(6)
                
            try:     
                # Gray Level Size Zone Matrix
                roi_features['A_GLSZM'], name = pyfeats.glszm_features(roi_intensity, roi_mask)   
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing GLSZM features: {e}")
                roi_features['A_GLSZM'] = np.zeros(14)
        
        # ---------- B. Morphological Features ----------
        if include_morphological:
            try:
                # Grayscale Morphology
                roi_features['B_Morphological_Grayscale_pdf'], roi_features['B_Morphological_Grayscale_cdf'] = \
                    pyfeats.grayscale_morphology_features(roi_intensity, N=30)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing morphological grayscale features: {e}")
                roi_features['B_Morphological_Grayscale_pdf'] = np.zeros(30)
                roi_features['B_Morphological_Grayscale_cdf'] = np.zeros(30)
                
            try:    
                # Multi-level Binary Morphology
                # Adaptive thresholds based on intensity range
                intensity_min = np.min(roi_intensity)
                intensity_max = np.max(roi_intensity)
                th_low = intensity_min + 0.25 * (intensity_max - intensity_min)
                th_high = intensity_min + 0.5 * (intensity_max - intensity_min)
                
                roi_features['B_Morphological_Binary_L_pdf'], roi_features['B_Morphological_Binary_M_pdf'], \
                roi_features['B_Morphological_Binary_H_pdf'], roi_features['B_Morphological_Binary_L_cdf'], \
                roi_features['B_Morphological_Binary_M_cdf'], roi_features['B_Morphological_Binary_H_cdf'] = \
                    pyfeats.multilevel_binary_morphology_features(
                        roi_intensity, roi_mask, N=30, thresholds=[th_low, th_high])
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing morphological Binary features: {e}")
                roi_features['B_Morphological_Binary_L_pdf'] = np.zeros(30)
                roi_features['B_Morphological_Binary_M_pdf'] = np.zeros(30)
                roi_features['B_Morphological_Binary_H_pdf'] = np.zeros(30)
                roi_features['B_Morphological_Binary_L_cdf'] = np.zeros(30)
                roi_features['B_Morphological_Binary_M_cdf'] = np.zeros(30)
                roi_features['B_Morphological_Binary_H_cdf'] = np.zeros(30)
        
        # ---------- C. Histogram-based Features ----------
        if include_histogram:
            try:
                # Basic Histogram
                roi_features['C_Histogram'] = pyfeats.histogram(roi_intensity, roi_mask, bins=32)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing C histogram features: {e}")
                roi_features['C_Histogram'] = np.zeros(32)
                
            try:    
                # Multi-region Histogram
                roi_features['C_MultiregionHistogram'] = pyfeats.multiregion_histogram(
                    roi_intensity, roi_mask, bins=32, num_eros=3, square_size=3)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing C MultiregionHistogram features: {e}")
                roi_features['C_MultiregionHistogram'] = np.zeros(32 * 4)  # Original + 3 erosions
            
            try:    
                # Correlogram
                roi_features['C_Correlogram'] = pyfeats.correlogram(
                    roi_intensity, roi_mask, bins_digitize=32, bins_hist=32, flatten=True) 
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing C Correlogram features: {e}")
                roi_features['C_Correlogram'] = np.zeros(32 * 32)  # flattened correlogram
        
        # ---------- D. Multi-scale Features ----------
        if include_multiscale:
            try:
                # Discrete Wavelet Transform
                roi_features['D_DWT'] = pyfeats.dwt_features(
                    roi_intensity, roi_mask, wavelet='bior3.3', levels=3)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing multi-scale D_DWT features: {e}")
                roi_features['D_DWT'] = np.zeros(7 * 3 * 4)  # 7 stats, 3 levels, 4 subbands
                
            try:    
                # Stationary Wavelet Transform
                roi_features['D_SWT'] = pyfeats.swt_features(
                    roi_intensity, roi_mask, wavelet='bior3.3', levels=3)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing multi-scale D_SWT features: {e}")
                roi_features['D_SWT'] = np.zeros(7 * 3 * 4)  # 7 stats, 3 levels, 4 subbands
                
            try:    
                # Wavelet Packet
                roi_features['D_WP'] = pyfeats.wp_features(
                    roi_intensity, roi_mask, wavelet='coif1', maxlevel=3)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing multi-scale D_WP features: {e}")
                roi_features['D_WP'] = np.zeros(7 * (4**3))  # 7 stats, 4^3 nodes at level 3
                
            try:    
                # Gabor Transform
                roi_features['D_GT'] = pyfeats.gt_features(roi_intensity, roi_mask)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing multi-scale D_GT features: {e}")
                roi_features['D_GT'] = np.zeros(7 * 4 * 6)  # 7 stats, 4 scales, 6 orientations
            
            try:    
                # Amplitude-Modulation Frequency-Modulation
                roi_features['D_AMFM'] = pyfeats.amfm_features(roi_intensity)  
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing multi-scale D_AMFM features: {e}")
                roi_features['D_AMFM'] = np.zeros(12)  # AMFM feature dimension
        
        # ---------- E. Other Features ----------
        if include_other:
            try:
                # HOG Features
                roi_features['E_HOG'] = pyfeats.hog_features(roi_intensity, ppc=8, cpb=3)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing HOG features: {e}")
                roi_features['E_HOG'] = np.zeros(36)  # HOG feature dimension
                
            try:    
                # Hu Moments
                roi_features['E_HuMoments'] = pyfeats.hu_moments(roi_intensity)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing HU features: {e}")
                roi_features['E_HuMoments'] = np.zeros(7)  # 7 Hu moments
                  
            try:  
                # Threshold Adjacency Statistics
                roi_features['E_TAS'] = pyfeats.tas_features(roi_intensity)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing TAS features: {e}")
                roi_features['E_TAS'] = np.zeros(9)  # TAS feature dimension
                
            try:   
                # Zernike Moments
                # Determine appropriate radius based on ROI size
                min_dim = min(roi_intensity.shape)
                radius = min(9, min_dim // 2)  # Use smaller of 9 or half the minimum dimension
                roi_features['E_ZernikesMoments'] = pyfeats.zernikes_moments(
                    roi_intensity, radius=radius)
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing Zernike features: {e}")
                roi_features['E_ZernikesMoments'] = np.zeros(25)  # Zernike moments up to order 9
        
        """print("---------- Extracted features for component -----------")
        # print dictionnary roi_features in a nice format
        for key, value in roi_features.items():
            print(f"{key}: {value}")
        print("-------------------------------------------------------")"""
        
        # Add this component's features to the list
        component_features.append(roi_features)
            
    # ---------- Aggregate component features ----------
    # For each feature type, compute mean, std, min, max across all components
    if component_features:
        aggregated_features = {}
        
        # List all feature keys from the first component
        feature_keys = component_features[0].keys()
        
        for key in feature_keys:
            # Stack this feature from all components
            stacked_feature = np.vstack([comp[key] for comp in component_features 
                                         if comp[key] is not None and len(comp[key]) > 0])
            
            if stacked_feature.size > 0:
                # Compute statistics
                mean_feature = np.mean(stacked_feature, axis=0)
                std_feature = np.std(stacked_feature, axis=0)
                min_feature = np.min(stacked_feature, axis=0)
                max_feature = np.max(stacked_feature, axis=0)
                
                # Append to feature vector
                feat_vector.extend(mean_feature)
                feat_vector.extend(std_feature)
                # Only include min/max for key features to control vector length, no need for other features
                if key.startswith(('A_FOS', 'A_GLCM', 'A_NGTDM', 'E_HuMoments')):
                    feat_vector.extend(min_feature)
                    feat_vector.extend(max_feature)
    
    # Add spatial distribution features
    try:
        # Compute centroid positions and Ripley's K function
        centroid_positions = np.array([prop.centroid for prop in component_props])
        if len(centroid_positions) > 1:
            # Compute nearest neighbor distances
            dist_matrix = distance_matrix(centroid_positions, centroid_positions)
            np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances
            nearest_neighbor_dists = np.min(dist_matrix, axis=1)
            
            # Calculate statistics of distances
            mean_nn_dist = np.mean(nearest_neighbor_dists)
            std_nn_dist = np.std(nearest_neighbor_dists)
            min_nn_dist = np.min(nearest_neighbor_dists)
            max_nn_dist = np.max(nearest_neighbor_dists)
            
            # Add to feature vector
            feat_vector.extend([mean_nn_dist, std_nn_dist, min_nn_dist, max_nn_dist])
            
            # Try to compute Ripley's K if pointpats is available
            try:
                max_dist = np.max(nearest_neighbor_dists) * 2
                radii = np.linspace(0, max_dist, 5)
                ripley_k_values = []
                
                for r in radii:
                    if r > 0:
                        count = np.sum(dist_matrix < r, axis=1)
                        k_r = np.mean(count) / (len(centroid_positions) - 1)
                        ripley_k_values.append(k_r)
                
                feat_vector.extend(ripley_k_values)
            except Exception as e:
                print(f"Warning: Error computing Ripley's K: {e}")
                feat_vector.extend(np.zeros(5))  # Placeholder for Ripley's K
        else:
            # Add zeros for spatial features if too few components
            feat_vector.extend(np.zeros(9))  # 4 for NN distances, 5 for Ripley's K
    except Exception as e:
        print(f"Warning: Error computing spatial distribution features: {e}")
        feat_vector.extend(np.zeros(9))  # Placeholder for spatial features
    
    
    # Return the final feature vector as a numpy array
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
        
        for im_num, image in enumerate(X):
            print(f'Extracting features for image {im_num+1}/{len(X)}')
            image_original = X_orig[im_num]
            maxima = max_images[im_num]
            mask = mask_images[im_num]
        
            component, label_seg = get_regions_of_interest(maxima, image_original, mask)
            feat = create_feature_vector(image, component, intensity[im_num], n_bins, 
                                         include_texture=True, include_morphological=False,
                                         include_histogram=False, include_multiscale=False,
                                         include_other=False)
            
                         
            
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

