import scipy
import joblib
import pyfeats
import numpy as np
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt
from pathlib import Path
from boruta import BorutaPy  
from scipy.spatial import distance_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
from feature_engine.selection import MRMR
import warnings

from constants import *


# ---------- Utility functions ----------

def two_neighbors_distance_histogram(G, bins):

    # Compute the distance matrix
    dist_matrix = distance_matrix(G.nodes, G.nodes)
    
    # Get the distances of neighboring nodes
    neighbor_distances = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            if node != neighbor:
                neighbor_distances.append(dist_matrix[node, neighbor])
                
    # Compute the histogram
    hist, _ = np.histogram(neighbor_distances, bins=bins)
    
    return hist

# ---------- Feature extraction ----------

def create_feature_vector(G, mean_intensity, median_width, Measure_diff_slice, Measure_diff_points_segment, image, component_props, intensity=None, n_bins=20,
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
        return np.array([0]), ["No Components Found"]
    
    # Always include number of detected components
    num_components = len(component_props)
    feat_vector = [num_components]
    feature_names = ["Number of Components"]
    
    # Get intensity image if not provided
    if intensity is None:
        intensity = image
    
    # For component-based processing, we'll compute features for each component
    # and then aggregate them (mean, std, etc.)
    component_features = []
    
    # Define expected dimensions for each feature type
    # This will help us handle inconsistencies
    expected_dimensions = {
        'A_FOS': 16,
        'A_GLCM': 13,
        'A_GLDS': 5,
        'A_NGTDM': 5,
        'A_SFM': 4,
        'A_LTE': 6,
        'A_FDTA': 4,
        'A_GLRLM': 11,
        'A_FPS': 2,
        'A_Shape_Parameters': 5,
        'A_HOS': 2,
        'A_LBP': 6,
        'A_GLSZM': 14,
        'B_Morphological_Grayscale_pdf': 30,
        'B_Morphological_Grayscale_cdf': 30,
        'B_Morphological_Binary_L_pdf': 30,
        'B_Morphological_Binary_M_pdf': 30,
        'B_Morphological_Binary_H_pdf': 30,
        'B_Morphological_Binary_L_cdf': 30,
        'B_Morphological_Binary_M_cdf': 30,
        'B_Morphological_Binary_H_cdf': 30,
        'C_Histogram': 32,
        'C_MultiregionHistogram': 32 * 4,  # 128
        'C_Correlogram': 32 * 32,          # 1024
        'D_DWT': 7 * 3 * 4,                # 84
        'D_SWT': 7 * 3 * 4,                # 84 
        'D_WP': 7 * (4**3),                # 7 * 64 = 448
        'D_GT': 7 * 4 * 6,                 # 168
        'D_AMFM': 12,
        'E_HOG': 36,
        'E_HuMoments': 7,
        'E_TAS': 9,
        'E_ZernikesMoments': 25
    }
    
    for prop in component_props:
        # Extract ROI using bounding box from regionprops
        minr, minc, maxr, maxc = prop.bbox
        roi_mask = prop.image  # binary mask of the region
        roi_intensity = image[minr:maxr, minc:maxc]  # intensity values
        # subtract the mean intensity from the ROI
        roi_intensity = roi_intensity - mean_intensity 
        roi_intensity = np.clip(roi_intensity, 0, None)  # Ensure non-negative values
        
        # Create dictionary to store features for this component
        roi_features = {}
        
        # ---------- A. Texture Features ----------
        if include_texture:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore tous les warnings à l'intérieur du bloc
                try:
                    # First Order Statistics
                    features, name = pyfeats.fos(roi_intensity, roi_mask)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_FOS']:
                        if verbose_warning:
                            print(f"Warning: FOS features dimension mismatch. Expected {expected_dimensions['A_FOS']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_FOS'])
                    roi_features['A_FOS'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing FOS features: {e}")
                    roi_features['A_FOS'] = np.zeros(expected_dimensions['A_FOS'])
                    
                try:    
                    # Gray Level Co-occurrence Matrix
                    features, name = pyfeats.glcm_features(roi_intensity, ignore_zeros=True)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_GLCM']:
                        if verbose_warning:
                            print(f"Warning: GLCM features dimension mismatch. Expected {expected_dimensions['A_GLCM']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_GLCM'])
                    roi_features['A_GLCM'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing GLCM features: {e}")
                    roi_features['A_GLCM'] = np.zeros(expected_dimensions['A_GLCM'])
                    
                try:     
                    # Gray Level Difference Statistics
                    features, name = pyfeats.glds_features(roi_intensity, roi_mask, 
                                                                Dx=[0,1,1,1], Dy=[1,1,0,-1])
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_GLDS']:
                        if verbose_warning:
                            print(f"Warning: GLDS features dimension mismatch. Expected {expected_dimensions['A_GLDS']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_GLDS'])
                    roi_features['A_GLDS'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing GLDS features: {e}")
                    roi_features['A_GLDS'] = np.zeros(expected_dimensions['A_GLDS'])
                    
                try:     
                    # Neighborhood Gray Tone Difference Matrix
                    features, name = pyfeats.ngtdm_features(roi_intensity, roi_mask, d=1)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_NGTDM']:
                        if verbose_warning:
                            print(f"Warning: NGTDM features dimension mismatch. Expected {expected_dimensions['A_NGTDM']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_NGTDM'])
                    roi_features['A_NGTDM'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing NGTDM features: {e}")
                    roi_features['A_NGTDM'] = np.zeros(expected_dimensions['A_NGTDM'])
                    
                try:     
                    # Statistical Feature Matrix
                    features, name = pyfeats.sfm_features(roi_intensity, roi_mask, Lr=4, Lc=4)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_SFM']:
                        if verbose_warning:
                            print(f"Warning: SFM features dimension mismatch. Expected {expected_dimensions['A_SFM']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_SFM'])
                    roi_features['A_SFM'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing SFM features: {e}")
                    roi_features['A_SFM'] = np.zeros(expected_dimensions['A_SFM'])
                    
                try:     
                    # Laws Texture Energy
                    features, name = pyfeats.lte_measures(roi_intensity, roi_mask, l=7)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_LTE']:
                        if verbose_warning:
                            print(f"Warning: LTE features dimension mismatch. Expected {expected_dimensions['A_LTE']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_LTE'])
                    roi_features['A_LTE'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing LTE features: {e}")
                    roi_features['A_LTE'] = np.zeros(expected_dimensions['A_LTE'])
                    
                try:     
                    # Fractal Dimension Texture Analysis
                    features, name = pyfeats.fdta(roi_intensity, roi_mask, s=3)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_FDTA']:
                        if verbose_warning:
                            print(f"Warning: FDTA features dimension mismatch. Expected {expected_dimensions['A_FDTA']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_FDTA'])
                    roi_features['A_FDTA'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing FDTA features: {e}")
                    roi_features['A_FDTA'] = np.zeros(expected_dimensions['A_FDTA'])
                    
                try:     
                    # Gray Level Run Length Matrix
                    features, name = pyfeats.glrlm_features(roi_intensity, roi_mask, Ng=256)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_GLRLM']:
                        if verbose_warning:
                            print(f"Warning: GLRLM features dimension mismatch. Expected {expected_dimensions['A_GLRLM']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_GLRLM'])
                    roi_features['A_GLRLM'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing GLRLM features: {e}")
                    roi_features['A_GLRLM'] = np.zeros(expected_dimensions['A_GLRLM'])
                    
                try:     
                    # Fourier Power Spectrum
                    features, name = pyfeats.fps(roi_intensity, roi_mask)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_FPS']:
                        if verbose_warning:
                            print(f"Warning: FPS features dimension mismatch. Expected {expected_dimensions['A_FPS']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_FPS'])
                    roi_features['A_FPS'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing FPS features: {e}")
                    roi_features['A_FPS'] = np.zeros(expected_dimensions['A_FPS'])
                    
                try:     
                    # Shape Parameters
                    # Calculate perimeter first
                    perimeter = prop.perimeter
                    features, name = pyfeats.shape_parameters(
                        roi_intensity, roi_mask, perimeter, pixels_per_mm2=1)
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    if median_width != 0:
                        features = features / median_width
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_Shape_Parameters']:
                        if verbose_warning:
                            print(f"Warning: Shape Parameters features dimension mismatch. Expected {expected_dimensions['A_Shape_Parameters']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_Shape_Parameters'])
                    roi_features['A_Shape_Parameters'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing shape parameters features: {e}")
                    roi_features['A_Shape_Parameters'] = np.zeros(expected_dimensions['A_Shape_Parameters'])
                    
                try:     
                    # Higher Order Spectra
                    # Using adaptive thresholds based on intensity range
                    intensity_min = np.min(roi_intensity)
                    intensity_max = np.max(roi_intensity)
                    th_low = intensity_min + 0.4 * (intensity_max - intensity_min)
                    th_high = intensity_min + 0.6 * (intensity_max - intensity_min)
                    features, name = pyfeats.hos_features(roi_intensity, th=[th_low, th_high])
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_HOS']:
                        if verbose_warning:
                            print(f"Warning: HOS features dimension mismatch. Expected {expected_dimensions['A_HOS']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_HOS'])
                    roi_features['A_HOS'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing HOS features: {e}")
                    roi_features['A_HOS'] = np.zeros(expected_dimensions['A_HOS'])
                    
                try:     
                    # Local Binary Patterns
                    roi_int_uint8 = (roi_intensity * 255 / roi_intensity.max()).astype(np.uint8)
                    features, name = pyfeats.lbp_features(roi_int_uint8, roi_int_uint8, 
                                                                P=[8,16,24], R=[1,2,3]) 
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_LBP']:
                        if verbose_warning:
                            print(f"Warning: LBP features dimension mismatch. Expected {expected_dimensions['A_LBP']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_LBP'])
                    roi_features['A_LBP'] = features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing LPB features: {e}")
                    roi_features['A_LBP'] = np.zeros(expected_dimensions['A_LBP'])
                    
                try:     
                    # Gray Level Size Zone Matrix
                    features, name = pyfeats.glszm_features(roi_intensity, roi_mask) 
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure correct dimension
                    if len(features) != expected_dimensions['A_GLSZM']:
                        if verbose_warning:
                            print(f"Warning: GLSZM features dimension mismatch. Expected {expected_dimensions['A_GLSZM']}, got {len(features)}")
                        features = np.zeros(expected_dimensions['A_GLSZM'])
                    roi_features['A_GLSZM'] = features  
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing GLSZM features: {e}")
                    roi_features['A_GLSZM'] = np.zeros(expected_dimensions['A_GLSZM'])
            
        # ---------- B. Morphological Features ---------- 
        if include_morphological:
            try:
                # Grayscale Morphology
                pdf_features, cdf_features = pyfeats.grayscale_morphology_features(roi_intensity, N=30)
                
                # Ensure correct dimensions
                if len(pdf_features) != expected_dimensions['B_Morphological_Grayscale_pdf']:
                    if verbose_warning:
                        print(f"Warning: Grayscale Morphology PDF features dimension mismatch. Expected {expected_dimensions['B_Morphological_Grayscale_pdf']}, got {len(pdf_features)}")
                    pdf_features = np.zeros(expected_dimensions['B_Morphological_Grayscale_pdf'])
                
                if len(cdf_features) != expected_dimensions['B_Morphological_Grayscale_cdf']:
                    if verbose_warning:
                        print(f"Warning: Grayscale Morphology CDF features dimension mismatch. Expected {expected_dimensions['B_Morphological_Grayscale_cdf']}, got {len(cdf_features)}")
                    cdf_features = np.zeros(expected_dimensions['B_Morphological_Grayscale_cdf'])
                
                roi_features['B_Morphological_Grayscale_pdf'] = pdf_features
                roi_features['B_Morphological_Grayscale_cdf'] = cdf_features
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing morphological grayscale features: {e}")
                roi_features['B_Morphological_Grayscale_pdf'] = np.zeros(expected_dimensions['B_Morphological_Grayscale_pdf'])
                roi_features['B_Morphological_Grayscale_cdf'] = np.zeros(expected_dimensions['B_Morphological_Grayscale_cdf'])

            # Désactiver temporairement les RuntimeWarnings ici
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                try:    
                    # Multi-level Binary Morphology
                    intensity_min = np.min(roi_intensity)
                    intensity_max = np.max(roi_intensity)
                    th_low = intensity_min + 0.25 * (intensity_max - intensity_min)
                    th_high = intensity_min + 0.5 * (intensity_max - intensity_min)

                    L_pdf, M_pdf, H_pdf, L_cdf, M_cdf, H_cdf = pyfeats.multilevel_binary_morphology_features(
                        roi_intensity, roi_mask, N=30, thresholds=[th_low, th_high])
                    
                    # Ensure correct dimensions for each output
                    binary_features = {
                        'B_Morphological_Binary_L_pdf': L_pdf,
                        'B_Morphological_Binary_M_pdf': M_pdf,
                        'B_Morphological_Binary_H_pdf': H_pdf,
                        'B_Morphological_Binary_L_cdf': L_cdf,
                        'B_Morphological_Binary_M_cdf': M_cdf,
                        'B_Morphological_Binary_H_cdf': H_cdf
                    }
                    
                    for key, features in binary_features.items():
                        if len(features) != expected_dimensions[key]:
                            if verbose_warning:
                                print(f"Warning: {key} features dimension mismatch. Expected {expected_dimensions[key]}, got {len(features)}")
                            features = np.zeros(expected_dimensions[key])
                        roi_features[key] = features
                        
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing morphological Binary features: {e}")
                    roi_features['B_Morphological_Binary_L_pdf'] = np.zeros(expected_dimensions['B_Morphological_Binary_L_pdf'])
                    roi_features['B_Morphological_Binary_M_pdf'] = np.zeros(expected_dimensions['B_Morphological_Binary_M_pdf'])
                    roi_features['B_Morphological_Binary_H_pdf'] = np.zeros(expected_dimensions['B_Morphological_Binary_H_pdf'])
                    roi_features['B_Morphological_Binary_L_cdf'] = np.zeros(expected_dimensions['B_Morphological_Binary_L_cdf'])
                    roi_features['B_Morphological_Binary_M_cdf'] = np.zeros(expected_dimensions['B_Morphological_Binary_M_cdf'])
                    roi_features['B_Morphological_Binary_H_cdf'] = np.zeros(expected_dimensions['B_Morphological_Binary_H_cdf'])
                    
        # ---------- C. Histogram-based Features ---------- 
        if include_histogram:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    hist_features = pyfeats.histogram(roi_intensity, roi_mask, bins=32)
                    hist_features = hist_features.astype(float)
                    
                    # Ensure correct dimension
                    if len(hist_features) != expected_dimensions['C_Histogram']:
                        if verbose_warning:
                            print(f"Warning: Histogram features dimension mismatch. Expected {expected_dimensions['C_Histogram']}, got {len(hist_features)}")
                        hist_features = np.zeros(expected_dimensions['C_Histogram'], dtype=float)
                    
                    roi_features['C_Histogram'] = hist_features
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing C histogram features: {e}")
                roi_features['C_Histogram'] = np.zeros(expected_dimensions['C_Histogram'], dtype=float)
                        
            try:    
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    multiregion_features = pyfeats.multiregion_histogram(
                        roi_intensity, roi_mask, bins=32, num_eros=3, square_size=3)
                    multiregion_features = multiregion_features.astype(float)
                    
                    # Ensure correct dimension
                    if len(multiregion_features) != expected_dimensions['C_MultiregionHistogram']:
                        if verbose_warning:
                            print(f"Warning: MultiregionHistogram features dimension mismatch. Expected {expected_dimensions['C_MultiregionHistogram']}, got {len(multiregion_features)}")
                        multiregion_features = np.zeros(expected_dimensions['C_MultiregionHistogram'], dtype=float)
                    
                    roi_features['C_MultiregionHistogram'] = multiregion_features
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing C MultiregionHistogram features: {e}")
                roi_features['C_MultiregionHistogram'] = np.zeros(expected_dimensions['C_MultiregionHistogram'], dtype=float)
                
            try:    
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    correlogram_features = pyfeats.correlogram(
                        roi_intensity, roi_mask, bins_digitize=32, bins_hist=32, flatten=True)
                    correlogram_features = correlogram_features.astype(float)
                    
                    # Ensure correct dimension
                    if len(correlogram_features) != expected_dimensions['C_Correlogram']:
                        if verbose_warning:
                            print(f"Warning: Correlogram features dimension mismatch. Expected {expected_dimensions['C_Correlogram']}, got {len(correlogram_features)}")
                        correlogram_features = np.zeros(expected_dimensions['C_Correlogram'], dtype=float)
                    
                    roi_features['C_Correlogram'] = correlogram_features
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing C Correlogram features: {e}")
                roi_features['C_Correlogram'] = np.zeros(expected_dimensions['C_Correlogram'], dtype=float)
        
        # ---------- D. Multi-scale Features ---------- 
        if include_multiscale:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    # Discrete Wavelet Transform
                    dwt_features = np.array(pyfeats.dwt_features(roi_intensity, roi_mask, wavelet='bior3.3', levels=3), dtype=float)
                    
                    # Ensure correct dimension
                    if len(dwt_features) != expected_dimensions['D_DWT']:
                        if verbose_warning:
                            print(f"Warning: DWT features dimension mismatch. Expected {expected_dimensions['D_DWT']}, got {len(dwt_features)}")
                        dwt_features = np.zeros(expected_dimensions['D_DWT'], dtype=float)
                    
                    roi_features['D_DWT'] = dwt_features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing multi-scale D_DWT features: {e}")
                    roi_features['D_DWT'] = np.zeros(expected_dimensions['D_DWT'], dtype=float)

                try:
                    # Stationary Wavelet Transform
                    swt_features = np.array(pyfeats.swt_features(roi_intensity, roi_mask, wavelet='bior3.3', levels=3), dtype=float)
                    
                    # Ensure correct dimension
                    if len(swt_features) != expected_dimensions['D_SWT']:
                        if verbose_warning:
                            print(f"Warning: SWT features dimension mismatch. Expected {expected_dimensions['D_SWT']}, got {len(swt_features)}")
                        swt_features = np.zeros(expected_dimensions['D_SWT'], dtype=float)
                    
                    roi_features['D_SWT'] = swt_features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing multi-scale D_SWT features: {e}")
                    roi_features['D_SWT'] = np.zeros(expected_dimensions['D_SWT'], dtype=float)

                try:
                    # Wavelet Packet
                    wp_features = np.array(pyfeats.wp_features(roi_intensity, roi_mask, wavelet='coif1', maxlevel=3), dtype=float)
                    
                    # Ensure correct dimension
                    if len(wp_features) != expected_dimensions['D_WP']:
                        if verbose_warning:
                            print(f"Warning: WP features dimension mismatch. Expected {expected_dimensions['D_WP']}, got {len(wp_features)}")
                        wp_features = np.zeros(expected_dimensions['D_WP'], dtype=float)
                    
                    roi_features['D_WP'] = wp_features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing multi-scale D_WP features: {e}")
                    roi_features['D_WP'] = np.zeros(expected_dimensions['D_WP'], dtype=float)
                
                try:
                    # Gabor Transform
                    gt_features = np.array(pyfeats.gt_features(roi_intensity, roi_mask), dtype=float)
                    
                    # Ensure correct dimension
                    if len(gt_features) != expected_dimensions['D_GT']:
                        if verbose_warning:
                            print(f"Warning: GT features dimension mismatch. Expected {expected_dimensions['D_GT']}, got {len(gt_features)}")
                        gt_features = np.zeros(expected_dimensions['D_GT'], dtype=float)
                    roi_features['D_GT'] = gt_features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing multi-scale D_GT features: {e}")
                    roi_features['D_GT'] = np.zeros(expected_dimensions['D_GT'], dtype=float)

                try:
                    # Amplitude-Modulation Frequency-Modulation
                    amfm_features = np.array(pyfeats.amfm_features(roi_intensity), dtype=float)
                    
                    # Ensure correct dimension
                    if len(amfm_features) != expected_dimensions['D_AMFM']:
                        if verbose_warning:
                            print(f"Warning: AMFM features dimension mismatch. Expected {expected_dimensions['D_AMFM']}, got {len(amfm_features)}")
                        amfm_features = np.zeros(expected_dimensions['D_AMFM'], dtype=float)
                    
                    roi_features['D_AMFM'] = amfm_features
                except Exception as e:
                    if verbose_warning:
                        print(f"Warning: Error computing multi-scale D_AMFM features: {e}")
                    roi_features['D_AMFM'] = np.zeros(expected_dimensions['D_AMFM'], dtype=float)

        # ---------- E. Other Features ----------
        if include_other:
            try:
                # HOG Features
                hog_features = pyfeats.hog_features(roi_intensity, ppc=8, cpb=3)
                # Ensure the result is a numeric array
                hog_features = np.array(hog_features, dtype=np.float64)
                
                # Ensure correct dimension
                if len(hog_features) != expected_dimensions['E_HOG']:
                    if verbose_warning:
                        print(f"Warning: HOG features dimension mismatch. Expected {expected_dimensions['E_HOG']}, got {len(hog_features)}")
                    hog_features = np.zeros(expected_dimensions['E_HOG'], dtype=np.float64)
                
                roi_features['E_HOG'] = hog_features
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing HOG features: {e}")
                roi_features['E_HOG'] = np.zeros(expected_dimensions['E_HOG'], dtype=np.float64)

            try:
                # Hu Moments
                hu_moments = pyfeats.hu_moments(roi_intensity)
                # Ensure the result is a numeric array
                hu_moments = np.array(hu_moments, dtype=np.float64)
                
                # Ensure correct dimension
                if len(hu_moments) != expected_dimensions['E_HuMoments']:
                    if verbose_warning:
                        print(f"Warning: HuMoments features dimension mismatch. Expected {expected_dimensions['E_HuMoments']}, got {len(hu_moments)}")
                    hu_moments = np.zeros(expected_dimensions['E_HuMoments'], dtype=np.float64)
                
                roi_features['E_HuMoments'] = hu_moments
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing HU features: {e}")
                roi_features['E_HuMoments'] = np.zeros(expected_dimensions['E_HuMoments'], dtype=np.float64)

            try:
                # Threshold Adjacency Statistics
                tas_features = pyfeats.tas_features(roi_intensity)
                # Ensure the result is a numeric array
                tas_features = np.array(tas_features, dtype=np.float64)
                
                # Ensure correct dimension
                if len(tas_features) != expected_dimensions['E_TAS']:
                    if verbose_warning:
                        print(f"Warning: TAS features dimension mismatch. Expected {expected_dimensions['E_TAS']}, got {len(tas_features)}")
                    tas_features = np.zeros(expected_dimensions['E_TAS'], dtype=np.float64)
                
                roi_features['E_TAS'] = tas_features
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing TAS features: {e}")
                roi_features['E_TAS'] = np.zeros(expected_dimensions['E_TAS'], dtype=np.float64)

            try:
                # Zernike Moments
                # Determine appropriate radius based on ROI size
                min_dim = min(roi_intensity.shape)
                radius = min(9, min_dim // 2)  # Use smaller of 9 or half the minimum dimension
                zernike_moments = pyfeats.zernikes_moments(roi_intensity, radius=radius)
                # Ensure the result is a numeric array
                zernike_moments = np.array(zernike_moments, dtype=np.float64)
                
                # Ensure correct dimension
                if len(zernike_moments) != expected_dimensions['E_ZernikesMoments']:
                    if verbose_warning:
                        print(f"Warning: ZernikesMoments features dimension mismatch. Expected {expected_dimensions['E_ZernikesMoments']}, got {len(zernike_moments)}")
                    zernike_moments = np.zeros(expected_dimensions['E_ZernikesMoments'], dtype=np.float64)
                
                roi_features['E_ZernikesMoments'] = zernike_moments
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing Zernike features: {e}")
                roi_features['E_ZernikesMoments'] = np.zeros(expected_dimensions['E_ZernikesMoments'], dtype=np.float64)
        
        # Add this component's features to the list
        component_features.append(roi_features)
            
    # ---------- Aggregate component features ----------
    # For each feature type, compute mean, std, min, max across all components
    if component_features:
        # Verify that all components have the same keys
        if len(component_features) > 1:
            first_keys = set(component_features[0].keys())
            for idx, comp in enumerate(component_features[1:], start=1):
                if set(comp.keys()) != first_keys:
                    if verbose_warning:
                        print(f"Warning: Component {idx} has different keys than the first component")
                    # Ensure all components have the same keys
                    for key in first_keys:
                        if key not in comp:
                            comp[key] = np.zeros(expected_dimensions.get(key, 1))
        
        # List all feature keys from the first component
        feature_keys = component_features[0].keys()
        
        for key in feature_keys:
            try:
                # First check if arrays have consistent shapes
                shapes = [comp[key].shape for comp in component_features]
                consistent = all(s == shapes[0] for s in shapes)
                
                if not consistent:
                    if verbose_warning:
                        print(f"Warning: Inconsistent shapes for {key}: {shapes}")
                    # Fix inconsistent shapes by using zeros with expected dimension
                    for i, comp in enumerate(component_features):
                        if key in expected_dimensions and comp[key].shape != (expected_dimensions[key],):
                            comp[key] = np.zeros(expected_dimensions[key])
                
                # Stack this feature from all components (with shape check)
                feature_arrays = []
                for comp in component_features:
                    if key in comp and comp[key] is not None and len(comp[key]) > 0:
                        # Make sure the array is proper shape (1D)
                        if comp[key].ndim > 1:
                            comp[key] = comp[key].flatten()
                        feature_arrays.append(comp[key])
                
                if feature_arrays:
                    stacked_feature = np.vstack(feature_arrays)
                    
                    if stacked_feature.size > 0:
                        # Compute statistics
                        mean_feature = np.mean(stacked_feature, axis=0)
                        std_feature = np.std(stacked_feature, axis=0)
                        min_feature = np.min(stacked_feature, axis=0)
                        max_feature = np.max(stacked_feature, axis=0)
                        
                        
                        # Append to feature vector and track names
                        for i in range(len(mean_feature)):
                            feat_vector.append(mean_feature[i])
                            feature_names.append(f"{key}_mean_{i}")
                        
                        for i in range(len(std_feature)):
                            feat_vector.append(std_feature[i])
                            feature_names.append(f"{key}_std_{i}")
                        
                        # Only include min/max for key features to control vector length
                        if key.startswith(('A_FOS', 'A_GLCM', 'A_NGTDM', 'E_HuMoments')):
                            for i in range(len(min_feature)):
                                feat_vector.append(min_feature[i])
                                feature_names.append(f"{key}_min_{i}")
                            
                            for i in range(len(max_feature)):
                                feat_vector.append(max_feature[i])
                                feature_names.append(f"{key}_max_{i}")
          
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error processing feature {key}: {e}")
                # Add zeros as placeholders for this feature
                dim = expected_dimensions.get(key, 1)
                for i in range(dim):
                    feat_vector.append(0.0)
                    feature_names.append(f"{key}_mean_{i}")
                for i in range(dim):
                    feat_vector.append(0.0)
                    feature_names.append(f"{key}_std_{i}")
                if key.startswith(('A_FOS', 'A_GLCM', 'A_NGTDM', 'E_HuMoments')):
                    for i in range(dim):
                        feat_vector.append(0.0)
                        feature_names.append(f"{key}_min_{i}")
                    for i in range(dim):
                        feat_vector.append(0.0)
                        feature_names.append(f"{key}_max_{i}")
                
    
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
            
            feat_vector.append(mean_nn_dist)
            feature_names.append("mean_nearest_neighbor_dist")
            feat_vector.append(std_nn_dist)
            feature_names.append("std_nearest_neighbor_dist")
            feat_vector.append(min_nn_dist)
            feature_names.append("min_nearest_neighbor_dist")
            feat_vector.append(max_nn_dist)
            feature_names.append("max_nearest_neighbor_dist")
            
            
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
                
                for i, k_r in enumerate(ripley_k_values):
                    feat_vector.append(k_r)
                    feature_names.append(f"ripley_k_{i}")
            except Exception as e:
                if verbose_warning:
                    print(f"Warning: Error computing Ripley's K: {e}")
                for i in range(5):
                    feat_vector.append(0.0)
                    feature_names.append(f"ripley_k_{i}")

        else:
            # Add zeros for spatial features if too few components
            for name in ["mean_nearest_neighbor_dist", "std_nearest_neighbor_dist", 
                        "min_nearest_neighbor_dist", "max_nearest_neighbor_dist"]:
                feat_vector.append(0.0)
                feature_names.append(name)
            for i in range(5):
                feat_vector.append(0.0)
                feature_names.append(f"ripley_k_{i}")
    except Exception as e:
        if verbose_warning:
            print(f"Warning: Error computing spatial distribution features: {e}")
        for name in ["mean_nearest_neighbor_dist", "std_nearest_neighbor_dist", 
                    "min_nearest_neighbor_dist", "max_nearest_neighbor_dist"]:
            feat_vector.append(0.0)
            feature_names.append(name)
        for i in range(5):
            feat_vector.append(0.0)
            feature_names.append(f"ripley_k_{i}")
    
    # Two neighbors distance histogram
    try:
        two_neighbors_distance_histogram = two_neighbors_distance_histogram(G, N_BINS_FEAT)
        for i, val in enumerate(two_neighbors_distance_histogram):
            feat_vector.append(val)
            feature_names.append(f"two_neighbors_dist_hist_{i}")
    except Exception as e:
        if verbose_warning:
            print(f"Warning: Error computing two-neighbors distance histogram: {e}")
        for i in range(N_BINS_FEAT):
            feat_vector.append(0.0)
            feature_names.append(f"two_neighbors_dist_hist_{i}")
        
    # compute the mean, max, median, min distance of edges in G
    try:
        # Compute the length of all edges in G
        edge_lengths = []
        for u, v in G.edges():
            length = np.linalg.norm(np.array(G.nodes[u]['centroid']) - np.array(G.nodes[v]['centroid']))
            edge_lengths.append(length)
        edge_lengths = np.array(edge_lengths)
        # Compute statistics
        mean_distance = np.mean(edge_lengths)
        max_distance = np.max(edge_lengths)
        median_distance = np.median(edge_lengths)
        min_distance = np.min(edge_lengths)
        # Add to feature vector
        feat_vector.append(mean_distance)
        feature_names.append("mean_edge_length")
        feat_vector.append(max_distance)
        feature_names.append("max_edge_length")
        feat_vector.append(median_distance)
        feature_names.append("median_edge_length")
        feat_vector.append(min_distance)
        feature_names.append("min_edge_length")
    except Exception as e:
        if verbose_warning:
            print(f"Warning: Error computing edge length statistics: {e}")
        for name in ["mean_edge_length", "max_edge_length", "median_edge_length", "min_edge_length"]:
            feat_vector.append(0.0)
            feature_names.append(name)
    
    # Measure diff features
    try:
        feat_vector.append(Measure_diff_slice)
        feature_names.append("measure_diff_slice")
        feat_vector.append(Measure_diff_points_segment)
        feature_names.append("measure_diff_points_segment")
    except Exception as e:
        if verbose_warning:
            print(f"Warning: Error adding Measure_diff_slice and Measure_diff_points_segment: {e}")
        feat_vector.append(0.0)
        feature_names.append("measure_diff_slice")
        feat_vector.append(0.0)
        feature_names.append("measure_diff_points_segment")
    
    # Convert feat_vector to numpy array and check for NaNs or infinities
    feat_vector = np.array(feat_vector, dtype=np.float64)
    feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Return both the feature vector and feature names
    return feat_vector, feature_names

def get_regions_of_interest(coord, image_original, binary_mask):
    
    # Step 1: Initial rough segmentation
    image_markers = np.zeros_like(image_original, dtype=np.int32)
    for i, (x, y) in enumerate(coord, 1):  # Start at 1 (0 is background)
        image_markers[int(x), int(y)] = i + 100  # Offset markers to avoid overlap
    
    rough_segmented = ski.segmentation.watershed(-image_original, connectivity=1, markers=image_markers, mask=binary_mask)
    refined_segmented = np.zeros_like(rough_segmented)
    
    real_image = image_original.copy()
    
    # blurring the image
    image_original = ski.filters.gaussian(image_original, sigma=0.5)
 
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
        foreground_mask = (refined_region == foreground_label)

        # show foreground mask
        """plt.figure(figsize=(8, 6))
        plt.imshow(foreground_mask, cmap='gray')
        plt.title(f"Foreground mask for region {region.label}")
        plt.axis('off')
        plt.show()"""

        # Keep only the largest connected component in the foreground        
        labeled_fg = ski.measure.label(foreground_mask, connectivity=1)
        regions = ski.measure.regionprops(labeled_fg)
        if regions:
            largest_region = max(regions, key=lambda r: r.area)
            largest_component_mask = (labeled_fg == largest_region.label)
            filled_component_mask = scipy.ndimage.binary_fill_holes(largest_component_mask) 
        else:
            filled_component_mask = np.zeros_like(foreground_mask)

        # Update refined mask
        refined_mask = np.zeros_like(mask, dtype=np.int32)
        refined_mask[filled_component_mask] = region.label
        

        # Place refined region in final segmented image
        refined_segmented[minr:maxr, minc:maxc][refined_mask > 0] = region.label
        
        
        # Plot the original image, filled_component_mask
        """fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # 1 row, 3 columns
        # Visualize original image (first subplot)
        axes[0].imshow(real_image[minr:maxr, minc:maxc], cmap='gray')
        axes[0].set_title(f"Original Image with Region {region.label}")
        axes[0].axis('off') # Turn off axis labels and ticks
        # Visualize original image (first subplot) only in the region of interest
        axes[1].imshow(image_original[minr:maxr, minc:maxc], cmap='gray')
        axes[1].set_title(f"Original Image with Region {region.label}")
        axes[1].axis('off') # Turn off axis labels and ticks
        # Visualize foreground mask (second subplot)
        axes[2].imshow(filled_component_mask, cmap='gray')
        axes[2].set_title(f"Filled Component Mask for Region {region.label}")
        axes[2].axis('off')
        # Adjust layout to prevent overlapping titles
        plt.tight_layout()
        # Show the combined image
        plt.show()"""
        
        
        
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

def get_feature_vector(G, median_width, Measure_diff_slice, Measure_diff_points_segment, X, y, X_orig, max_images, mask_images, intensity, recompute=False, pkl_name=DEFAULT_PKL_NAME, n_features=N_FEAT, n_bins=N_BINS_FEAT):
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
            'label_components': [],
            'features name': []
        }
        
        for im_num, image in enumerate(X):
            print(f'Extracting features for image {im_num+1}/{len(X)}')
            image_original = X_orig[im_num]
            maxima = max_images[im_num]
            mask = mask_images[im_num]
            
            # get mean intensity of background
            mean_intensity = np.mean(image[mask == 0])
        
            component, label_seg = get_regions_of_interest(maxima, image_original, mask)
            feat, features_name = create_feature_vector(G[im_num], mean_intensity, median_width[im_num], Measure_diff_slice[im_num], Measure_diff_points_segment[im_num], image, component, intensity[im_num], n_bins, 
                                         include_texture=True, include_morphological=True,
                                         include_histogram=False, include_multiscale=False,
                                         include_other=False)

            features['label'].append(y[im_num])
            features['data'].append(feat)
            features['filename'].append(f"image_{im_num}")
            features['components'].append(component)
            features['label_components'].append(label_seg)
            features['features name'].append(features_name)
            
            X_feat.append(feat)
        
        # compute number max of features in a row of X_feat
        max_len = max(len(row) for row in X_feat)
        
        for i, row in enumerate(X_feat):
            if len(row) == 1:
                # add zeros to the row until it has max_len elements
                X_feat[i] = np.zeros(max_len)
        
        X_feat = np.array(X_feat)
        
        # print number of features
        print(f'Number of features: {X_feat.shape[1]}')
        # print number of images
        print(f'Number of images: {X_feat.shape[0]}')
        
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
            
        #print(X_df)
        
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
        
        #print(f"Selected features (mRMR): {X_df.columns}")
   
      
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

