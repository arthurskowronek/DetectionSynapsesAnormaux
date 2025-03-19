import numpy as np
import skimage as ski
import datetime
import joblib
import shutil
from pathlib import Path
from skimage.feature import hog
import pyfeats
import pointpats
from boruta import BorutaPy  
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV


# Constants
N_FEAT = 12
N_BINS_FEAT = 20
DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_PKL_NAME = f'dataset_{DATE}.pkl'
DATASET_PKL_DIR = Path('./dataset_pkl')


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

def compute_zernike_features(roi_mask, degree=5, radius=10):
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
    return pyfeats.zernikes_moments(roi_mask.astype(np.float32), radius=radius, degree=degree)

def compute_hog_features(roi_intensity, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
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
    # Resize ROI to a fixed size for uniformity (e.g., 64x64)
    roi_resized = ski.transform.resize(roi_intensity, (64, 64), anti_aliasing=True)
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

def create_feature_vector(image, component_props, intensity, n_bins=N_BINS_FEAT):
    """
    Create an extended feature vector from a preprocessed image.
    
    The feature vector includes:
      - Histogram features of geometric and intensity properties (original features)
      - Mean intensity derivative along the skeleton for each ROI
      - Zernike moments (aggregated as mean over ROIs)
      - HOG features (aggregated as mean over ROIs)
      - Spatial distribution feature using Ripley’s K-function
    
    Parameters
    ----------
    image : ndarray
        Preprocessed intensity image.
    component_props : list
        List of region properties (from skimage.measure.regionprops).
    n_bins : int
        Number of bins for histograms.
        
    Returns
    -------
    ndarray
        Extended feature vector.
    """
    if not component_props:
        print("Warning: No components found in image.")
        # Return a vector of zeros; length depends on the features added
        # Here we assume: original histograms (N_FEAT*(n_bins-1) + 1) plus
        # one value for intensity derivative, a few for Zernike (say 6 moments),
        # HOG features (depends on hog descriptor length, here assumed 1764 for 64x64 images with default parameters)
        return np.zeros(N_FEAT * (n_bins - 1) + 1 + 1 + 6 + 1764 + 1)
    
    # Number of detected synapses
    num_synapses = len(component_props)
    
    # Original feature extraction: geometric and intensity properties
    feature_dict = {
        "major_axis_length": [prop.axis_major_length for prop in component_props],
        "axis_ratio": [prop.axis_minor_length / prop.axis_major_length if prop.axis_major_length != 0 else 0 for prop in component_props],
        "extent": [prop.extent for prop in component_props],
        "area": [prop.area for prop in component_props],
        "perimeter": [prop.perimeter for prop in component_props],
        "convex_area": [prop.convex_area for prop in component_props],
        "eccentricity": [prop.eccentricity for prop in component_props],
        "solidity": [prop.solidity for prop in component_props],
        "mean_intensity": [prop.mean_intensity for prop in component_props],
        "max_intensity": [prop.max_intensity for prop in component_props],
        "min_intensity": [prop.min_intensity for prop in component_props]
    }
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
    
    feat_vector = [num_synapses]
    for feature, values in feature_dict.items():
        bin_range = bin_ranges[feature]
        hist, bins = np.histogram(values, bins=np.linspace(*bin_range, num=n_bins), density=True)
        # Multiply by bin width to get an approximation of the density integrated over each bin
        feat_vector.extend(hist * (bins[1] - bins[0]))
    
    # Compute nearest neighbor histogram (spatial proximity of centroids)
    bins_nn = np.linspace(3, 30, num=n_bins)
    centroid_positions = np.array([prop.centroid for prop in component_props])
    hist_nn = first_neighbor_distance_histogram(centroid_positions, bins_nn)
    feat_vector.extend(hist_nn * (bins_nn[1] - bins_nn[0]))
    
    # For each component, extract ROI features: intensity derivative, Zernike moments, HOG features.
    intensity_derivs = []
    zernike_list = []
    hog_list = []
    for prop in component_props:
        # Extract ROI using bounding box from regionprops
        minr, minc, maxr, maxc = prop.bbox
        # Make sure we have a non-zero region
        if maxr - minr < 5 or maxc - minc < 5:
            continue
        roi_mask = prop.image  # binary mask of the region
        intensity_roi = image[minr:maxr, minc:maxc]
        
        # Intensity derivative along skeleton
        intensity_deriv = np.gradient(intensity)
        intensity_derivs.append(intensity_deriv)
        
        # Zernike moments (assumes ROI is roughly centered; you might need to pad/center)
        try:
            zernike_vals = compute_zernike_features(roi_mask, degree=5, radius=10)
            zernike_list.append(zernike_vals)
        except Exception as e:
            # In case of errors (e.g., ROI too small), append zeros
            zernike_list.append(np.zeros(6))
        
        # HOG features
        hog_desc = compute_hog_features(intensity_roi)
        hog_list.append(hog_desc)
    
    # Aggregate ROI features: use mean values over regions.
    mean_deriv = np.mean(intensity_derivs) if intensity_derivs else 0.0
    feat_vector.append(mean_deriv)
    
    # For Zernike, assume each call returns a fixed-length vector (here assumed length 6)
    if zernike_list:
        mean_zernike = np.mean(np.vstack(zernike_list), axis=0)
    else:
        mean_zernike = np.zeros(6)
    feat_vector.extend(mean_zernike)
    
    # For HOG, assume fixed-length descriptor for each ROI; take the mean descriptor.
    if hog_list:
        mean_hog = np.mean(np.vstack(hog_list), axis=0)
    else:
        # Determine expected length (here using 64x64 ROI, default hog returns length ~1764)
        mean_hog = np.zeros(1764)
    feat_vector.extend(mean_hog)
    
    # Spatial distribution: compute Ripley’s K at a given support.
    ripley_K = compute_ripley_k(centroid_positions, support=50)
    feat_vector.append(ripley_K)
    
    return np.array(feat_vector)

def get_regions_of_interest(coord, image_original, binary_mask):
    """
    Segment the image via watershed given initial coordinates.
    
    Parameters
    ----------
    coord : list or array
        List of (x, y) coordinates for markers.
    image_original : ndarray
        Original intensity image.
    binary_mask : ndarray
        Binary mask of the foreground.
        
    Returns
    -------
    list
        Region properties of segmented regions.
    ndarray
        Labeled segmentation mask.
    """
    image_markers = np.zeros_like(image_original, dtype=np.int32)
    for i, (x, y) in enumerate(coord, 1):  # Start at 1 (0 is background)
        image_markers[int(x), int(y)] = i + 100  # offset markers to avoid overlap
    
    # Uncomment dilation if needed:
    # image_markers = morphology.dilation(image_markers, morphology.disk(2))
    
    segmented = ski.segmentation.watershed(-image_original, connectivity=1, markers=image_markers, mask=binary_mask)
    region_props = ski.measure.regionprops(segmented, intensity_image=image_original)
    
    # Visualize segmentation
    """colored_labels = label2rgb(segmented, image=image_original, bg_label=0)
    plt.figure(figsize=(8, 6))
    plt.imshow(colored_labels)
    plt.title("Refined Watershed Segmentation")
    plt.show()"""
    
    return region_props, segmented

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
            feat = create_feature_vector(image, component, intensity[im_num], n_bins)
            
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

def select_features(X, y, method='lasso'):
    """
    Reduce feature dimensions using feature selection.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray
        Labels.
    method : str
        'boruta' to use Boruta or 'lasso' to use LASSO.
        
    Returns
    -------
    ndarray
        Reduced feature matrix.
    """
    if method == 'boruta':
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
        boruta_selector.fit(X, y)
        return boruta_selector.transform(X)
    elif method == 'lasso':
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)
        # Select features with non-zero coefficients
        mask = lasso.coef_ != 0
        print(f"Selected {np.sum(mask)} features out of {X.shape[1]}")
        return X[:, mask]
    else:
        raise ValueError("Method must be 'boruta' or 'lasso'.")

