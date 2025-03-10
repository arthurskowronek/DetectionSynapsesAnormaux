import os
import shutil
import joblib
import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.transform import resize

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

N_RUNS=100
MAX_BINS=255
LEARN_RATE=0.1
MAX_ITER=1000
IN_PARAM=np.array([MAX_BINS,LEARN_RATE,MAX_ITER],dtype='float')


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
    if not reimport_images:
        data = joblib.load(DATASET_PKL_DIR / pkl_name)
        print('Data loaded')
        return data
    
    print("Reimporting images...")
    
    # Ensure directories exist and are empty
    for directory in [MUTANT_DIR, WT_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True)
        else:
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
        subdir_name = Path(subdirectory).name
        
        if subdir_name.startswith('Mut'):
            target_dir = MUTANT_DIR
            prefix = "Mut"
            counter = count_mutant
            count_mutant = process_directory(subdirectory, target_dir, prefix, counter)
            
        elif subdir_name.startswith('WildType'):
            target_dir = WT_DIR
            prefix = "WT"
            counter = count_wildtype
            count_wildtype = process_directory(subdirectory, target_dir, prefix, counter)
    
    print(f"Images imported. Mutant files: {count_mutant}, WildType files: {count_wildtype}")
    
    # Load images into data dictionary
    for label, directory in [("Mutant", MUTANT_DIR), ("WildType", WT_DIR)]:
        for file in directory.glob('*.tif'):
            im = imread(file)
            im = process_image(im)
            
            data["label"].append(label)
            data["filename"].append(file.name)
            data["data"].append(im)
    
    # Save dataset
    DATASET_PKL_DIR.mkdir(exist_ok=True)
    joblib.dump(data, pkl_name)
    shutil.move(pkl_name, DATASET_PKL_DIR)
    print(f"Dataset saved as {pkl_name}")
   
    return joblib.load(DATASET_PKL_DIR / pkl_name)


def process_directory(source_dir, target_dir, prefix, counter):
    """Process files in a directory by copying and renaming them."""
    for file in os.listdir(source_dir):
        shutil.copy(os.path.join(source_dir, file), target_dir)
        new_name = f"{prefix}{counter}.tif"
        os.rename(target_dir / file, target_dir / new_name)
        counter += 1
    return counter


def process_image(image):
    """Process an image to ensure consistent format and size."""
    if len(image.shape) > 2:  # Handle multi-channel images
        image = image[1, :, :]
    
    if image.shape != IMAGE_SIZE:
        image = resize(image, IMAGE_SIZE, preserve_range=True)
        image = image.astype(np.uint8)  # Ensure consistent dtype
    
    return image


def show_dataset_properties(data):
    """Display properties of the dataset."""
    print('Number of samples:', len(data['data']))
    print('Keys:', list(data.keys()))
    print('Description:', data['description'])
    print('Image shape:', data['data'][0].shape)
    print('Labels:', np.unique(data['label']))


def display_image(image, number=None, image_type=''):
    """
    Display an image with matplotlib.
    
    Parameters
    ----------
    image : array
        Image to display
    number : int or str, optional
        Identifier for the image
    image_type : str, optional
        Type of the image (e.g., 'original', 'Frangi')
    """
    _, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    
    title = "Image"
    if number is not None:
        title += f" {number}"
    if image_type:
        title += f" {image_type}"
    
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()


def preprocess_images(recompute=False, X=None, pkl_name=DEFAULT_PKL_NAME):
    """
    Apply preprocessing (Frangi filter) to images.
    
    Parameters
    ----------
    recompute : bool
        If True, recompute preprocessing. If False, load from file.
    X : array
        Array of images to preprocess
    pkl_name : str
        Base name for the preprocessing pkl file
        
    Returns
    -------
    X_preprocessed : array
        Preprocessed images
    """
    print('Preprocessing images...')
    
    preprocess_file = f'{pkl_name}_preprocessing.pkl'
    
    if not recompute:
        try:
            X_preprocessed = joblib.load(DATASET_PKL_DIR / preprocess_file)
            print('Preprocessing loaded from file.')
            return X_preprocessed
        except FileNotFoundError:
            print('Preprocessing file not found. Recomputing...')
            recompute = True
    
    if X is None:
        raise ValueError("Input images (X) must be provided when recomputing preprocessing")
    
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
    print('Preprocessing done and saved.')
    
    return X_preprocessed


def compute_pdnn(positions, bins):
    """compute the histogram of the distance to first neighbor of the centroids"""
    min_dist = np.zeros(len(positions))
    # take each blob positions in turn
    for indx in range(len(positions)):
        curr_pos = positions[indx, :]
        # square of the differences of the positions with the other centroids
        sq_diff = np.square(np.array(positions) - curr_pos)
        # distances with other centroids
        dist = np.sqrt(np.sum(sq_diff, axis=1))
        # remove the zero distance of curr_pos with itself
        dist = dist[dist > 0]
        # keep the smallest distance
        min_dist[indx] = dist.min()

    histo, _ = np.histogram(min_dist, bins)
    return histo / np.sum(histo)

def create_feature_vector(X, y, recompute= False, pkl_name=DEFAULT_PKL_NAME, n_features=N_FEAT, n_bins=N_BINS_FEAT):
    
    X_feat = np.zeros((len(X), n_features * (n_bins - 1)))
    features_file_name = f'{pkl_name}_features.pkl'

    if recompute:
        print('Computing features...')
        features = dict()
        features['description'] = f'C elegans images features from frangi blobs'
        features['label']=[]
        features['data']=[]
        features['filename'] = []

        for im_num,_ in enumerate(X):
            threshold = threshold_otsu(X[im_num])
            binary_image = X[im_num] > threshold
            labeled_components = label(binary_image)
            component_props = regionprops(labeled_components,intensity_image=X[im_num])


            sel_component_props=[x for x in component_props if x.area>MIN_AREA_COMPO]
            axis_M_ls=[x.axis_major_length for x in sel_component_props]
            ratio_axis=[x.axis_minor_length/x.axis_major_length for x in sel_component_props]
            centroids=[x.centroid for x in sel_component_props]
            extents=[x.extent for x in sel_component_props]
            # intensities=[x.image_intensity.sum() for x in sel_component_props]
            feat=[]

            feat1,bins =np.histogram(axis_M_ls,bins=np.linspace(start=3,stop=20,num=N_BINS_FEAT),density=True)
            feat=np.append(feat,feat1*(bins[1]-bins[0]))

            feat1,bins =np.histogram(ratio_axis,bins=np.linspace(start=0,stop=1,num=N_BINS_FEAT),density=True)
            feat=np.append(feat,feat1*(bins[1]-bins[0]))

            feat1,bins =np.histogram(extents,bins=np.linspace(start=0,stop=1,num=N_BINS_FEAT),density=True)
            feat=np.append(feat,feat1*(bins[1]-bins[0]))

            bins=np.linspace(start=3,stop=30,num=N_BINS_FEAT)
            feat1=compute_pdnn(np.array(centroids),bins)
            feat=np.append(feat,feat1*(bins[1]-bins[0]))

            # feat1,bins =np.histogram(intensities,bins=np.linspace(start=7e3,stop=2e4,num=N_BINS_FEAT),density=True)
            # feat=np.append(feat,feat1*(bins[1]-bins[0]))

            features['label'].append(y[im_num])
            features['data'].append(feat)
            features['filename'].append(data['filename'][im_num])
            X_feat[im_num,:]=np.copy(feat)

        joblib.dump(features, features_file_name)
        shutil.move(features_file_name, DATASET_PKL_DIR)
        print('Features computed and saved.')

    else:
        print('Loading features...')
        features=joblib.load(DATASET_PKL_DIR / features_file_name)
        for im_num, feat in enumerate(features['data']):
            X_feat[im_num,:]=feat
        y=features['label']
        print('Features loaded.')

    return X_feat

def Training():
    
    return None

if __name__ == "__main__":
    # Load dataset
    filename_pkl_dataset = 'dataset_2025-03-10_08-05-48'
    data = create_dataset(reimport_images=False, pkl_name=filename_pkl_dataset + '.pkl')
    
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    # Preprocessing
    X_preprocessed = preprocess_images(recompute=False, X=X, pkl_name=filename_pkl_dataset)
    
    # Display sample images
    display_image(X[50], 50, 'original')
    display_image(X_preprocessed[50], 50, 'Frangi')
    
    # Compute features
    X_features = create_feature_vector(X_preprocessed, y, recompute=True)
    
    ##### Training
    print('Learning....')
    mean_corr_estim = Training(X_features, y, rs, N_RUNS,IN_PARAM)
    print(f'Mean % correct= {100*mean_corr_estim:.1f}')
