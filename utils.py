import os
import shutil
import joblib
import datetime
from skimage.io import imread
from skimage.transform import resize
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

DATE = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pklname_dataset = f'dataset_{DATE}.pkl'

N_FEAT = 4
N_BINS_FEAT = 20

def createDataset(REIMPORT_IMAGES, pklname = pklname_dataset):   
    '''
    Create a dataset from images in directory "data" and save it as a pkl file.
    If REIMPORT_IMAGES is True, the images in the directory "data" are reimported.
    If REIMPORT_IMAGES is False, the pkl file is loaded.
    
    Parameters
    ----------
    REIMPORT_IMAGES : bool
        If True, reimport images from directory "data" and save them as a pkl file.
        If False, load the pkl file.    
    pklname : str
        Name of the pkl file to save the dataset.
        
    Returns
    ------- 
    data : dict
        Dictionary containing the dataset.
    ''' 
    if REIMPORT_IMAGES:
        print("Reimporting images...")
        # create directory "_Mutant" in directory "data" if it does not exist, else erase all files
        if not os.path.exists(r'.\data\_Mutant'):
            os.makedirs(r'.\data\_Mutant')
        else: # erase all files in directory "_Mutant"
            for file in os.listdir(r'.\data\_Mutant'):
                os.remove(os.path.join(r'.\data\_Mutant', file))
        # create directory "_WT" in directory "data" if it does not exist, else erase all files
        if not os.path.exists(r'.\data\_WT'):
            os.makedirs(r'.\data\_WT')
        else: # erase all files in directory "_WT"
            for file in os.listdir(r'.\data\_WT'):
                os.remove(os.path.join(r'.\data\_WT', file))
            
        # get all subdirectories of directory "data"
        subdirectories = [x[0] for x in os.walk(r'.\data')][1:]

        # copie files in all subdirectories to directory "_Mutant" and "_WT"
        count_mutant = 0
        count_wildtype = 0
        for subdirectory in subdirectories:
            if subdirectory.split('\\')[-1].startswith('Mut'):
                for file in os.listdir(subdirectory):
                    shutil.copy(os.path.join(subdirectory, file), r'.\data\_Mutant')
                    #rename file : "Mut" + str(count_mutant)
                    os.rename(os.path.join(r'.\data\_Mutant', file), os.path.join(r'.\data\_Mutant', f"Mut{count_mutant}.tif"))
                    count_mutant += 1
            if subdirectory.split('\\')[-1].startswith('WildType'):
                for file in os.listdir(subdirectory):
                    shutil.copy(os.path.join(subdirectory, file), r'.\data\_WT')
                    #rename file : "WT" + str(count_wildtype)
                    os.rename(os.path.join(r'.\data\_WT', file), os.path.join(r'.\data\_WT', f"WT{count_wildtype}.tif"))
                    count_wildtype += 1
        print("Images imported.")
        print(f"Number of mutant files : {count_mutant}")
        print(f"Number of wildtype files : {count_wildtype}")
        
        # create pkl file with all images in directory "_Mutant" and "_WT"
        data = dict()
        data["description"] = f"original (1024x1024) C elegans images in grayscale"
        data["label"] = []
        data["filename"] = []
        data["data"] = []
        for file in os.listdir(r'.\data\_Mutant'):
            im = imread(os.path.join(r'.\data\_Mutant', file))
            if len(im.shape) > 2: #resize
                im = im[1, :, :]
            if im.shape != (1024, 1024):
                im = resize(im, (1024, 1024), preserve_range=True)
                im = im.astype(np.uint8)  # Ensure consistent dtype
            data["label"].append("Mutant")
            data["filename"].append(file)
            data["data"].append(im)
        for file in os.listdir(r'.\data\_WT'):
            im = imread(os.path.join(r'.\data\_WT', file))
            if len(im.shape) > 2: #resize
                im = im[1, :, :]
            if im.shape != (1024, 1024):
                im = resize(im, (1024, 1024), preserve_range=True)
                im = im.astype(np.uint8)
            data["label"].append("WildType")
            data["filename"].append(file)
            data["data"].append(im)
        joblib.dump(data, pklname)
        print(f"Dataset saved as {pklname}")
   
        return joblib.load(pklname)
    else:
        data = joblib.load(pklname)
        print('Data loaded')
        return data
    
def show_propoerties_dataset(data):
    '''
    Show properties of the dataset.
    '''
    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['description'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))
        
def print_image(image, number, type):
    '''
    Print an image.
    
    Parameters
    ----------
    image : array
        Image to print.
    number : int
        Number of the image.
    type : str
        Type of the image : 'original', 'Frangi', 'Hessian'...
    '''

    _, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Image {number} {type}")
    ax.set_axis_off()
    plt.show()
             
def preprocessing(RECOMPUTE_PREPROCESSING, X, pklname = pklname_dataset):
    print('Preprocessing images...')
    X_preprocessing =np.zeros_like(X,dtype=np.float64)
    if RECOMPUTE_PREPROCESSING:
        for im_num,_ in enumerate(X):
            image=X[im_num]
            print(f'Processing image {im_num+1}/{len(X)}')
            X_preprocessing[im_num]=ski.filters.frangi(image,black_ridges=False,sigmas=range(1,5,1),gamma=70)
        joblib.dump(X_preprocessing, pklname +'_preprocessing')
    else:
        X_preprocessing=joblib.load(pklname +'_preprocessing.pkl')
    print('Preprocessing done.')
    
    return X_preprocessing

def create_feature_vector():
    X_feat=np.zeros((len(X),N_FEAT*(N_BINS_FEAT-1)))

    return X_feat
    
    
if __name__ == "__main__":
    # Load 
    filename_pkl_dataset = 'dataset_2025-03-07_21-25-13.pkl'
    data = createDataset(False, filename_pkl_dataset) # If true, don't keep the name
    
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    # preprocessing
    filename_pkl_dataset = 'dataset_2025-03-07_21-48-33'
    X_preprocessing = preprocessing(False, X, filename_pkl_dataset)
    
    # show first image
    print_image(X[200], 'original', '')
    print_image(X_preprocessing[200], 'Frangi','')


    