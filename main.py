from utils import *
from analyse import *


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
    """if len(X) > 0:
        sample_idx = min(50, len(X) - 1)
        display_image(X[sample_idx], sample_idx, 'original')
        display_histogram(X[sample_idx], X[sample_idx].max(), sample_idx, 'original')
        display_image(X_preprocessed[sample_idx], sample_idx, 'Frangi')"""
    
    # Compute features
    X_features, features = create_feature_vector(X_preprocessed, y, recompute=False, pkl_name=filename_pkl_dataset)
    
    # Training
    print('Training model...')
    mean_corr_estim = train_model(X_features, y, SEED, N_RUNS, IN_PARAM)
    print(f'Mean accuracy: {100*mean_corr_estim:.1f}%')
    
    #show_errors(X_features, y, X_features, X, X_preprocessed, random_state=SEED)
    
    show_distribution_features(X_features, features)
    