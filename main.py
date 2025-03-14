from utils import *


def pipeline():
    # Load dataset
    filename_pkl_dataset = 'dataset_2025-03-10_08-05-48'
    data = create_dataset(reimport_images=False, pkl_name=filename_pkl_dataset + '.pkl')
    
    # Display dataset properties
    show_dataset_properties(data)
    
    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    # Preprocessing
    X_preprocessed = get_preprocess_images(recompute=False, X=X, pkl_name=filename_pkl_dataset)
    
    # Display sample images
    if len(X) > 0:
        sample_idx = min(50, len(X) - 1)
        display_image(X[sample_idx], sample_idx, 'original')
        display_histogram(X[sample_idx], X[sample_idx].max(), sample_idx, 'original')
        display_image(X_preprocessed[sample_idx], sample_idx, 'Frangi')
    
    # Compute features
    X_features, features = get_feature_vector(X_preprocessed, y, recompute=False, pkl_name=filename_pkl_dataset)
    
    # Training
    print('Training model...')
    mean_corr_estim = train_model(X_features, y, SEED, N_RUNS, IN_PARAM)
    print(f'Mean accuracy: {100*mean_corr_estim:.1f}%')
    
    #show_errors(X_features, y, X_features, X, X_preprocessed, random_state=SEED)
    
    show_distribution_features(X_features, features)
    

def test():
    # Load dataset
    filename_pkl_dataset = 'dataset_2025-03-11_10-07-49'
    data = create_dataset(reimport_images=False, pkl_name=filename_pkl_dataset + '.pkl')
    
    # Display dataset properties
    #show_dataset_properties(data)

    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    # Preprocessing
    #filename_pkl_dataset = 'dataset_2025-03-11_10-07-49'
    X_preprocessed, intensity, derivative_intensity = get_preprocess_images(recompute=True, X=X) #, pkl_name=filename_pkl_dataset)
    
    #X_hist = get_histogram_vector(X_preprocessed)
                                      
    #display_6_images(X[0], X_preprocessed[0], X_hist[0], X[1], X_preprocessed[1], X_hist[1], ["Original Mutant", "Process Mutant", "Histogram Mutant", "Original Wild-Type", "Process Wild-Type", "Histogram Wild-Type"])
    display_4_images(X[0], X_preprocessed[0], X[1], X_preprocessed[1], ["Original Mutant", "Process Mutant", "Original Wild-Type", "Process Wild-Type"])
    

    # Compute features
    """mask = np.zeros_like(X_preprocessed)
    for i in range(len(mask)):
        mask[i] = creat_mask_synapse(X[i])
        print(f'Image {i} done')"""
        
        
    X_features, features = get_feature_vector(X_preprocessed, y, recompute=True) # mask
    
    # Show features
    X_colored = colorize_image(X, features)
    display_4_images(X[0], X_colored[0], X[1], X_colored[1], ["Original Mutant", "Colored Mutant", "Original Wild-Type", "Colored Wild-Type"])

    # Training
    #mean_accuracy = train_model(X_features, y, SEED, N_RUNS, IN_PARAM)
    #print(f'Mean accuracy: {100*mean_accuracy:.1f}%')


if __name__ == "__main__":
    
    #pipeline()
    test()
    
    
    