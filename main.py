from utils import *
from training import *
from feature import *
from preprocessing import *

def pipeline():
    # ---------- Load dataset ----------
    filename_pkl_dataset = 'dataset_2025-03-19_19-32-21'
    data = create_dataset(reimport_images=False, pkl_name=filename_pkl_dataset + '.pkl')
    
    # Display dataset properties
    #show_dataset_properties(data)
    
    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    X_copy = X.copy()
    
    # ---------- Preprocessing ----------
    X_preprocessed, intensity, derivative_intensity, maxima, mask = get_preprocess_images(recompute=False, X=X_copy, pkl_name=filename_pkl_dataset)
    
    # Display sample images
    """if len(X) > 0:
        sample_idx = min(50, len(X) - 1)
        display_image(X[sample_idx], sample_idx, 'original')
        display_histogram(X[sample_idx], X[sample_idx].max(), sample_idx, 'original')
        display_image(X_preprocessed[sample_idx], sample_idx, 'Frangi')"""
    
    # ---------- Compute features ----------
    X_features, features = get_feature_vector(X_preprocessed, y, X, maxima, mask, recompute=True)
    
    # Training
    #mean_corr_estim = train_model(X_features, y, model_type='random_forest', n_runs=100)
    # hist_gradient_boosting, svm_rbf, knn, decision_tree, mlp, random_forest, siamese_network
    
    
    
    # ---------- Test all models and generate a comprehensive report ----------
    model_types = ['hist_gradient_boosting', 'svm_rbf', 'knn', 'decision_tree', 'mlp', 'random_forest', 'siamese_network']
    results = {}

    print("Starting model comparison...\n")

    for model_type in model_types:
        print(f"\n{'-'*50}")
        print(f"Evaluating {model_type}...")
        mean_corr_estim = train_model(X_features, y, model_type=model_type, n_runs=100)
        results[model_type] = mean_corr_estim
        print(f"{'-'*50}\n")

    # Find the best model
    best_model = max(results, key=results.get)

    # Print comprehensive report
    print("\n" + "="*70)
    print("MODEL COMPARISON REPORT".center(70))
    print("="*70)
    print(f"{'Model Type':<35} | {'Accuracy':<15} | {'Rank':<10}")
    print("-"*70)

    # Sort models by accuracy for ranking
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for i, (model, accuracy) in enumerate(sorted_results):
        is_best = model == best_model
        model_name = f"{model} {'(BEST)' if is_best else ''}"
        print(f"{model_name:<35} | {accuracy*100:.2f}%{' '*9} | {i+1}")

    print("="*70)
    print(f"Best model: {best_model}")
    print(f"Best accuracy: {results[best_model]*100:.2f}%")
    print("="*70)
    
    
    #show_errors(X_features, y, X_features, X, X_preprocessed, random_state=SEED)
    
    #show_distribution_features(features)
    

def test():
    # Load dataset
    filename_pkl_dataset = 'dataset_2025-03-11_10-07-49'
    data = create_dataset(reimport_images=True, test_random=True) #, pkl_name=filename_pkl_dataset + '.pkl')
    
    # Display dataset properties
    #show_dataset_properties(data)

    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    X_copy = X.copy()
    # Preprocessing
    #filename_pkl_dataset = 'dataset_2025-03-11_10-07-49'
    X_preprocessed, intensity, derivative_intensity, maxima, mask = get_preprocess_images(recompute=True, X=X_copy) #, pkl_name=filename_pkl_dataset)
    
    #X_hist = get_histogram_vector(X_preprocessed)
                                      
    #display_6_images(X[0], X_preprocessed[0], X_hist[0], X[1], X_preprocessed[1], X_hist[1], ["Original Mutant", "Process Mutant", "Histogram Mutant", "Original Wild-Type", "Process Wild-Type", "Histogram Wild-Type"])
    #display_4_images(X[0], X_preprocessed[0], X[1], X_preprocessed[1], ["Original Mutant", "Process Mutant", "Original Wild-Type", "Process Wild-Type"])
    

    # Compute features
    """mask = np.zeros_like(X_preprocessed)
    for i in range(len(mask)):
        mask[i] = creat_mask_synapse(X[i])
        print(f'Image {i} done')"""
        
        
    X_features, features = get_feature_vector(X_preprocessed, y, X, maxima, mask, recompute=True) # mask
    
    # Show features
    X_colored = colorize_image(X, features)
    display_4_images(X[0], X_colored[0], X[1], X_colored[1], ["Original Mutant", "Colored Mutant", "Original Wild-Type", "Colored Wild-Type"])

    # Training
    #mean_accuracy = train_model(X_features, y, SEED, N_RUNS, IN_PARAM)
    #print(f'Mean accuracy: {100*mean_accuracy:.1f}%')


if __name__ == "__main__":
    
    pipeline()
    #test()
    
    
    