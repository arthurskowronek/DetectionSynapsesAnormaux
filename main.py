from utils import *
from training import *
from feature import *
from preprocessing import *
from crible_functions import *


def pipeline_optimisation(k_features):
    # ---------- Load dataset ----------
    filename_pkl_dataset = 'dataset_2025-03-21_13-32-05'
    data = create_dataset(reimport_images=True, test_random=False, data_augmentation=False) #, pkl_name=filename_pkl_dataset + '.pkl')
    
    # Display dataset properties
    #show_dataset_properties(data)
    
    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    # Convert labels to numeric
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    
    X_copy = X.copy()
    
    # ---------- Preprocessing ----------
    X_preprocessed, intensity, derivative_intensity, maxima, mask = get_preprocess_images(recompute=True, X=X_copy) #, pkl_name=filename_pkl_dataset)
    
    # Display sample images
    """if len(X) > 0:
        sample_idx = min(50, len(X) - 1)
        display_image(X[sample_idx], sample_idx, 'original')
        display_histogram(X[sample_idx], X[sample_idx].max(), sample_idx, 'original')
        display_image(X_preprocessed[sample_idx], sample_idx, 'Frangi')"""
    
    # ---------- Compute features ----------
    X_features, features = get_feature_vector(X_preprocessed, y, X, maxima, mask, intensity, recompute=True)
    
    # ---------- Feature Selection ----------
    number_features_before = X_features.shape[1]
    # Here we choose the top k features 
    X_features, selector = select_features(X_features, y, k=k_features, method='boruta', verbose_features_selected=False) 
    
    number_features_after = X_features.shape[1]
    
    # Training
    #mean_corr_estim = train_model(X_features, y, model_type='random_forest', n_runs=100)
    # hist_gradient_boosting, svm_rbf, knn, decision_tree, mlp, random_forest, siamese_network
    
    
    
    # ---------- Test all models and generate a comprehensive report ----------
    model_types = ['hist_gradient_boosting', 'svm_rbf', 'random_forest', 'knn', 'decision_tree', 'mlp', 'siamese_network']
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
    
    
    print(f"Number of features before: {number_features_before}")
    print(f"Number of features after: {number_features_after}")

    
    #show_errors(X_features, y, X_features, X, X_preprocessed, random_state=SEED)
    
    #show_distribution_features(features)
    
    return results[best_model]*100
 
def pipeline():
    # ---------- Load dataset ----------
    filename_pkl_dataset = 'dataset_2025-03-21_13-32-05'
    data = create_dataset(reimport_images=True, data_augmentation=False) #, pkl_name=filename_pkl_dataset + '.pkl')
    
    # ---------- Preprocessing ----------
    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    X_copy = X.copy()
    
    # Convert labels to numeric
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    
    # Preprocess images
    X_preprocessed, intensity, derivative_intensity, maxima, mask = get_preprocess_images(recompute=True, X=X_copy) #, pkl_name=filename_pkl_dataset)
    
    
    # ---------- Compute features ----------
    X_features, features = get_feature_vector(X_preprocessed, y, X, maxima, mask, intensity, recompute=True)
    
    
    # ---------- Feature Selection ----------
    # Choose the top k features 
    X_features, selector = select_features(X_features, y, k=55, method='boruta', verbose_features_selected=True) 
    
    
    # ---------- Training ----------
    mean_corr_estim = train_model(X_features, y, model_type='svm_rbf', n_runs=100)

    print(f'Mean accuracy: {100*mean_corr_estim:.1f}%')
    
    #show_errors(X_features, y, X_features, X, X_preprocessed, random_state=SEED)
    
    #show_distribution_features(features)
    
def test():
    # ---------- Load dataset ----------
    filename_pkl_dataset = 'dataset_2025-03-11_10-07-49'
    data = create_dataset(reimport_images=True, test_random=True, augment_data=True) #, pkl_name=filename_pkl_dataset + '.pkl')
    
    # Display dataset properties
    #show_dataset_properties(data)

    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    X_copy = X.copy()
    
    # ---------- Preprocessing ----------
    #filename_pkl_dataset = 'dataset_2025-03-11_10-07-49'
    X_preprocessed, intensity, derivative_intensity, maxima, mask = get_preprocess_images(recompute=True, X=X_copy) #, pkl_name=filename_pkl_dataset)
    
    
    #X_hist = get_histogram_vector(X_preprocessed)
                                      
    #display_6_images(X[0], X_preprocessed[0], X_hist[0], X[1], X_preprocessed[1], X_hist[1], ["Original Mutant", "Process Mutant", "Histogram Mutant", "Original Wild-Type", "Process Wild-Type", "Histogram Wild-Type"])
    #display_4_images(X[0], X_preprocessed[0], X[1], X_preprocessed[1], ["Original Mutant", "Process Mutant", "Original Wild-Type", "Process Wild-Type"])
    

    # ---------- Compute features ----------
    """mask = np.zeros_like(X_preprocessed)
    for i in range(len(mask)):
        mask[i] = creat_mask_synapse(X[i])
        print(f'Image {i} done')"""
        
        
    X_features, features = get_feature_vector(X_preprocessed, y, X, maxima, mask, intensity, recompute=True) # mask
    # X_reduced = select_features(X_feat, y, method='lasso')
    
    # Show features
    X_colored = colorize_image(X, features)
    display_4_images(X[0], X_colored[0], X[1], X_colored[1], ["Original Mutant", "Colored Mutant", "Original Wild-Type", "Colored Wild-Type"])

    # ---------- Training ----------
    #mean_accuracy = train_model(X_features, y, SEED, N_RUNS, IN_PARAM)
    #print(f'Mean accuracy: {100*mean_accuracy:.1f}%')

def crible_genetique():
    
    # on a un dataset avec 1 mutant parmi X wild-type
    # à chaque image est assigné sa proba d'être un mutant
    # les images sont montrées dans l'ordre décroissant de proba
    # on doit trouver le mutant le plus probable en un minimum d'images
    # l'utilisateur doit dire si l'image est un mutant ou non
    # on garde en mémoire les images déjà vues
    # le modèle s'améliorer en fonction des réponses de l'utilisateur
    
    
    # ---------- Load dataset ----------
    # load dataset : à créer
    data = create_dataset(reimport_images=True, test_random=False, data_augmentation=False)
    
    # ---------- Preprocessing ----------
    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    X_copy = X.copy()
    
    # ---------- Preprocessing ----------
    #filename_pkl_dataset = 'dataset_2025-03-11_10-07-49'
    X_preprocessed, intensity, derivative_intensity, maxima, mask = get_preprocess_images(recompute=True, X=X_copy) #, pkl_name=filename_pkl_dataset)
    
    # ---------- Compute features ----------
    X_features, features = get_feature_vector(X_preprocessed, y, X_copy, maxima, mask, intensity, recompute=True)
    
    # ---------- Feature Selection ----------
    # get indice of features to keep in 'models/selected_features.txt' file
    with open('models/selected_features.txt', 'r') as f:
        selected_features = f.read().splitlines()
    selected_features = [int(i) for i in selected_features]
    
    X_features = X_features[:, selected_features]
    
    # ---------- Probabilities ----------
    # load model from disk
    clf = joblib.load('models/model.pkl')
    # compute probabilities
    # X_proba is a probability list of each image to be a mutant and an index list of the images
    X_proba = np.zeros((len(X), 2))
    for i in range(len(X)):            
        proba = clf.predict_proba(X_features[i].reshape(1, -1)) # proba[0] = proba of being a wild-type, proba[1] = proba of being a mutant
        X_proba[i][0] = proba[0][1]
        X_proba[i][1] = i
  
            
    # sort the images by probability
    X_proba = X_proba[X_proba[:,0].argsort()[::-1]] # sort by descending order and keep the original index
    
    # Show images in order of probability and ask the user if it is a mutant or not
    # the user's answers are stored in a list
    # the images already seen are stored in a list
    
    images_seen = []
    user_answers = []
    not_seen = True
    i = 0
    while i < len(X) and not_seen:
        
        # show the image
        # ask the user if it is a mutant or not
        # store the answer
        # store the image
        
        index = int(X_proba[i][1])
        
        print(f"Image {i+1}/{len(X)}")
        print(f"Is this image a mutant ?")
        display_image(X[index], index, 'Is this image a mutant ?')
   
        answer = input("y/n") 
        
        if answer == 'y':
            user_answers.append(1)
            not_seen = False
        else:
            user_answers.append(0)
        
        images_seen.append(int(X_proba[i][1]))
        
        print("\n")
    
    # ---------- Model improvement ----------
    # the model is improved with the user's answers
    # the model is saved on disk
    

if __name__ == "__main__":
    
    """accuracy = []
    for i in range(55, 56):
        accuracy.append(pipeline(i))
    
    # plot the accuracy
    plt.plot(accuracy)
    plt.xlabel('Number of features selected')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of features selected')
    plt.show()"""
    
    #pipeline_optimisation(55)
    
    #pipeline()
    
    #test()
    
    crible_genetique()
    
    
    
    
    
    