from utils import *
from training import *
from feature import *
from preprocessing import *
from crible_functions import *


def test_model_accuracy(model_types):
    # ---------- Load dataset ----------
    filename_pkl_dataset = 'dataset_2025-04-04_11-35-21'
    data = create_dataset(reimport_images=False, test_random_mutant=False, test_random_wildtype=False, data_augmentation=False, pkl_name=filename_pkl_dataset + '.pkl')
    
    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    # Convert labels to numeric
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    
    X_copy = X.copy()
    
    # ---------- Preprocessing ----------
    filename_pkl_dataset = 'dataset_2025-03-12_22-29-09_preprocessing_1'
    X_preprocessed, intensity, derivative_intensity, maxima, mask = get_preprocess_images(recompute=False, X=X_copy, pkl_name=filename_pkl_dataset)
    
    # ---------- Compute features ----------
    X_features, features = get_feature_vector(X_preprocessed, y, X, maxima, mask, intensity, recompute=True)
    
    # ---------- Feature Reduction ----------
    # Scale features
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    
    # change NaN values to 0
    X_features = np.nan_to_num(X_features)
    
    pca = PCA(n_components=2)
    X_features = pca.fit_transform(X_features)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Explained variance: {pca.explained_variance_}")
    
    # plot the feature space with the 2 first components and label each point by if it is a mutant or not
    plt.figure(figsize=(10, 10))
    plt.scatter(X_features[:, 0], X_features[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.title('Feature space with PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(ticks=[0, 1], label='Label') 
    plt.clim(-0.5, 1.5)
    plt.show()
    
    # ---------- Feature Selection ----------
    number_features_before = X_features.shape[1]
    
    # scale features
    scaler = StandardScaler()
    #X_features = scaler.fit_transform(X_features)
    
    # Here we choose the top k features 
    #X_features, selector = select_features(X_features, y, k=10, method='mRMR', verbose_features_selected=False) 
    
    number_features_after = X_features.shape[1]
    

    # ---------- Test all models and generate a comprehensive report ----------
    results = {}

    print("Starting model comparison...\n")

    for model_type in model_types:
        print(f"\n{'-'*50}")
        print(f"Evaluating {model_type}...")
        mean_corr_estim = train_model(X_features, y, verbose_plot = True, model_type=model_type, n_runs=100)
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
    
    
    #print(f"Number of features before: {number_features_before}")
    #print(f"Number of features after: {number_features_after}")
    
       
def test_pipeline():
    # ---------- Load dataset ----------
    filename_pkl_dataset = 'dataset_56'
    data = create_dataset(reimport_images=True, test_random_mutant=True, test_random_wildtype=True, data_augmentation=False) #, pkl_name=filename_pkl_dataset + '.pkl')
    
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
    display_4_images(X[0], X_preprocessed[0], X[1], X_preprocessed[1], ["Original Mutant", "Process Mutant", "Original Wild-Type", "Process Wild-Type"])
    

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
    data = create_dataset(reimport_images=True, test_random_mutant=True, test_random_wildtype=False, data_augmentation=False)
    
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
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    
    # compute the features to keep
    X_features, selector = select_features(X_features, y, k=10, method='mRMR', verbose_features_selected=False) 
    
    # get indice of features to keep in 'models/selected_features.txt' file
    """with open('models/selected_features.txt', 'r') as f:
        selected_features = f.read().splitlines()
    selected_features = [int(i) for i in selected_features]
    X_features = X_features[:, selected_features]"""
    
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
        
        # Display image and ask user if it is a mutant
        """print(f"Image {i+1}/{len(X)}")
        print(f"Is this image a mutant ?")
        display_image(X[index], index, 'Is this image a mutant ?')
        answer = input("y/n") 
        if answer == 'y':
            user_answers.append(1)
            not_seen = False
        else:
            user_answers.append(0)
        images_seen.append(int(X_proba[i][1]))
        print("\n")"""
        
        # check if the image is a mutant
        # if it is a mutant, the loop stops
        
        if y[index] == 'Mutant':
            print("Mutant found")
            not_seen = False
        else:
            print(f"Mutant not found. Image {i+1}/{len(X)}")
        
        i += 1
        
    print(f"Mutant found in {i} images")
    
    
    return i
    # ---------- Model improvement ----------
    # the model is improved with the user's answers
    

if __name__ == "__main__":
        
    # ---------- Test model accuracy ----------
    model_types = ['hist_gradient_boosting', 'svm_rbf', 'random_forest', 'knn', 'decision_tree', 'mlp', 'siamese_network']
    model_types = ['hist_gradient_boosting']
    test_model_accuracy(model_types)
    
    
    # ---------- Test pipeline ----------
    #test_pipeline()
    
    
    # ---------- Test crible genetique ----------
    """number_images_seen = []
    for i in range(20):
        number_images_seen.append(crible_genetique())
    
    print(f"Average number of images seen: {np.mean(number_images_seen)}")
    
    # show histogram of the number of images seen
    plt.hist(number_images_seen, bins=20)
    plt.xlabel('Number of images seen')
    plt.ylabel('Frequency')
    plt.title('Histogram of the number of images seen')
    plt.show()
    
    # plot the number of images seen
    plt.plot(number_images_seen)
    plt.xlabel('Number of trials')
    plt.ylabel('Number of images seen')
    plt.title('Number of images seen vs Number of trials')
    plt.show()
    
    # show the number of images seen
    print(number_images_seen)"""
    
    
    
    
    
    
    