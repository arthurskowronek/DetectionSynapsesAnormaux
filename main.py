from utils import *
from training import *
from feature import *
from preprocessing import *
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer, FunctionTransformer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
from PIL import Image

def test_model_accuracy(model_types, k_test=12):
    # ---------- Load dataset ----------
    filename_pkl_dataset = 'dataset_2025-04-16_11-42-01'
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
    filename_pkl_dataset = 'dataset_2025-04-16_11-42-01_preprocessing_1'
    X_preprocessed, intensity, derivative_intensity, maxima, mask, G, median_width, Measure_diff_slice, Measure_diff_points_segment = get_preprocess_images(method=2, recompute=False, X=X_copy, pkl_name=filename_pkl_dataset)
    
    # ---------- Compute features ----------
    if False:
        X_features, features = get_feature_vector(G, median_width, Measure_diff_slice, Measure_diff_points_segment, X_preprocessed, y, X, maxima, mask, intensity, recompute=True)    
        # save features to a xlsx file
        df = pd.DataFrame(X_features)
        columns = features['features name'][1]
        df.columns = columns
        # save to xlsx file in excel directory
        # create excel directory if it doesn't exist
        if not os.path.exists('excel'):
            os.makedirs('excel')
        # save to excel file
        df.to_excel('excel/features.xlsx', index=False)
    else:
        # load features from xlsx file
        df = pd.read_excel('excel/features.xlsx')
        X_features = df.values
        # get features name from xlsx file
        features_name = df.columns.values

    
    # Detect indice of elements in X_feat which contain only 0s
    indices = np.where(np.all(X_features == 0, axis=1))[0]
    # Remove these elements from X_feat and y
    X_features = np.delete(X_features, indices, axis=0)
    y = np.delete(y, indices, axis=0)
    
    # Detect features with all 0s
    indices = np.where(np.all(X_features == 0, axis=0))[0]
    # Remove these features from X_feat
    X_features = np.delete(X_features, indices, axis=1)
    
    
    # ---------- Feature Reduction ----------
    X_features_copied = X_features.copy()
    
    # Scale features
    scaler = StandardScaler() # the best for a PCA
    X_features_copied = scaler.fit_transform(X_features_copied)
    
    # change NaN values to 0
    X_features_copied = np.nan_to_num(X_features_copied)
    
    pca = PCA(n_components=31)
    X_features_PCA = pca.fit_transform(X_features_copied)
    
    #print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    #print(f"Explained variance: {pca.explained_variance_}")
    
    # plot the feature space with the 2 first components and label each point by if it is a mutant or not
    """plt.figure(figsize=(10, 10))
    plt.scatter(X_features_PCA[:, 0], X_features_PCA[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.title('Feature space with PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(ticks=[0, 1], label='Label') 
    plt.clim(-0.5, 1.5)
    plt.show()"""
    
    # 3D Plot
    """fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        X_features_PCA[:, 0],
        X_features_PCA[:, 1],
        X_features_PCA[:, 2],
        c=y,
        cmap='viridis',
        alpha=0.6
    )
    # Set color limits on the scatter, not on the colorbar
    sc.set_clim(-0.5, 1.5)
    cbar = fig.colorbar(sc, ticks=[0, 1], label='Label')
    ax.set_title('3D Feature space with PCA')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.tight_layout()
    plt.show()"""

    #X_features = X_features_PCA
    
    # ---------- Feature Selection ----------
    number_features_before = X_features.shape[1]
    
    # scale features
    X_features_selection = X_features.copy()
    scaler = StandardScaler()
    X_features_selection = scaler.fit_transform(X_features_selection)
    
    # Here we choose the top k features 
    X_features, selector = select_features(X_features_selection, y, k=k_test, method='lasso', verbose_features_selected=False) 

    number_features_after = X_features.shape[1]

    
    # ---------- Test all models and generate a comprehensive report ----------
    # Define all scalers to test
    scalers = {
        'NoScaler': FunctionTransformer(func=None),
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'Normalizer': Normalizer(),
        'QuantileTransformer': QuantileTransformer()
    }

    scalers = {
        'Normalizer': Normalizer()
        #'NoScaler': FunctionTransformer(func=None)
    }

    # Results will be a nested dictionary: scaler_name -> model_type -> accuracy
    all_results = {}
    best_combinations = {}

    print("Starting comprehensive model and scaler comparison...\n")

    # Loop through each scaler
    for scaler_name, scaler in scalers.items():
        print(f"\n{'#'*80}")
        print(f" TESTING WITH {scaler_name} ".center(80, '#'))
        print(f"{'#'*80}\n")
        
        if scaler_name == 'NoScaler':
            X_scaled = X_features.copy()  # Use a copy of the original data
            print("Using raw unscaled data...")
        else:
            X_scaled = scaler.fit_transform(X_features)
        
        # save features to a xlsx file
        df = pd.DataFrame(X_scaled)
        # save to xlsx file in excel directory
        df.to_excel('excel/features_crible.xlsx', index=False)
    
        # Store results for this scaler
        scaler_results = {}
        
        # Test each model with the current scaler
        for model_type in model_types:
            print(f"\n{'-'*50}")
            print(f"Evaluating {model_type} with {scaler_name}...")
            mean_corr_estim = train_model(X_scaled, y, verbose_plot=True, verbose_print=True, model_type=model_type)
            #best_model_optimized, best_params, mean_corr_estim = optimize_hyperparameters(X_scaled, y, model_type=model_type, method='grid')
            scaler_results[model_type] = mean_corr_estim
            print(f"Accuracy: {mean_corr_estim*100:.2f}%")
            print(f"{'-'*50}\n")
        
        # Store results for this scaler
        all_results[scaler_name] = scaler_results
        
        # Find the best model for this scaler
        best_model = max(scaler_results, key=scaler_results.get)
        best_accuracy = scaler_results[best_model]
        best_combinations[scaler_name] = (best_model, best_accuracy)
        
        # Print summary for this scaler
        print(f"\n{'='*70}")
        print(f"SUMMARY FOR {scaler_name}".center(70))
        print(f"{'='*70}")
        print(f"{'Model Type':<35} | {'Accuracy':<15} | {'Rank':<10}")
        print(f"{'-'*70}")
        
        # Sort models by accuracy for ranking
        sorted_results = sorted(scaler_results.items(), key=lambda x: x[1], reverse=True)
        for i, (model, accuracy) in enumerate(sorted_results):
            is_best = model == best_model
            model_name = f"{model} {'(BEST)' if is_best else ''}"
            print(f"{model_name:<35} | {accuracy*100:.2f}%{' '*9} | {i+1}")
        
        print(f"{'='*70}")
        print(f"Best model with {scaler_name}: {best_model}")
        print(f"Best accuracy: {best_accuracy*100:.2f}%")
        print(f"{'='*70}\n")

    # Final comprehensive report
    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL-SCALER COMPARISON REPORT".center(100))
    print("="*100)
    print(f"{'Scaling Method':<20} | {'Best Model':<35} | {'Accuracy':<15} | {'Overall Rank':<10}")
    print("-"*100)

    # Sort by overall best accuracy across all scalers
    sorted_combinations = sorted(best_combinations.items(), key=lambda x: x[1][1], reverse=True)
    for i, (scaler_name, (best_model, accuracy)) in enumerate(sorted_combinations):
        is_overall_best = i == 0
        scaler_display = f"{scaler_name} {'(BEST)' if is_overall_best else ''}"
        print(f"{scaler_display:<20} | {best_model:<35} | {accuracy*100:.2f}%{' '*9} | {i+1}")

    print("="*100)
    best_scaler, (best_model, best_acc) = sorted_combinations[0]
    print(f"Overall best combination: {best_scaler} with {best_model}")
    print(f"Overall best accuracy: {best_acc*100:.2f}%")
    print("="*100)

    # --- Create a heatmap visualization of all results ---

    # Convert nested dictionary to DataFrame for visualization
    results_df = pd.DataFrame({
        scaler_name: {model: acc for model, acc in models.items()} 
        for scaler_name, models in all_results.items()
    })

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(results_df.T * 100, annot=True, fmt='.2f', cmap='viridis', 
                xticklabels=results_df.index, yticklabels=results_df.columns)
    plt.title('Model Accuracy (%) for Different Scaling Methods')
    plt.ylabel('Scaling Method')
    plt.xlabel('Model Type')
    plt.tight_layout()
    plt.show()
    
    
    return best_acc
  
  
def display_errors():
    # ---------- Load dataset ----------
    filename_pkl_dataset = 'dataset_2025-04-16_11-42-01'
    data = create_dataset(reimport_images=False, pkl_name=filename_pkl_dataset + '.pkl')
    X = np.array(data['data'])
    # load features from xlsx file
    df = pd.read_excel('excel/features.xlsx')
    X_features_excel = df.values
    # Detect indice of elements in X_feat which contain only 0s
    indices = np.where(np.all(X_features_excel == 0, axis=1))[0]
    # Remove these elements from X_feat and y
    X = np.delete(X, indices, axis=0)
    
    print(f"Length of X: {len(X)}")
    
    
    # ---------- Load model ----------
    clf = joblib.load('models/svm_rbf_optimized.pkl')
    
    X_features = pd.read_excel('excel/features_crible.xlsx').values
    y = np.empty(109, dtype=object)
    y[:56] = 0 # mutants are the first 56 images
    y[56:] = 1 # wild-types are the last 53 images
    
    y_predict = clf.predict(X_features)
    
    indices_errors = np.where(y_predict != y)[0]
    print(f"Number of errors: {len(indices_errors)}")
    
    # create directory 'errors' in 'data' if it doesn't exist in 'data' directory
    if not os.path.exists('data/errors'):
        os.makedirs('data/errors')
    
    
    # plot images which were missclassified (indices_errors)
    for i in range(len(indices_errors)):
        index = indices_errors[i]
        # Display image
        real_classification = y[index]
        if real_classification == 0:
            title = 'Classify as a wild-type but it is a mutant'
        else:
            title = 'Classify as mutant but it is a wild-type'
        display_image(X[index], index, title)
        # create a copy of images which were missclassified and put this copy in 'errors' directory
        image = X[index]
        image = Image.fromarray(image) # convert to PIL image
        # save image in 'errors' directory
        image.save(f'data/errors/image_{index}.tif')
                
       
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
    X_preprocessed, intensity, derivative_intensity, maxima, mask, G, median_width, Measure_diff_slice, Measure_diff_points_segment = get_preprocess_images(method=2, recompute=True, X=X_copy) #, pkl_name=filename_pkl_dataset)
    
    
    #X_hist = get_histogram_vector(X_preprocessed)
                                      
    #display_6_images(X[0], X_preprocessed[0], X_hist[0], X[1], X_preprocessed[1], X_hist[1], ["Original Mutant", "Process Mutant", "Histogram Mutant", "Original Wild-Type", "Process Wild-Type", "Histogram Wild-Type"])
    display_4_images(X[0], X_preprocessed[0], X[1], X_preprocessed[1], ["Original Mutant", "Process Mutant", "Original Wild-Type", "Process Wild-Type"])
    

    # ---------- Compute features ----------
    """mask = np.zeros_like(X_preprocessed)
    for i in range(len(mask)):
        mask[i] = creat_mask_synapse(X[i])
        print(f'Image {i} done')"""
        
        
    X_features, features = get_feature_vector(G, median_width, Measure_diff_slice, Measure_diff_points_segment, X_preprocessed, y, X, maxima, mask, intensity, recompute=True) # mask
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
    
    
    # load dataset from excel file
    data = pd.read_excel('excel/features_crible.xlsx')
    # Convert to numpy arrays
    X_features = data.values
    y = np.empty(109, dtype=object)
    y[:56] = 'Mutant'
    y[56:] = 'Wild-Type'
        
    # Create masks for Mutant and Wild-Type samples
    mutant_mask = y == 'Mutant'
    wild_type_mask = y == 'Wild-Type'

    # Get all mutant and wild-type samples
    mutant_samples = X_features[mutant_mask]
    wild_type_samples = X_features[wild_type_mask]

    # Select 1 random mutant sample
    random_mutant_index = np.random.choice(len(mutant_samples), 1)[0]
    selected_mutant = mutant_samples[random_mutant_index:random_mutant_index+1]

    # Create new dataset with 1 random mutant and all wild-type samples
    X_new = np.vstack([selected_mutant, wild_type_samples])
    y_new = np.array(['Mutant'] + ['Wild-Type'] * len(wild_type_samples))

    # Shuffle the new dataset
    X_features, y = shuffle(X_new, y_new)
    

    """# ---------- Load dataset ----------
    # load dataset : à créer
    data = create_dataset(reimport_images=True, test_random_mutant=True, test_random_wildtype=False, data_augmentation=False)
    
    # ---------- Preprocessing ----------
    # Convert to numpy arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    # Convert labels to numeric
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    
    X_copy = X.copy()
    
    filename_pkl_dataset = 'dataset_2025-04-16_11-42-01_preprocessing_1'
    X_preprocessed, intensity, derivative_intensity, maxima, mask, G, median_width, Measure_diff_slice, Measure_diff_points_segment = get_preprocess_images(method=2, recompute=True, X=X_copy) 
    
    # ---------- Compute features ----------
    X_features, features = get_feature_vector(G, median_width, Measure_diff_slice, Measure_diff_points_segment, X_preprocessed, y, X, maxima, mask, intensity, recompute=True)    

    print(f"Number of features before 0s : {X_features.shape[1]}")

    # Detect features with all 0s
    indices = np.where(np.all(X_features == 0, axis=0))[0]
    # Remove these features from X_feat
    X_features = np.delete(X_features, indices, axis=1)


    print(f"Number of features before selection : {X_features.shape[1]}")
    # ---------- Feature Selection ----------
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    
    # compute the features to keep
    X_features, selector = select_features(X_features, y, k=12, method='lasso', verbose_features_selected=False) 
    
    print(f"Number of features after selection : {X_features.shape[1]}")"""

    # get indice of features to keep in 'models/selected_features.txt' file
    """with open('models/selected_features.txt', 'r') as f:
        selected_features = f.read().splitlines()
    selected_features = [int(i) for i in selected_features]
    X_features = X_features[:, selected_features]"""
    
    # ---------- Probabilities ----------
    # load model from disk
    #clf = joblib.load('models/model.pkl')
    clf = joblib.load('models/svm_rbf_optimized.pkl')
    
    # Detect indice of elements in X_feat which contain only 0s
    #indices = np.where(np.all(X_features == 0, axis=1))[0]
    indices = []
    
    # compute probabilities
    # X_proba is a probability list of each image to be a mutant and an index list of the images
    X_proba = np.zeros((len(X_features), 2))
    for i in range(len(X_features)):   
        if i in indices: # detect roll worm
            X_proba[i][0] = 1
            X_proba[i][1] = i # store the index of the image
        else:      
            #print(clf.classes_)
            proba = clf.predict_proba(X_features[i].reshape(1, -1))[0] # proba[1] = proba of being a wild-type, proba[0] = proba of being a mutant
            X_proba[i][0] = proba[0]
            X_proba[i][1] = i # store the index of the image
  
            
    # sort the images by probability
    X_proba = X_proba[X_proba[:,0].argsort()[::-1]] # sort by descending order and keep the original index
    
    # print the probabilities
    """for i in range(len(X_proba)):
        print(f"Image {i+1}/{len(X_proba)}: {X_proba[i][0]} - {X_proba[i][1]}")"""
    
    # Show images in order of probability and ask the user if it is a mutant or not
    # the user's answers are stored in a list
    # the images already seen are stored in a list
    
    images_seen = []
    user_answers = []
    not_seen = True
    i = 0
    while i < len(X_features) and not_seen:
        
        # show the image
        # ask the user if it is a mutant or not
        # store the answer
        # store the image
        
        index = int(X_proba[i][1])
        
        # Display image and ask user if it is a mutant
        """print(f"Image {i+1}/{len(X_features)}")
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
        print(f"Image {i+1}/{len(X_features)} : {X_proba[i][0]} - {y[index]}")
        
        if y[index] == 'Mutant':
            print(f"Mutant found in {i+1} images. Probability: {X_proba[i][0]}")
            not_seen = False
        #else:
            #print(f"Mutant not found. Image {i+1}/{len(X_features)}")
        
        i += 1
        
    #print(f"Mutant found in {i} images")
    
    return i
    # ---------- Model improvement ----------
    # the model is improved with the user's answers
    

if __name__ == "__main__":
        
        
    #display_errors()
        
    # ---------- Test model accuracy ----------
    model_types = ['hist_gradient_boosting', 'svm_rbf', 'random_forest', 'knn', 'mlp', 'siamese_network']
    model_types = ['svm_rbf']
    """best_acc = []
    for i in range(40,51):
        print(f"Testing with {i} features")
        best_acc.append(test_model_accuracy(model_types, i))
    
    # save the best accuracy to a txt file
    with open('best_accuracy.txt', 'w') as f:
        for i in range(len(best_acc)):
            f.write(f"{i} {best_acc[i]}\n")
            
    
    # plot the best accuracy
    plt.plot(range(40,51), best_acc)
    plt.xlabel('Number of features')
    plt.ylabel('Best accuracy')
    plt.title('Best accuracy vs Number of features')
    plt.show()"""
    
    #test_model_accuracy(model_types)
    
    
    # ---------- Test pipeline ----------
    #test_pipeline()
    
    
    # ---------- Test crible genetique ----------
    """number_images_seen = []
    for i in range(20):
        number_images_seen.append(crible_genetique())
        
    # save the number of images seen to a xlsx file
    df = pd.DataFrame(number_images_seen)
    # save to xlsx file in excel directory
    if not os.path.exists('excel'):
        os.makedirs('excel')
    # save to excel file
    df.to_excel('excel/number_images_seen.xlsx', index=False)
    
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
    
    
    
    
    
    