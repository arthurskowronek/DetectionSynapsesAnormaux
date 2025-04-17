import joblib
import seaborn as sns
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from utils import *


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SiameseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        self.fc = nn.Linear(hidden_dim//4, 1)
        
    def forward_one(self, x):
        return self.encoder(x)
    
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        distance = torch.abs(out1 - out2)
        out = self.fc(distance)
        return torch.sigmoid(out)

class SiamesePairDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.labels = np.unique(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 50% chance of same class, 50% chance of different class
        same_class = np.random.random() > 0.5
        
        # Get first sample
        x1 = self.X[idx]
        y1 = self.y[idx]
        
        # Get second sample
        if same_class:
            # Find indices of samples with the same class
            indices = np.where(self.y == y1)[0]
            if len(indices) > 1:
                # Pick a random sample from the same class (excluding the current one)
                idx2 = np.random.choice([i for i in indices if i != idx])
            else:
                # If only one sample in class, use it again
                idx2 = idx
        else:
            # Find indices of samples with different classes
            indices = np.where(self.y != y1)[0]
            if len(indices) > 0:
                # Pick a random sample from a different class
                idx2 = np.random.choice(indices)
            else:
                # If no samples from different classes, use the current one
                idx2 = idx
                same_class = True
        
        x2 = self.X[idx2]
        
        return torch.FloatTensor(x1), torch.FloatTensor(x2), torch.FloatTensor([float(same_class)])

class SiameseClassifier:
    def __init__(self, input_dim, hidden_dim=128, lr=0.001, n_epochs=10, batch_size=32):
        self.model = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prototypes = {}
        
    def fit(self, X, y):
        # Convert to numpy arrays if they're not already
        X = np.array(X)
        y = np.array(y)
        
        # Initialize model
        self.model = SiameseNetwork(self.input_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        # Create dataset and dataloader
        dataset = SiamesePairDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train model
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for x1, x2, label in dataloader:
                x1, x2, label = x1.to(self.device), x2.to(self.device), label.to(self.device)
                
                # Forward pass
                output = self.model(x1, x2)
                loss = criterion(output, label)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print epoch statistics
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        # Compute prototypes for each class
        self.model.eval()
        for label in np.unique(y):
            indices = np.where(y == label)[0]
            class_samples = X[indices]
            
            # Convert to tensor
            class_samples_tensor = torch.FloatTensor(class_samples).to(self.device)
            
            # Compute mean embedding
            with torch.no_grad():
                embeddings = self.model.forward_one(class_samples_tensor)
                prototype = embeddings.mean(dim=0)
                self.prototypes[label] = prototype
        
        return self
    
    def predict(self, X):
        # Convert to numpy array if it's not already
        X = np.array(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.forward_one(X_tensor)
        
        # Compute distance to each prototype
        predictions = []
        for i in range(len(X)):
            embedding = embeddings[i]
            min_distance = float('inf')
            best_label = None
            
            for label, prototype in self.prototypes.items():
                distance = torch.norm(embedding - prototype).item()
                if distance < min_distance:
                    min_distance = distance
                    best_label = label
            
            predictions.append(best_label)
        
        return np.array(predictions)

def train_model_epoch(X_features, y, verbose_plot=False, model_type='random_forest', n_runs=10, test_size=0.3, random_state=26):
    """
    Train a model on the feature vectors with support for multiple classifier types.
    Plots the accuracy across epochs when verbose_plot is True.
    """
    print(f"Training {model_type} model...")
    # Convert to numpy arrays if necessary
    X_features = np.array(X_features)
    y = np.array(y)
    # Encode labels numerically
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])
    correct_estimations = []
    
    # For tracking accuracy across epochs
    train_accuracies = []
    test_accuracies = []
    epochs = []
    
    seed = RandomState(random_state) if isinstance(random_state, int) else random_state
    
    # For iterative models that support partial_fit or warm_start
    iterative_models = {
        'mlp': (MLPClassifier, {'hidden_layer_sizes': (100, 50), 'max_iter': 1, 'warm_start': True, 'random_state': seed}),
    }
    
    # For models that don't natively support tracking per-epoch performance
    standard_models = {
        'hist_gradient_boosting': (HistGradientBoostingClassifier, {'max_iter': 100, 'random_state': seed}),
        'svm_rbf': (SVC, {'kernel': 'rbf', 'probability': True, 'random_state': seed}),
        'knn': (KNeighborsClassifier, {'n_neighbors': 5}),
        'decision_tree': (DecisionTreeClassifier, {'random_state': seed}),
        'random_forest': (RandomForestClassifier, {'n_estimators': 10, 'random_state': seed}),
    }
    
    # Split data once for consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_numeric, test_size=test_size, random_state=seed
    )
    
    # Select appropriate training approach based on model type
    if model_type in iterative_models:
        # For models that support incremental training
        clf_class, clf_params = iterative_models[model_type]
        clf = clf_class(**clf_params)
        
        num_epochs = 100  # Set number of epochs for iterative training
        
        for epoch in range(num_epochs):
            # Train for one epoch
            clf.partial_fit(X_train, y_train, classes=np.unique(y_numeric))
            
            # Calculate training and test accuracy
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            
            # Store metrics
            epochs.append(epoch + 1)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            # Print progress periodically
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Store final accuracy
        final_accuracy = test_acc
        correct_estimations.append(final_accuracy)
        
    else:
        # For models that don't support incremental training, we'll simulate epochs
        # by increasing complexity gradually
        
        if model_type == 'random_forest':
            # For random forest, we can simulate epochs by increasing trees
            param_range = range(1, 51)  # 1 to 50 trees
            param_name = 'n_estimators'
        elif model_type == 'hist_gradient_boosting':
            # For gradient boosting, increase iterations
            param_range = range(1, 101)  # 1 to 100 iterations
            param_name = 'max_iter'
        elif model_type == 'decision_tree':
            # For decision tree, increase max_depth
            param_range = range(1, 21)  # 1 to 20 max depth
            param_name = 'max_depth'
        else:
            # Default for other models - we'll use a subset of data to simulate epochs
            param_range = np.linspace(0.1, 1.0, 20)  # 10% to 100% of training data
            param_name = 'subset'
        
        clf_class = standard_models[model_type][0]
        base_params = standard_models[model_type][1].copy()
        
        for i, param_value in enumerate(param_range):
            if param_name == 'subset':
                # For subset simulation, use increasing amounts of training data
                subset_size = int(len(X_train) * param_value)
                X_train_subset = X_train[:subset_size]
                y_train_subset = y_train[:subset_size]
                
                clf = clf_class(**base_params)
                clf.fit(X_train_subset, y_train_subset)
            else:
                # For parameter-based simulation
                params = base_params.copy()
                params[param_name] = param_value
                clf = clf_class(**params)
                clf.fit(X_train, y_train)
            
            # Calculate training and test accuracy
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            
            # Store metrics
            epochs.append(i + 1)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            # Print progress periodically
            if (i + 1) % 5 == 0:
                print(f"Epoch {i + 1}/{len(param_range)}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Store final accuracy
        final_accuracy = test_acc
        correct_estimations.append(final_accuracy)
    
    # Plot results if requested
    if verbose_plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion Matrix
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(cmap='Blues', ax=axes[0])
        axes[0].set_title("Confusion Matrix")
        
        # Accuracy vs Epochs Plot
        axes[1].plot(epochs, train_accuracies, '-o', label='Training Accuracy')
        axes[1].plot(epochs, test_accuracies, '-o', label='Test Accuracy')
        axes[1].set_title(f"Accuracy vs Epochs ({model_type})")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim([0, 1.05])
        axes[1].legend(loc="best")
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # Save model
    joblib.dump(clf, 'models/model.pkl')
    
    # Final accuracy 
    mean_correct_estim = np.mean(correct_estimations)
    print(f"Final accuracy: {mean_correct_estim:.4f}")
    return mean_correct_estim

def optimize_hyperparameters(X_features, y, model_type='random_forest', method='grid', n_iter=20, cv=5):
    """
    Optimize hyperparameters for the selected model type using either grid search or random search.
    
    Parameters:
    -----------
    X_features : array-like
        Feature vectors
    y : array-like
        Target labels
    model_type : str
        Type of model to optimize ('random_forest', 'svm_rbf', 'knn', etc.)
    method : str
        Optimization method ('grid' or 'random')
    n_iter : int
        Number of parameter settings sampled in random search
    cv : int
        Number of cross-validation folds
    
    Returns:
    --------
    best_model : estimator
        Trained model with optimized hyperparameters
    best_params : dict
        Best hyperparameters found
    best_score : float
        Score of the best model
    """
    # Convert to numpy arrays if needed
    X_features = np.array(X_features)
    y = np.array(y)
    
    # Encode labels numerically if they're not already
    if not np.issubdtype(y.dtype, np.number):
        unique_labels = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])
    
    # Define parameter grids for each model type
    param_grids = {
        'random_forest': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],
            'bootstrap': [True, False]
        },
        'hist_gradient_boosting': {
            'max_iter': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [None, 5, 10, 20],
            'min_samples_leaf': [1, 5, 20]
        },
        'svm_rbf': {
            'C': [5, 10, 15, 20],  
            'gamma': [0.05, 0.1, 0.15],
            'class_weight': [None, 'balanced']
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # p=1 is Manhattan, p=2 is Euclidean
        },
        'decision_tree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy', 'log_loss']
        },
        'mlp': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }
    
    # Initialize the base model
    base_models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'hist_gradient_boosting': HistGradientBoostingClassifier(random_state=42),
        'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42),
        'knn': KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'mlp': MLPClassifier(max_iter=300, random_state=42)
    }
    
    if model_type not in base_models:
        raise ValueError(f"Invalid model_type: {model_type}. Options are: {', '.join(base_models.keys())}")
    
    base_model = base_models[model_type]
    param_grid = param_grids[model_type]
    
    # Set up the search
    if method == 'grid':
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
    elif method == 'random':
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='accuracy',
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    else:
        raise ValueError("method should be either 'grid' or 'random'")
    
    # Perform the search
    print(f"Optimizing {model_type} with {method} search...")
    search.fit(X_features, y)
    
    # Get the best model and parameters
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Save the best model
    joblib.dump(best_model, f'models/{model_type}_optimized.pkl')
    
    return best_model, best_params, best_score

def train_model(X_features, y, model_type='random_forest', cv=5, verbose_plot=False, verbose_print=False, random_state=26):
    print(f"Training {model_type} model with cross-validation...")

    X_features = np.array(X_features)
    y = np.array(y)

    if not np.issubdtype(y.dtype, np.number):
        unique_labels = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])
    else:
        unique_labels = np.unique(y)

    clf_configs = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=100, random_state=random_state),
        'svm_rbf': SVC(kernel='rbf', probability=True, random_state=random_state, C=10, gamma=0.1, class_weight='balanced'),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'decision_tree': DecisionTreeClassifier(random_state=random_state),
        'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=random_state)
    }

    if model_type not in clf_configs:
        raise ValueError(f"Invalid model_type: {model_type}")

    clf = clf_configs[model_type]
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Get accuracy scores
    cv_scores = cross_val_score(clf, X_features, y, cv=kf, scoring='accuracy', n_jobs=-1)
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
    if verbose_print:
        print(f"Cross-validation scores: {cv_scores}")

    if verbose_print:
        print("\nClassification Report:")
        print(classification_report(y, y_pred_cv))

    # Train full model on all data
    clf.fit(X_features, y)
    joblib.dump(clf, f'models/{model_type}_cv_trained.pkl')

    if verbose_plot:
        # 1. Generate CV predictions for Confusion Matrix
        y_pred_cv = cross_val_predict(clf, X_features, y, cv=kf, n_jobs=-1)
        cm = confusion_matrix(y, y_pred_cv)

        # 2. Learning Curve
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X_features, y, cv=kf, scoring='accuracy', n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # 3. Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # 4. Plot Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axs[0], cmap='Blues', colorbar=False)
        axs[0].set_title(f"Confusion Matrix ({model_type})")

        # 5. Plot Learning Curve
        axs[1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        axs[1].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
        axs[1].plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
        axs[1].plot(train_sizes, test_mean, 'o-', color="orange", label="Cross-validation score")
        axs[1].set_title("Learning Curve")
        axs[1].set_xlabel("Training Examples")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend(loc="best")
        axs[1].grid(True)

        # 6. Show combined plot
        plt.tight_layout()
        plt.show()

    return np.mean(cv_scores)

    print(f"Training {model_type} model with cross-validation...")

    X_features = np.array(X_features)
    y = np.array(y)

    if not np.issubdtype(y.dtype, np.number):
        unique_labels = np.unique(y)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])
    else:
        unique_labels = np.unique(y)

    clf_configs = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=100, random_state=random_state),
        'svm_rbf': SVC(kernel='rbf', probability=True, random_state=random_state, C=10, gamma=0.1, class_weight='balanced'),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'decision_tree': DecisionTreeClassifier(random_state=random_state),
        'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=random_state)
    }

    if model_type not in clf_configs:
        raise ValueError(f"Invalid model_type: {model_type}")

    clf = clf_configs[model_type]

    # Cross-validation accuracy
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X_features, y, cv=kf, scoring='accuracy', n_jobs=-1)

    if verbose_print:
        print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")

    clf.fit(X_features, y)
    joblib.dump(clf, f'models/{model_type}_cv_trained.pkl')

    if verbose_plot:
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X_features, y, cv=kf, scoring='accuracy', n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
        plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="orange", label="Cross-validation score")
        plt.title("Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()
        
        

    return np.mean(cv_scores)