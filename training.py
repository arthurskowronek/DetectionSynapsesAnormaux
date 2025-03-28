import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from numpy.random import RandomState


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

def train_model(X_features, y, verbose_plot = False, model_type='random_forest', n_runs=10, test_size=0.3, random_state=26):
    """
    Train a model on the feature vectors with support for multiple classifier types.
    
    Parameters:
    -----------
    X_features : array-like
        Feature vectors for training
    y : array-like
        Target labels
    model_type : str, default='random_forest'
        Type of model to train. Options are:
        - 'hist_gradient_boosting'
        - 'svm_rbf'
        - 'light_gbm'
        - 'knn'
        - 'decision_tree'
        - 'mlp'
        - 'random_forest'
        - 'siamese_network'
    n_runs : int, default=10
        Number of training runs to perform
    test_size : float, default=0.3
        Proportion of data to use for testing
    random_state : int or RandomState, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    float
        Mean accuracy across all runs
    """
    print(f"Training {model_type} model...")
    
    # Convert to numpy arrays if they're not already
    X_features = np.array(X_features)
    y = np.array(y)
    
    # Convert labels to numeric
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])
    
    # Initialize results and seed
    correct_estimations = []
    learning_curves = {
        'train_sizes': [],
        'train_scores': [],
        'test_scores': []
    }
    seed = RandomState(random_state) if isinstance(random_state, int) else random_state
    
    # Define model configurations
    model_configs = {
        'hist_gradient_boosting': {
            'cls': HistGradientBoostingClassifier,
            'params': {'max_iter': 100, 'random_state': seed}
        },
        'svm_rbf': {
            'cls': SVC,
            'params': {'kernel': 'rbf', 'probability': True, 'random_state': seed}
        },
        'knn': {
            'cls': KNeighborsClassifier,
            'params': {'n_neighbors': 5}
        },
        'decision_tree': {
            'cls': DecisionTreeClassifier,
            'params': {'random_state': seed}
        },
        'mlp': {
            'cls': MLPClassifier,
            'params': {'hidden_layer_sizes': (100, 50), 'max_iter': 300, 'random_state': seed}
        },
        'random_forest': {
            'cls': RandomForestClassifier,
            'params': {'n_estimators': 10, 'random_state': seed}
        },
        'siamese_network': {
            'cls': SiameseClassifier,
            'params': {'input_dim': X_features.shape[1], 'n_epochs': 10}
        }
    }
    
    # Validate model_type
    if model_type not in model_configs:
        raise ValueError(f"Invalid model_type: {model_type}. Options are: {', '.join(model_configs.keys())}")
    
    # Run multiple training iterations
    for run in range(n_runs):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_numeric, test_size=test_size, random_state=seed
        )
        
        # Initialize model
        model_config = model_configs[model_type]
        clf = model_config['cls'](**model_config['params'])
        
        # Compute learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X_train, y_train, 
            train_sizes=np.linspace(0.1, 1.0, 5),  # 5 different training set sizes
            cv=5,  # 5-fold cross-validation
            scoring='accuracy'
        )
        
        # Store learning curve data
        learning_curves['train_sizes'].append(train_sizes)
        learning_curves['train_scores'].append(train_scores)
        learning_curves['test_scores'].append(test_scores)
        
        # Train model
        clf.fit(X_train, y_train)    
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        correct_estimations.append(accuracy)
        
        # Show distribution of confidence in predictions
        y_pred_proba = clf.predict_proba(X_test) if model_type != 'siamese_network' else None        
        
        
        # Update seed for next iteration
        seed = RandomState(seed.randint(0, 2**32 - 1))
        
        # Print progress
        if (run + 1) % 5 == 0 or run == 0:
            print(f"Completed {run + 1}/{n_runs} runs. Current mean accuracy: {np.mean(correct_estimations):.4f}")
            
        if verbose_plot :
            # if this is the last run, print the confidence histograms
            if run == n_runs - 1:
                plot_probability_histograms(y_pred_proba, class_names=['Mutant', 'Wildtype'])
                plot_probability_boxplots(y_pred_proba, class_names=['Mutant', 'Wildtype'])
                plot_probability_scatter(y_pred_proba, y_test, y_pred)
                
                # Learning Curve Visualization
                plt.figure(figsize=(10, 6))
                
                # Average train and test scores across runs
                avg_train_scores = np.mean(learning_curves['train_scores'], axis=0)
                avg_test_scores = np.mean(learning_curves['test_scores'], axis=0)
                
                # Standard deviation of train and test scores
                std_train_scores = np.std(learning_curves['train_scores'], axis=0)
                std_test_scores = np.std(learning_curves['test_scores'], axis=0)
                
                avg_train_sizes = np.mean(learning_curves['train_sizes'], axis=0)
                
                # Plot learning curves with error bands
                plt.plot(avg_train_sizes, avg_train_scores.mean(axis=1), 
                         label='Training Score', color='blue', marker='o')
                plt.fill_between(avg_train_sizes, 
                                 avg_train_scores.mean(axis=1) - std_train_scores.mean(axis=1),
                                 avg_train_scores.mean(axis=1) + std_train_scores.mean(axis=1), 
                                 alpha=0.15, color='blue')
                
                plt.plot(avg_train_sizes, avg_test_scores.mean(axis=1), 
                         label='Validation Score', color='red', marker='s')
                plt.fill_between(avg_train_sizes, 
                                 avg_test_scores.mean(axis=1) - std_test_scores.mean(axis=1),
                                 avg_test_scores.mean(axis=1) + std_test_scores.mean(axis=1), 
                                 alpha=0.15, color='red')
                
                plt.title(f'Learning Curves - {model_type.replace("_", " ").title()}')
                plt.xlabel('Training Set Size')
                plt.ylabel('Accuracy')
                plt.legend(loc='lower right')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            
    # save model
    if model_type == 'siamese_network':
        torch.save(clf.model.state_dict(), 'models/model.pth')
    else:
        joblib.dump(clf, 'models/model.pkl')
        
    """import shap 
    model = clf
    explainer = shap.TreeExplainer(model) 
    shap_values = explainer.shap_values(X_features) 
    shap.summary_plot(shap_values, X_features)"""
            
    # Calculate and return mean accuracy
    mean_correct_estim = np.mean(correct_estimations)
    print(f"Final mean accuracy: {mean_correct_estim:.4f}")
    return mean_correct_estim


