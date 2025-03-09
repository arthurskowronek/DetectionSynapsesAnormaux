from utils import *



if __name__ == "__main__":
    # Load 
    # 'dataset_2025-03-07_15-58-46.pkl'
    data = createDataset(True) # If true, don't keep the name
    
    X = np.array(data['data'])
    y = np.array(data['label'])
    
    # show details of X
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    