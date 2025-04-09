import os
import skan
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import skimage as ski
from skimage import io, filters
from skimage.measure import label
from skimage.draw import line
from skimage.morphology import remove_small_holes, binary_closing, disk

def signed_distance_to_segment_2d(seg, p1):
    seg = np.array(seg)  # Convert 'seg' to a NumPy array
    if seg.shape[0] < 2:
        # Handle cases where 'seg' doesn't define a segment
        if seg.shape[0] == 1:
            return np.linalg.norm(seg[0] - np.array(p1)), 0
        else:
            return np.inf, 0
    v = seg[-1] - seg[0]
    w = seg[0] - np.array(p1) # Ensure p1 is also a NumPy array for subtraction
    cross_product = v[0] * w[1] - v[1] * w[0]
    segment_length = np.linalg.norm(v)
    if segment_length == 0:
        return np.linalg.norm(w), 0
    distance = np.abs(cross_product) / segment_length
    return distance, np.sign(cross_product)

def skeleton_to_graph(skel):
    G = nx.Graph()
    coords = np.column_stack(np.where(skel))
    for y, x in coords:
        for dy in [-1, 0, 1]: 
            for dx in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                ny, nx_ = y + dy, x + dx # new y and new x : we take the 8 neighbors of the pixel
                if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]: # check if the new pixel is in the image
                    if skel[ny, nx_]: # check if the new pixel is a skeleton pixel
                        G.add_edge((y, x), (ny, nx_))
    return G

def graph_to_skeleton(G, node_to_coord, shape=None):
    if shape is None:
        ys, xs = zip(*node_to_coord.values())
        shape = (max(ys) + 1, max(xs) + 1)

    skel = np.zeros(shape, dtype=bool)
    for node in G.nodes:
        y, x = node_to_coord[node]
        skel[y, x] = True
        
    for edge in G.edges:
        y1, x1 = node_to_coord[edge[0]]
        y2, x2 = node_to_coord[edge[1]]
        rr, cc = line(y1, x1, y2, x2)
        skel[rr, cc] = True
        
    return skel

def find_endpoints(G):
    return [n for n in G.nodes if G.degree[n] == 1]

def get_longest_path(G):
    endpoints = find_endpoints(G)
    max_len = 0
    longest_path = []

    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            try:
                path = nx.shortest_path(G, endpoints[i], endpoints[j])
                if len(path) > max_len:
                    max_len = len(path)
                    longest_path = path
            except nx.NetworkXNoPath:
                continue
    return longest_path

def skeleton_keep_main_branch(skel, keep=1):
    G = skeleton_to_graph(skel)
    endpoints = find_endpoints(G)

    all_paths = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            try:
                path = nx.shortest_path(G, endpoints[i], endpoints[j])
                all_paths.append(path)
            except nx.NetworkXNoPath:
                continue

    # Sort by path length, descending
    all_paths.sort(key=len, reverse=True)

    selected_paths = []

    if keep == 1:
        if all_paths:
            selected_paths = [all_paths[0]]
    elif keep == 2:
        for path in all_paths:
            used_nodes = set(n for p in selected_paths for n in p)
            if not used_nodes.intersection(path):
                selected_paths.append(path)
            if len(selected_paths) == 2:
                break

    # Create new image
    main_branch = np.zeros_like(skel, dtype=bool)
    for path in selected_paths:
        for y, x in path:
            main_branch[y, x] = True

    return G, main_branch

def order_skeleton_points_skan(skeleton):
    # Create the Skeleton object
    skel_obj = skan.csr.Skeleton(skeleton)
    
    # Get the summary with branch information
    summary = skan.summarize(skel_obj, separator='-')
    
    # Create a flat list of all points from all paths
    all_points = []
    
    for i in range(len(summary)):
        # Get coordinates for each path
        path_coords = skel_obj.path_coordinates(i)
        # Add all points from this path to the flat list
        for coord in path_coords:
            all_points.append(tuple(coord))
    
    return all_points

def remove_small_objects(image, option=1, min_size_value=30):
    if option == 1:
        bool_img = image.astype(bool)
        temp_result = ski.morphology.remove_small_objects(bool_img, min_size=min_size_value)
        image = temp_result
        return image
    elif option == 2:
        # Label connected components and remove small objects (intestines likely being smaller than synapses)
        labeled_image = ski.measure.label(image) 
        regions = ski.measure.regionprops(labeled_image)
        # Filter based on region properties (e.g., area) to keep only large regions (synapse chain)
        large_regions = np.zeros_like(labeled_image)

        for region in regions:
            if region.area > min_size_value:
                large_regions[labeled_image == region.label] = 1       
        # The final processed image should only contain the chain-like synapse regions
        return large_regions.astype(np.uint16)
    else:
        print("Option not available")
        return image

def worm_segmentation(img):
    
    # Apply Gaussian filter to smooth the image
    img = ski.filters.meijering(img, sigmas=range(8, 14, 2), black_ridges=False) # quand on baisse le sigma max, on garde seulement les vaisseaux fins
    
    # plot
    """plt.imshow(img, cmap='gray')
    plt.title('Meijering Filter Response')
    plt.show()"""
    
    # Thresholding to create a binary mask
    threshold = filters.threshold_otsu(img)
    """print("Threshold value:", threshold)
    print("Mean value:", np.mean(img))"""
    binary_mask = img > np.mean(img)
    
    # plot
    """plt.imshow(binary_mask, cmap='gray')
    plt.title('Binary Mask')
    plt.show()"""
    
    # Remove small objects from the binary mask
    cleaned_mask = remove_small_objects(binary_mask, option=2, min_size_value=30)
    
    # transform it in boolean
    cleaned_mask = cleaned_mask.astype(bool)
    
    # Fill small holes inside the worm
    worm_mask = remove_small_holes(cleaned_mask, area_threshold=50)

    # plot 
    """plt.imshow(worm_mask, cmap='gray')
    plt.title('After removing small holes : Mask')
    plt.show()"""
    
    # Close small gaps in the worm mask
    worm_mask = binary_closing(worm_mask, disk(20))
    
    # keep the largest connected component (the worm)
    labeled_mask = label(worm_mask)
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1  # +1 to skip background label
    worm_mask = (labeled_mask == largest_component).astype(np.uint8)
    
    
    # plot cleaned mask
    """plt.imshow(worm_mask, cmap='gray')
    plt.title('Cleaned Mask')
    plt.show()"""
    
    
    return worm_mask

def get_synapses_graph(worm_mask):
    
    # 1. Skeletonize the worm
    skeleton = ski.morphology.skeletonize(worm_mask)
    
    G, skeleton = skeleton_keep_main_branch(skeleton, keep=1)

    # 2. Get ordered skeleton coordinates
    skel_path = order_skeleton_points_skan(skeleton)
    
    return skel_path


# Example usage
if __name__ == "__main__":
    
    length_skeleton = []
    
    # image_path is a random image of the directory "data/WildType 2023_12_22"
    path_directory = "data/WildType 2023_12_22"
    #path_directory = "data/Mut0 2023_12_22"
    list_of_images = os.listdir(path_directory)
    for image in list_of_images:
        image_path = os.path.join(path_directory, image)
        print(f"---------- Processing {image_path} ----------")
        
        img = io.imread(image_path)
        
        # Segmentation
        try: 
            worm_mask = worm_segmentation(img)

            # Get synapses graph
            skeleton = get_synapses_graph(worm_mask)
            
            # Get length of the skeleton
            length_skeleton.append(len(skeleton))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
        
    # show distribution of the length of the skeleton
    plt.hist(length_skeleton, bins=50)
    plt.xlabel('Length of the skeleton')
    plt.ylabel('Frequency')
    plt.title('Distribution of the length of the skeleton')
    plt.show()
    
    # show the mean, max, min and std of the length of the skeleton
    mean_length = np.mean(length_skeleton)
    std_length = np.std(length_skeleton)
    min_length = np.min(length_skeleton)
    max_length = np.max(length_skeleton)
    print(f"Mean length of the skeleton: {mean_length}")
    print(f"Std length of the skeleton: {std_length}")
    print(f"Min length of the skeleton: {min_length}")
    print(f"Max length of the skeleton: {max_length}")
        
        