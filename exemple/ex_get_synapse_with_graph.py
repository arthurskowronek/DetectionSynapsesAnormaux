import os
import skan
import numpy as np
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import skimage as ski
from skimage import io, filters, feature
from skimage.filters import frangi, apply_hysteresis_threshold
from skimage.measure import label
from skimage.draw import line
from skimage.morphology import remove_small_holes, binary_closing, disk
from scipy.spatial.distance import cdist

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

def skeleton_keep_main_branch(skel, maxima_coords, skeletonize = False, keep=1):
    G = skeleton_to_graph(skel)
    endpoints = find_endpoints(G)
    
    # show endpoints
    plt.figure(figsize=(8, 8))
    plt.imshow(skel, cmap='gray')
    for y, x in endpoints:
        plt.scatter(x, y, color='red', s=10)
    plt.title("Endpoints")
    plt.show()

    all_paths_dict = {}

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            try:
                path = nx.shortest_path(G, endpoints[i], endpoints[j])
                
                # Count how many points in the path are also in maxima_coords
                maxima_count = 0
                path_points = set(path)  # Convert to set for faster lookups
                
                # If maxima_coords is a numpy array of coordinates like [(y1,x1), (y2,x2), ...]
                for point in maxima_coords:
                    if tuple(point) in path_points:
                        maxima_count += 1
                
                # Create a unique key for each path
                path_key = f"path_{endpoints[i]}_{endpoints[j]}"
                
                # Store the path, node count, and maxima count
                all_paths_dict[path_key] = {
                    "path": path,
                    "maxima_count": maxima_count,
                    "length": len(path)
                }
            except nx.NetworkXNoPath:
                continue

    # Sort paths by node count, descending
    if skeletonize == False:
        sorted_paths = sorted(all_paths_dict.items(), key=lambda x: x[1]["maxima_count"], reverse=True)
    else: # on veut le plus grand squelette, pas celui qui traverse le plus de maxima
        sorted_paths = sorted(all_paths_dict.items(), key=lambda x: x[1]["length"], reverse=True)

    # Show all paths and print their length
    for i, (path_key, path_data) in enumerate(sorted_paths):
        print(f"Path {i}: Length = {path_data['maxima_count']}")
        plt.figure(figsize=(8, 8))
        plt.imshow(skel, cmap='gray')
        for y, x in path_data["path"]:
            plt.scatter(x, y, color='red', s=1)
        plt.title(f"Path {i}")
        plt.show()

    selected_paths = []
    if keep == 1:
        if sorted_paths:
            selected_paths = [sorted_paths[0][1]["path"]]
    elif keep == 2:
        for _, path_data in sorted_paths:
            path = path_data["path"]
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

def visualize_results(img, frangi_response, mask, local_max, filtered_maxima):
    """
    Visualize the results of the synapse segmentation.
    
    Parameters:
    -----------
    img : ndarray
        Original image
    frangi_response : ndarray
        Frangi filter response
    mask : ndarray
        Binary mask from thresholded Frangi response
    local_max : ndarray
        All detected local maxima
    filtered_maxima : ndarray
        Filtered maxima representing synapses
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Frangi response
    axes[1].imshow(frangi_response, cmap='viridis')
    axes[1].set_title('Frangi Filter Response')
    axes[1].axis('off')
    
    # Mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Binary Mask')
    axes[2].axis('off')
    
    # Local maxima
    axes[3].imshow(img, cmap='gray')
    axes[3].plot(local_max[:, 1], local_max[:, 0], 'r.', markersize=1)
    axes[3].set_title(f'All Local Maxima ({len(local_max)})')
    axes[3].axis('off')
    
    # Filtered maxima
    axes[4].imshow(img, cmap='gray')
    if len(filtered_maxima) > 0:
        axes[4].plot(filtered_maxima[:, 1], filtered_maxima[:, 0], 'g.', markersize=1)
    axes[4].set_title(f'Filtered Maxima ({len(filtered_maxima)})')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.show()

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

def preprocessing(image_path, threshold_percentile=95):
    """
    Segment synapses in C. elegans fluorescent images.
    
    Parameters:
    -----------
    image_path : str
        Path to the fluorescent image
    frangi_scale_range : tuple
        Range of scales for Frangi filter (min, max)
    frangi_scale_step : float
        Step size for scales in Frangi filter
    threshold_percentile : int
        Percentile for thresholding Frangi response
    connectivity_angle_threshold : float
        Maximum angle deviation for considering two maxima connected (in degrees)
    max_distance : int
        Maximum distance between connected maxima
    min_connections : int
        Minimum number of connections for a maximum to be considered part of a synapse cord
    
    Returns:
    --------
    tuple
        (original image, frangi filter response, mask, detected maxima, filtered maxima)
    """
    # Load the image
    img = io.imread(image_path)
    if len(img.shape) > 2:
        img = img[:,:,0]  # Take first channel if it's RGB
    
    # show image
    """plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.show()"""
    
    # Apply Frangi filter to enhance tubular structures (synapses)
    frangi_response = frangi(img ,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    # show image
    """plt.imshow(frangi_response, cmap='gray')
    plt.title('Frangi Filter Response')
    plt.show()"""
    frangi_response = apply_hysteresis_threshold(frangi_response, 0.01, 0.2)
    """plt.imshow(frangi_response, cmap='gray')
    plt.title('Hysteresis Thresholding')
    plt.show()"""
    
    frangi_response = remove_small_objects(frangi_response, option=2, min_size_value=25)
    """plt.imshow(frangi_response, cmap='gray')
    plt.title('Remove Small Objects')
    plt.show()"""
    
    # keep only components that are more like a line than a blob
    labeled_image = ski.measure.label(frangi_response)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        # if components is more like a line than a blob, keep it
        if component.major_axis_length/component.minor_axis_length > 4:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    frangi_response = label_components
    
    
    # plot
    """plt.imshow(frangi_response, cmap='gray')
    plt.title('Filtered 4/1')
    plt.show()"""
    
    # Normalize Frangi response to 0-1
    frangi_response = (frangi_response - frangi_response.min()) / (frangi_response.max() - frangi_response.min())
    
    # Create mask from Frangi response
    threshold = np.percentile(frangi_response, threshold_percentile)
    mask = frangi_response > threshold
    
    # show mask
    """plt.imshow(mask, cmap='gray')
    plt.title('Binary Mask')
    plt.show()"""
    
    # Apply mask to the original image
    masked_img = img.copy()
    masked_img[~mask] = 0

    # Find local maxima in the masked image
    local_max = feature.peak_local_max(masked_img, 
                                    min_distance=5, # minimum distance between maxima
                                    threshold_abs=0, # absolute threshold, means that we only consider maxima above this value
                                    exclude_border=False)
    
    # show local maxima on image
    plt.imshow(img, cmap='gray')
    plt.plot(local_max[:, 1], local_max[:, 0], 'r.', markersize=1)
    plt.title(f'Local Maxima ({len(local_max)})')
    plt.show()
    
    print("Number of local maxima detected:", len(local_max))
    print("Filtering maxima based on connectivity...")
    
    # Analyze connectivity of maxima based on direction
    filtered_maxima = local_max.copy()
    """filtered_maxima = filter_maxima_by_connectivity(local_max, 
                                                  connectivity_angle_threshold, 
                                                  max_distance, 
                                                  min_connections)"""
    
    return img, frangi_response, mask, local_max, filtered_maxima

def get_synapses_graph(worm_mask, maxima_coords):
    
    NUMBER_OF_SEGMENTS = 15
    
    # 1. Skeletonize the worm
    skeleton = ski.morphology.skeletonize(worm_mask)
    
    G, skeleton = skeleton_keep_main_branch(skeleton, maxima_coords, skeletonize = True, keep=1)
    
    # plot
    """plt.figure(figsize=(8, 8))
    plt.imshow(skeleton, cmap='gray')
    plt.title("Skeleton of Worm (Main Branch)")
    plt.show()"""
    
    # 2. Get ordered skeleton coordinates
    skel_path = order_skeleton_points_skan(skeleton)

    # 3. Divide skeleton into N segments
    n = len(skel_path)
    seg_len = n // NUMBER_OF_SEGMENTS
    centers = [skel_path[i * seg_len + seg_len // 2] for i in range(NUMBER_OF_SEGMENTS)]

    # 4. KMeans classification of maxima by center points
    #kmeans = KMeans(n_clusters=NUMBER_OF_SEGMENTS, init=np.array(centers), n_init=1, max_iter=1)
    #labels = kmeans.fit_predict(maxima_coords)
    # Compute distance between every maxima and every center
    distances = cdist(maxima_coords, centers)
    # Assign each point to the nearest center
    labels = np.argmin(distances, axis=1)

    # 5. Plot maxima colored by segment and the centers
    plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    # Create a colormap or color list for NUMBER_OF_SEGMENTS segments
    colors = plt.cm.get_cmap('hsv', NUMBER_OF_SEGMENTS)
    for i in range(NUMBER_OF_SEGMENTS):
        segment_points = maxima_coords[labels == i]
        # Pick a color for this segment
        color = colors(i)
        # Plot segment maxima with that color
        plt.scatter(segment_points[:, 1], segment_points[:, 0], label=f'Segment {i}', color=color)
        # Plot center in the same color but with a different marker
        plt.scatter(centers[i][1], centers[i][0], marker='x', s=100, color=color)
    plt.title("Maxima Segmentation")
    plt.legend()
    plt.show()

    # 6. Estimate direction of each segment
    directions = []
    for i in range(NUMBER_OF_SEGMENTS):
        start = np.array(skel_path[i * seg_len])
        end = np.array(skel_path[min((i + 1) * seg_len - 1, n - 1)])
        vec = end - start
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        directions.append(vec)
        
    # show on the image directions of the segments
    plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    for i in range(NUMBER_OF_SEGMENTS):
        start = skel_path[i * seg_len]
        end = skel_path[min((i+1) * seg_len - 1, n - 1)]
        plt.arrow(start[1], start[0], (end[1] - start[1]), (end[0] - start[0]), 
                  head_width=2, head_length=5, fc='red', ec='red')
    plt.title("Segment Directions")
    plt.show()
    
    # put a black pixel
    worm_mask[0, :] = 0  # Top row
    worm_mask[-1, :] = 0 # Bottom row
    worm_mask[:, 0] = 0  # Left column
    worm_mask[:, -1] = 0 # Right column
                
    
    # find coordinate of the middle of each segment
    middle_coords = [skel_path[i * seg_len + seg_len // 2] for i in range(NUMBER_OF_SEGMENTS)]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    dic_segments = {}
    for i in range(NUMBER_OF_SEGMENTS):
        start = middle_coords[i]
        end = skel_path[min((i + 1) * seg_len - 1, len(skel_path) - 1)]

        # Vector between start and end point
        vec = np.array(end) - np.array(start)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm  # Normalize the vector

        # Perpendicular vector
        perp_vec = np.array([-vec[1], vec[0]])

        # Compute long lines in both directions
        end_pos = start + perp_vec * 1000
        end_neg = start - perp_vec * 1000

        end_pos = np.round(end_pos).astype(int)
        end_neg = np.round(end_neg).astype(int)

        # Find intersection with the mask in positive direction
        rr_pos, cc_pos = line(start[0], start[1], end_pos[0], end_pos[1])
        for j in range(len(rr_pos)):
            if 0 <= rr_pos[j] < worm_mask.shape[0] and 0 <= cc_pos[j] < worm_mask.shape[1]:
                if worm_mask[rr_pos[j], cc_pos[j]] == 0:
                    end_pos = (rr_pos[j], cc_pos[j])
                    break

        # Find intersection with the mask in negative direction
        rr_neg, cc_neg = line(start[0], start[1], end_neg[0], end_neg[1])
        for j in range(len(rr_neg)):
            if 0 <= rr_neg[j] < worm_mask.shape[0] and 0 <= cc_neg[j] < worm_mask.shape[1]:
                if worm_mask[rr_neg[j], cc_neg[j]] == 0:
                    end_neg = (rr_neg[j], cc_neg[j])
                    break
                
        # if > 1024, we take the end of the image
        if end_pos[0] > 1024:
            end_pos = (1024, end_pos[1])
        if end_pos[1] > 1024:
            end_pos = (end_pos[0], 1024)
        if end_neg[0] > 1024:
            end_neg = (1024, end_neg[1])
        if end_neg[1] > 1024:
            end_neg = (end_neg[0], 1024)
        # if < 0, we take the beginning of the image
        if end_pos[0] < 0:
            end_pos = (0, end_pos[1])
        if end_pos[1] < 0:
            end_pos = (end_pos[0], 0)
        if end_neg[0] < 0:
            end_neg = (0, end_neg[1])
        if end_neg[1] < 0:
            end_neg = (end_neg[0], 0)
            
        # divide the segment end_pos to start and end_neg to start in 2 parts
        mid_pos = (start[0] + end_pos[0]) // 2, (start[1] + end_pos[1]) // 2
        mid_neg = (start[0] + end_neg[0]) // 2, (start[1] + end_neg[1]) // 2
        
        length_mid_pos = np.linalg.norm(np.array(mid_pos) - np.array(start)) # length of the segment from start to mid_pos
        length_mid_neg = np.linalg.norm(np.array(mid_neg) - np.array(start)) # length of the segment from start to mid_neg
        length_end_pos = np.linalg.norm(np.array(end_pos) - np.array(start)) # length of the segment from start to end_pos
        length_end_neg = np.linalg.norm(np.array(end_neg) - np.array(start)) # length of the segment from start to end_neg
            
        dic_segments[i] = (start, mid_pos, mid_neg, end_pos, end_neg, length_mid_pos, length_mid_neg, length_end_pos, length_end_neg)

        # Plot the full perpendicular line
        plt.plot([end_neg[1], end_pos[1]], [end_neg[0], end_pos[0]], 'g-', label=f'Segment {i}' if i == 0 else "")
        

    # plot lines between each end_pos points
    for i in range(NUMBER_OF_SEGMENTS-1):
        start = dic_segments[i][0]
        end = dic_segments[i+1][0]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
        start = dic_segments[i][1]
        end = dic_segments[i+1][1]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
        start = dic_segments[i][2]
        end = dic_segments[i+1][2]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
        start = dic_segments[i][3]
        end = dic_segments[i+1][3]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
        start = dic_segments[i][4]
        end = dic_segments[i+1][4]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
        
    
    plt.title("Segment Directions with Perpendicular Lines (Both Directions)")
    plt.legend()
    plt.show()
    

    
    # Assign each maxima to the slice it belongs to
    labels_slice = np.zeros(len(maxima_coords), dtype=int)
    for i, p1 in enumerate(maxima_coords):
        seg_idx1 = labels[i]
        seg = skel_path[seg_idx1 * seg_len: (seg_idx1 + 1) * seg_len]
        # distance to the segment
        dist_to_seg, signe = signed_distance_to_segment_2d(seg, p1)
        # if the point is in the positive direction of the segment, we compare dist_to_seg to dic_segments[seg_idx1][5] and dic_segments[seg_idx1][7]
        if signe > 0:
            if dist_to_seg < dic_segments[seg_idx1][5] and dist_to_seg < dic_segments[seg_idx1][7]:
                labels_slice[i] = 2
            elif dist_to_seg > dic_segments[seg_idx1][5] and dist_to_seg < dic_segments[seg_idx1][7]:
                labels_slice[i] = 1
            elif dist_to_seg > dic_segments[seg_idx1][5] and dist_to_seg > dic_segments[seg_idx1][7]:
                labels_slice[i] = 0
        else:
            if dist_to_seg < dic_segments[seg_idx1][6] and dist_to_seg < dic_segments[seg_idx1][8]:
                labels_slice[i] = 3
            elif dist_to_seg > dic_segments[seg_idx1][6] and dist_to_seg < dic_segments[seg_idx1][8]:
                labels_slice[i] = 4
            elif dist_to_seg > dic_segments[seg_idx1][6] and dist_to_seg > dic_segments[seg_idx1][8]:
                labels_slice[i] = 5
                        
        """# plot the point and the segment
        seg = np.array(seg)
        plt.figure(figsize=(5, 5))
        plt.imshow(worm_mask, cmap='gray')
        plt.plot(seg[:, 1], seg[:, 0], 'b-', label='Segment')
        plt.plot(p1[1], p1[0], 'ro', label='Maxima')
        plt.title(f"i = {i} | Assigned slice = {labels_slice[i]}")
        plt.legend()
        plt.show()"""

    # print numbers of points in each slice
    print("Number of points in each slice:")
    for i in range(6):
        print(f"Slice {i}: {np.sum(labels_slice == i)}")
    
    
    # Decide number of cords based on the number of maxima in each slice
    if np.sum(np.isin(labels_slice, [0, 1])) > len(maxima_coords) / 4 and np.sum(np.isin(labels_slice, [4, 5])) > len(maxima_coords) / 4:
        NUMBER_OF_CORDS = 2
    else:
        NUMBER_OF_CORDS = 1
    
    print("Number of cords:", NUMBER_OF_CORDS)

    # 6. Plot maxima with their assigned slice
    plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    # Map node positions
    pos = {i: (x[1], x[0]) for i, x in enumerate(maxima_coords)}
    # Create a colormap for the 6 label categories
    cmap = cm.get_cmap('tab10')  # tab10 is good for up to 10 categories
    norm = mcolors.Normalize(vmin=0, vmax=5)
    # Map each label to a color
    node_colors = [cmap(norm(label)) for label in labels_slice]
    plt.scatter([x[1] for x in maxima_coords], [x[0] for x in maxima_coords],
                c=node_colors, s=10, alpha=1)
    plt.title("Maxima with Assigned Slices")
    plt.show()
    

               
    # 7. Create graph based on directionality
    G = nx.Graph()
    for i, p1 in enumerate(maxima_coords):
        seg_idx1 = labels[i]
        slices1 = labels_slice[i]
        dir_vec = directions[seg_idx1]
        

        best_j = None
        min_dist = np.inf

        for j, p2 in enumerate(maxima_coords):
            if i == j:
                continue

            if abs(slices1 - labels_slice[j]) > 1:
                continue
            
            vec = p2 - p1
            dist = np.linalg.norm(vec)
        
            if dist == 0:
                continue

            vec_normed = vec / dist
            dot = np.dot(vec_normed, dir_vec)

            """if abs(slices1 - labels_slice[j]) != 0: # different slice
                dist = dist * 2 # penalize the distance if the points are in different slices"""
                
            if dot > 0.9 and dist < min_dist:
                best_j = j
                min_dist = dist

        if best_j is not None:
            G.add_edge(i, best_j)
        else:
            G.add_node(i)
           
    """# for points that have more than 2 connections, keep only the 2 connections that are the most aligned with the direction of the segment
    for i in list(G.nodes):  # safer than range(len(...))
        if G.degree[i] > 2:
            neighbors = list(G.neighbors(i))
            best_neighbors = sorted(
                neighbors,
                key=lambda x: np.dot(directions[labels[x]], directions[labels[i]]),
                reverse=True
            )[:2]
            for j in neighbors:
                if j not in best_neighbors:
                    G.remove_edge(i, j)

    # remove isolated nodes
    for i in list(G.nodes):  # safer than range(...)
        if G.degree[i] == 0:
            G.remove_node(i)"""

    print("Number of nodes in the graph:", len(G.nodes))
    print("Lenght of maxima_coords:", len(maxima_coords))
    # 8. Plot graph
    plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    pos = {i: (x[1], x[0]) for i, x in enumerate(maxima_coords)}
    nx.draw(G, pos, node_size=1, node_color='red', edge_color='blue')
    plt.title("Directional Graph of Maxima")
    plt.show()
    
    # 9. Convert graph to skeleton
    node_to_coord = {i: tuple(coord) for i, coord in enumerate(maxima_coords)}
    skeleton = graph_to_skeleton(G, node_to_coord, shape=worm_mask.shape)
    
    # plot skeleton
    plt.figure(figsize=(8, 8))
    plt.imshow(skeleton, cmap='gray')
    plt.title("Skeleton from Graph")
    plt.show()
    
    # 10. Keep only the NUMBER_OF_CORDS main branches of the skeleton
    G, skeleton = skeleton_keep_main_branch(skeleton, maxima_coords, skeletonize = False, keep=NUMBER_OF_CORDS)
    
    # 11. Plot the skeleton
    plt.figure(figsize=(8, 8))
    plt.imshow(skeleton, cmap='gray')
    plt.title(f"Skeleton of Worm ({NUMBER_OF_CORDS} Branches)")
    plt.show()    
    
    # keep only nodes that are in skeleton
    nodes = np.array([node for node in maxima_coords if skeleton[node[0], node[1]] == 1])
        
    print("Number of nodes in the final graph:", len(nodes))
    # plot image with maxima_filtered
    plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    plt.scatter(nodes[:, 1], nodes[:, 0], s=1, color='red')
    #for i in range(len(nodes)):
        #plt.scatter(nodes[i][1], nodes[i][0], s=1, color='red')
    plt.title("Maxima Coordinates")
    plt.show()

    return centers, labels, directions, G


# Example usage
if __name__ == "__main__":
    # image_path is a random image of the directory "data/WildType 2023_12_22"
    path_directory = "data/WildType 2023_12_22"
    path_directory = "data/Mut0 2023_12_22"
    list_of_images = os.listdir(path_directory)
    
    for image in list_of_images:        
        image = "EN6009-12_MMStack.ome.tif"
        image_path = os.path.join(path_directory, image)
        print(f"---------- Processing {image_path} ----------")
    
        try:
            # Preprocessing
            img, frangi_response, mask, local_max, filtered_maxima = preprocessing(image_path,threshold_percentile=95)

            # Segmentation  
            worm_mask = worm_segmentation(img)

            # Get synapses graph
            centers, labels, directions, G = get_synapses_graph(worm_mask, filtered_maxima)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue