import skan
import joblib
import shutil
import itertools
import numpy as np
import skimage as ski
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
from skimage.draw import line
from skimage.morphology import binary_closing, disk
from pathlib import Path

from constants import *


# Utils functions
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

# Image processing functions
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

def close_gap_between_edges(image, max_distance=5):
    # Morphological operation: Dilation followed by Erosion to close the chain gaps
    selem = ski.morphology.disk(max_distance)  # Use a disk-shaped structuring element
    dilated = ski.morphology.dilation(image, selem)
    closed = ski.morphology.erosion(dilated, selem)
    return closed

# graph functions
def find_endpoints(skel, G, maxima_coords, skeletonize, angle_threshold_degrees=90):
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    angle_junctions = []

    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        
        if len(neighbors) >= 2:
            coords = []
            if skeletonize:
                coords = [np.array(neighbor) for neighbor in neighbors]
                node_coords = np.array(node)
            else:
                coords = [np.array(maxima_coords[n]) for n in neighbors]
                node_coords = np.array(maxima_coords[node])

            all_angles_sharp = True

            for a, b in itertools.combinations(coords, 2):
                v1 = a - node_coords
                v2 = b - node_coords

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 == 0 or norm2 == 0:
                    continue

                cosine_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle_rad = np.arccos(cosine_angle)
                angle_deg = np.degrees(angle_rad)

                if angle_deg >= angle_threshold_degrees:
                    all_angles_sharp = False
                    break  # One wide angle is enough to skip the node

            if all_angles_sharp:
                angle_junctions.append(node if skeletonize else maxima_coords[node])

    return endpoints, angle_junctions

def skeleton_keep_main_branch(G, skel, maxima_coords, skeletonize=False, keep=1):
    if skeletonize:
        G = skeleton_to_graph(skel)

    endpoints, angle_junctions = find_endpoints(skel, G, maxima_coords, skeletonize, angle_threshold_degrees=90)

    if not skeletonize:
        endpoints = [maxima_coords[i] for i in endpoints]
    else:
        angle_junctions = [maxima_coords[i] for i in angle_junctions]

    # Plotting the skeleton with endpoints and junctions
    """plt.figure(figsize=(8, 8))
    plt.imshow(skel, cmap='gray')
    for y, x in endpoints:
        plt.scatter(x, y, color='red', s=10)
    for y, x in angle_junctions:
        plt.scatter(x, y, color='blue', s=10)
    plt.title("Endpoints and Junctions") 
    plt.show()"""

    # Initiate variables for pathfinding
    all_paths = {}
    endpoint_indices = list(range(len(endpoints)))
    angle_junction_set = set(map(tuple, angle_junctions))
    
    if len(endpoints) < 2:
        # create path with the skeleton
        if skeletonize:
            path = list(np.argwhere(skel))
            all_paths[endpoint_indices[0]] = {
                "path": path,
                "maxima_count": len(path),
                "length": len(path)
            }
        
    
    # Iterate over all pairs of endpoints to find all possible paths
    for i in endpoint_indices:
        for j in endpoint_indices[i + 1:]:
            try:
                s_idx, e_idx = (endpoints[i], endpoints[j]) if skeletonize else (
                    np.where((maxima_coords == endpoints[i]).all(axis=1))[0][0],
                    np.where((maxima_coords == endpoints[j]).all(axis=1))[0][0]
                )
                path = nx.shortest_path(G, s_idx, e_idx)

                if skeletonize:
                    maxima_set = set(map(tuple, maxima_coords))
                    maxima_count = sum(node in maxima_set for node in path)
                else:
                    maxima_count = sum(1 for node in path if node < len(maxima_coords))
                key = f"path_{s_idx}_{e_idx}"
                
                if skeletonize:
                    # if the path does not contain angle junctions, we keep it
                    if not any(tuple(node) in angle_junction_set for node in path):
                        all_paths[key] = {
                            "path": path,
                            "maxima_count": maxima_count,
                            "length": len(path)
                        }
                    else: # Divide the path into subpath around the angle junctions
                        list_of_nodes = list(path)
                        # get indices of start, end, and junctions in list_of_nodes
                        indices = [list_of_nodes.index(node) for node in path if tuple(node) in angle_junction_set]
                        indices.append(len(list_of_nodes)-1) # add the last index
                        indices = [0] + indices # add the first index
                        indices = sorted(indices) # sort the indices
                        
                        for start, end in zip(indices[:-1], indices[1:]):
                            subpath = list_of_nodes[start:end+1]
                            subpath_key = f"subpath_{start}_{end}"
                            subpath_maxima_count = sum(node in maxima_set for node in subpath)
                            all_paths[subpath_key] = {
                                "path": subpath,
                                "maxima_count": subpath_maxima_count,
                                "length": len(subpath)
                            }
                        
                else:
                    if not any(tuple(maxima_coords[node]) in angle_junction_set for node in path if node < len(maxima_coords)):
                        all_paths[key] = {
                            "path": path,
                            "maxima_count": maxima_count,
                            "length": len(path)
                        }
                    else:
                        list_of_nodes = list(path)
                        # get indices of start, end, and junctions in list_of_nodes
                        indices = [list_of_nodes.index(node) for node in path if tuple(maxima_coords[node]) in angle_junction_set]
                        indices.append(len(list_of_nodes)-1) # add the last index
                        indices = [0] + indices # add the first index
                        indices = sorted(indices) # sort the indices
    
                        
                        for start, end in zip(indices[:-1], indices[1:]):
                            subpath = list_of_nodes[start:end+1]
                            subpath_key = f"subpath_{start}_{end}"
                            subpath_maxima_count = sum(1 for node in subpath if node < len(maxima_coords))
                            all_paths[subpath_key] = {
                                "path": subpath,
                                "maxima_count": subpath_maxima_count,
                                "length": len(subpath)
                            }
                
            except nx.NetworkXNoPath:
                #print(f"No path between {s_idx} and {e_idx}")
                continue
            
  

    # Sorting paths
    sort_key = "length" if skeletonize else "maxima_count"
    sorted_paths = sorted(all_paths.values(), key=lambda x: x[sort_key], reverse=True)

    # Print sorted paths with their lengths or maxima counts
    """for i, data in enumerate(sorted_paths):
        print(f"Path {i}: {sort_key.capitalize()} = {data[sort_key]}")"""


    selected_paths = []
    used_nodes = set()

    for path_data in sorted_paths:
        path = path_data["path"]
        if keep == 1 and not selected_paths:
            selected_paths.append(path)
        elif keep == 2 and not used_nodes.intersection(path):
            selected_paths.append(path)
            used_nodes.update(path)
        if len(selected_paths) == keep:
            break

    # Convert node indices to coordinates if needed
    if not skeletonize:
        selected_paths = [[maxima_coords[pt] for pt in path] for path in selected_paths]

    # Plot selected paths
    """plt.figure(figsize=(8, 8))
    plt.imshow(skel, cmap='gray')
    for path in selected_paths:
        y, x = zip(*path)
        plt.plot(x, y, color='red', linewidth=2)
    plt.title("Selected Paths")
    plt.show()"""
    
    # Create binary image of the main branch
    flat_points = [pt for path in selected_paths for pt in path]
    
    main_branch = np.zeros_like(skel, dtype=bool)
    for y, x in flat_points:
        main_branch[y, x] = True
        
    # Show the main branch
    """plt.figure(figsize=(8, 8))
    plt.imshow(main_branch, cmap='gray')
    plt.title("Main Branch")
    plt.show()"""

    return G, main_branch

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

def preprocess_image_for_graph(img):

    threshold_percentile=95
    
    # Apply Frangi filter to enhance tubular structures (synapses)
    frangi_response = ski.filters.frangi(img ,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    # Apply hysteresis thresholding to the Frangi response
    frangi_response = ski.filters.apply_hysteresis_threshold(frangi_response, 0.01, 0.2)
    
    frangi_response = remove_small_objects(frangi_response, option=2, min_size_value=25)

    
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
    
    
    # Normalize Frangi response to 0-1
    frangi_response = (frangi_response - frangi_response.min()) / (frangi_response.max() - frangi_response.min())
    
    # Create mask from Frangi response
    threshold = np.percentile(frangi_response, threshold_percentile)
    mask = frangi_response > threshold
    
    
    # Apply mask to the original image
    masked_img = img.copy()
    masked_img[~mask] = 0

    # Find local maxima in the masked image
    local_max = ski.feature.peak_local_max(masked_img, 
                                    min_distance=5, # minimum distance between maxima
                                    threshold_abs=0, # absolute threshold, means that we only consider maxima above this value
                                    exclude_border=False)


    return img, local_max
    
def get_synapses_graph(worm_mask, maxima_coords):
    
    NUMBER_OF_SEGMENTS = 20
    
    # 1. Skeletonize the worm
    skeleton = ski.morphology.skeletonize(worm_mask)
    
    G, skeleton = skeleton_keep_main_branch(None, skeleton, maxima_coords, skeletonize = True, keep=1)
    
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
    """plt.figure(figsize=(8, 8))
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
    plt.show()"""

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
    """plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    for i in range(NUMBER_OF_SEGMENTS):
        start = skel_path[i * seg_len]
        end = skel_path[min((i+1) * seg_len - 1, n - 1)]
        plt.arrow(start[1], start[0], (end[1] - start[1]), (end[0] - start[0]), 
                  head_width=2, head_length=5, fc='red', ec='red')
    plt.title("Segment Directions")
    plt.show()"""
    
    # put a black pixel
    worm_mask[0, :] = 0  # Top row
    worm_mask[-1, :] = 0 # Bottom row
    worm_mask[:, 0] = 0  # Left column
    worm_mask[:, -1] = 0 # Right column
                
    
    # find coordinate of the middle of each segment
    middle_coords = [skel_path[i * seg_len + seg_len // 2] for i in range(NUMBER_OF_SEGMENTS)]
    
    #plt.figure(figsize=(8, 8))
    #plt.imshow(worm_mask, cmap='gray')
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
        length_total = np.linalg.norm(np.array(end_pos) - np.array(end_neg)) # length of the segment from end_pos to end_neg
            
        dic_segments[i] = (start, mid_pos, mid_neg, end_pos, end_neg, length_mid_pos, length_mid_neg, length_end_pos, length_end_neg, length_total)

        # Plot the full perpendicular line
        #plt.plot([end_neg[1], end_pos[1]], [end_neg[0], end_pos[0]], 'g-', label=f'Segment {i}' if i == 0 else "")
       
    # Calculate the median of the lengths of the perpendicular lines
    median_width = np.median([dic_segments[i][9] for i in range(NUMBER_OF_SEGMENTS)])

    # plot lines between each end_pos points
    """for i in range(NUMBER_OF_SEGMENTS-1):
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
    plt.show()"""
    

    
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
    """print("Number of points in each slice:")
    for i in range(6):
        print(f"Slice {i}: {np.sum(labels_slice == i)}")"""
    
    
    # Decide number of cords based on the number of maxima in each slice
    if np.sum(np.isin(labels_slice, [0, 1])) > len(maxima_coords) / 5 and np.sum(np.isin(labels_slice, [4, 5])) > len(maxima_coords) / 5:
        NUMBER_OF_CORDS = 2
    else:
        NUMBER_OF_CORDS = 1
    
    #print("Number of cords:", NUMBER_OF_CORDS)

    """# 6. Plot maxima with their assigned slice
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
    plt.show()"""
    

               
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
    

    """print("Number of nodes in the graph:", len(G.nodes))
    print("Lenght of maxima_coords:", len(maxima_coords))
    # 8. Plot graph
    plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    pos = {i: (x[1], x[0]) for i, x in enumerate(maxima_coords)}
    nx.draw(G, pos, node_size=1, node_color='red', edge_color='blue')
    plt.title("Directional Graph of Maxima")
    plt.show()"""
    
    # 9. Convert graph to skeleton
    node_to_coord = {i: tuple(coord) for i, coord in enumerate(maxima_coords)}
    skeleton = graph_to_skeleton(G, node_to_coord, shape=worm_mask.shape)
    
    # plot skeleton
    """plt.figure(figsize=(8, 8))
    plt.imshow(skeleton, cmap='gray')
    plt.title("Skeleton from Graph")
    plt.show()"""
    
    # 10. Keep only the NUMBER_OF_CORDS main branches of the skeleton
    G, skeleton = skeleton_keep_main_branch(G, skeleton, maxima_coords, skeletonize = False, keep=NUMBER_OF_CORDS)
    
    # 11. Plot the skeleton
    """plt.figure(figsize=(8, 8))
    plt.imshow(skeleton, cmap='gray')
    plt.title(f"Skeleton of Worm ({NUMBER_OF_CORDS} Branches)")
    plt.show()  """  
    
    # keep only nodes that are in skeleton
    maxima = np.array([node for node in maxima_coords if skeleton[node[0], node[1]] == 1])
        
    #print("Number of nodes in the final graph:", len(maxima))
    # plot image with maxima_filtered
    """plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    plt.scatter(maxima[:, 1], maxima[:, 0], s=1, color='red')
    #for i in range(len(nodes)):
        #plt.scatter(nodes[i][1], nodes[i][0], s=1, color='red')
    plt.title("Maxima Coordinates")
    plt.show()"""

    return maxima, G, median_width



def create_mask_synapse(image):
    
    image_copy = image.copy()
    
    #display_image(image)
    
    # ------ MASK 1 ------
    image = ski.filters.frangi(image_copy,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    image = ski.filters.apply_hysteresis_threshold(image, 0.01, 0.2)
    # Remove small objects
    image = remove_small_objects(image, option=2, min_size_value=25)
    # keep only components that are more like a line than a blob
    labeled_image = ski.measure.label(image)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        # if components is more like a line than a blob, keep it
        if component.major_axis_length/component.minor_axis_length > 4:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    mask1 = label_components
    
    
    # ------ MASK 2 ------
    """image = ski.filters.frangi(image_copy,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    image = ski.filters.apply_hysteresis_threshold(image, 0.02, 0.15)
    # get skeleton
    skeleton = ski.morphology.skeletonize(image)
    # display_image(skeleton)
    # keep only components of skeleton that are longer than 10 pixels 
    labeled_image = ski.measure.label(skeleton)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        if component.major_axis_length > 50:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    image = label_components  
    # dilate image
    selem = ski.morphology.disk(1)
    mask2 = ski.morphology.dilation(image, selem)"""
    
    
    image = mask1 #| mask2 # combine masks
    
    #display_image(image)
    
    # ----- ADJUST CONTRAST ----- 
    #image = anisotropic_diffusion(image) # remove noise and enhance edges
    #image = exposure.adjust_gamma(image, gamma=3) 
    #image = exposure.adjust_log(image, gain=2, inv=False) 
    #image = ski.exposure.equalize_hist(image) # not a good idea
        
        
    # ----- TUBNESS FILTERS -----
    # Meijering filter
    #meij_image = ski.filters.meijering(image, sigmas=range(1, 8, 2), black_ridges=False) # quand on baisse le sigma max, on garde seulement les vaisseaux fins
    # Sato filter
    #image_sato = ski.filters.sato(image, sigmas=range(1, 3, 1), black_ridges=False)
    # Hessian filter
    #image = ski.filters.hessian(image,black_ridges=False,sigmas=range(1, 5, 1), alpha=2, beta=0.5, gamma=15)
    # Franji filter
    #image = ski.filters.frangi(image,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=70)
    #image = ski.filters.frangi(image,black_ridges=False,sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=15)
        
        
    #display_image(0.9 * image + 0.1 * image_sato)
    #display_image(image)
    
    #image =  0.9 * image + 0.1 * meij_image
    
    #display_image(image)
        
    # gabors filter
    #real, imag = ski.filters.gabor(image, frequency=0.5)
    #image = real 
        
    # hysterisis thresholding
    #image = ski.filters.apply_hysteresis_threshold(image, 0.01, 0.2)
      
        
    # ----- DENOISE -----
    #image = ski.restoration.denoise_nl_means(image, h=0.7)
    #image = ski.restoration.denoise_bilateral(image)
    #image = ski.restoration.denoise_tv_chambolle(image, weight=0.1)
    #image = ski.restoration.denoise_bilateral(image)
        

    # ----- REMOVE SMALL OBJECTS -----
    #image = remove_small_objects(image, option=2, min_size_value=25)
        
    
    # keep only components that are more like a line than a blob
    """labeled_image = ski.measure.label(image)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        # if components is more like a line than a blob, keep it
        if component.major_axis_length/component.minor_axis_length > 4:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    image = label_components"""
        
        
    """# get skeleton
    skeleton = ski.morphology.skeletonize(image)

    #display_image(skeleton)
    
    # keep only components of skeleton that are longer than 10 pixels
    labeled_image = ski.measure.label(skeleton)
    components = ski.measure.regionprops(labeled_image)
    label_components = np.zeros_like(labeled_image)
    for component in components:
        if component.major_axis_length > 50:
            label_components[labeled_image == component.label] = 1
        else:
            label_components[labeled_image == component.label] = 0
    image = label_components"""
    
    #display_image(image)
        
    # ----- THRESHOLDING -----
    # threshold otsu
    #threshold_value = ski.filters.threshold_otsu(image)
    # threshold local
    #threshold_value = ski.filters.threshold_local(image, block_size=3)
    # threshold mean
    #threshold_value = ski.filters.threshold_mean(image)
    # threshold triangle
    #threshold_value = ski.filters.threshold_triangle(image)
    # threshold yen
    #threshold_value = ski.filters.threshold_yen(image)
    # threshold li
    #threshold_value = ski.filters.threshold_li(image)
    #fig, ax = ski.filters.try_all_threshold(image, figsize=(8, 5), verbose=True) 
    #plt.show()
    #image = image  > threshold_value
        
    # ----- EDGE DETECTION -----
    # canny edge detector
    #image = ski.feature.canny(image, sigma=1)
    # sobel filter - edge detection
    #image = ski.filters.sobel(image)
    # prewitt filter - edge detection
    #image = ski.filters.prewitt(image)
    # scharr filter
    #image = ski.filters.scharr(image)
    # roberts filter
    #image = ski.filters.roberts(image)
    # laplace filter
    #image = ski.filters.laplace(image, ksize=3) # doesn't work
   
    
    # Hough Transform to detect long edges
    #lines = ski.transform.probabilistic_hough_line(mask_synapses, threshold=10, line_length=5, line_gap=3)
    #for line in lines:
        #p0, p1 = line
        #mask_synapses[p0[0]:p1[0], p0[1]:p1[1]] = 1
        
    
    # dilate image
    selem = ski.morphology.disk(1)
    image = ski.morphology.dilation(image, selem)

    
    # ----- CLOSE GAP BETWEEN EDGES -----
    #image = close_gap_between_edges(image, max_distance=10)
    
    #display_image(image)
    
    return image
     
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

def get_high_intensity_pixels (mask, image):
    
    method = 1
    # ------------------------ METHOD 1 --------------------------------
    if method == 1:
        
        skeleton = ski.morphology.skeletonize(mask)
            
        ordered_skeleton_points = order_skeleton_points_skan(skeleton)
        intensities = []
        for x, y in ordered_skeleton_points:
            intensities.append(image[x, y])
            
        # smooth intensities
        smoothed_intensities = intensities
        #window_size = 1
        #smoothed_intensities = np.convolve(intensities, np.ones(window_size), 'valid') / window_size 
        
        # get the index of the local maxima. A maxima is a point where the intensity is greater than its neighbors (2 left and 2 right)
        maxima = []
        for i in range(2, len(smoothed_intensities)-2):
            if smoothed_intensities[i] > smoothed_intensities[i-1] and smoothed_intensities[i] > smoothed_intensities[i-2] and smoothed_intensities[i] > smoothed_intensities[i+1] and smoothed_intensities[i] > smoothed_intensities[i+2]:
                maxima.append(i)
        
        # plot the maxima
        """plt.plot(smoothed_intensities)
        plt.plot(maxima, [smoothed_intensities[i] for i in maxima], 'ro')
        plt.show() """
        
        # x is a vector from 1 to the length of the smoothed intensities
        x = np.arange(len(smoothed_intensities))
        
        # get the plot to the derive of the smoothed intensities
        derive = np.gradient(smoothed_intensities, x)

        # get pixel coordinates of the maxima
        maxima_coords = []
        for i in maxima:
            maxima_coords.append(ordered_skeleton_points[i])
            
            
        # plot the maxima on the image
        """for x, y in maxima_coords:
            for i in range(-1,1):
                for j in range(-1,1):
                    if x+i >= 0 and x+i < image.shape[0] and y+j >= 0 and y+j < image.shape[1]:
                        image[x+i, y+j] = 65535
            
        display_image(image)"""
        
        # complete smoothed intensities until have MAX_LENGTH_OF_FEATURES values
        while len(smoothed_intensities) < MAX_LENGTH_OF_FEATURES:
            smoothed_intensities.append(0)
    
        derive = list(derive)
        while len(derive) < MAX_LENGTH_OF_FEATURES:
            derive.append(0)

        # get the distance map
        """distance_map = scipy.ndimage.distance_transform_edt(mask)  
        # get the local maxima of the distance map
        def detect_local_maxima(image):
            # get the boolean mask of the local maxima
            peaks_mask = ski.feature.peak_local_max(image, min_distance=4, threshold_abs=0)
            # get the coordinates of the local maxima
            coords = np.transpose(np.nonzero(peaks_mask))
            return coords
        maxima_coords = detect_local_maxima(distance_map)"""
        
    # ------------------------ METHOD 2 --------------------------------
    else: # ne marche pas
        smoothed_intensities = []
        derive = []
        # complete smoothed intensities until have MAX_LENGTH_OF_FEATURES values
        while len(smoothed_intensities) < MAX_LENGTH_OF_FEATURES:
            smoothed_intensities.append(0)
        derive = list(derive)
        while len(derive) < MAX_LENGTH_OF_FEATURES:
            derive.append(0)

        
        # apply mask to original image
        image = image * mask
        
        # Step 1: Apply Hessian Matrix to Image
        hessian_elems = ski.feature.hessian_matrix(image, sigma=2, order='rc')
        hessian_eigenvals = ski.feature.hessian_matrix_eigvals(hessian_elems)
        
        # Step 2: Threshold negative eigenvalues
        eigenvalue1, eigenvalue2 = hessian_eigenvals[0], hessian_eigenvals[1]
        
        # Keep points where both eigenvalues are negative
        negative_eigenvalue_mask = (eigenvalue1 < 0) & (eigenvalue2 < 0)
        
        # Step 3: Find local maxima in the negative eigenvalue mask (intensity peaks)
        hessian_response = np.abs(eigenvalue1)  # or use the largest eigenvalue for intensity peaks
        local_maxima = ski.feature.peak_local_max(hessian_response, min_distance=3, threshold_abs=np.mean(hessian_response))
        
        # Step 4: Filter maxima based on the mask and thresholding condition
        maxima_coords = [(x, y) for x, y in local_maxima if mask[x, y] > 0 and negative_eigenvalue_mask[x, y]]


        # Plot original vs. Hessian response
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[1].imshow(hessian_response, cmap='inferno')
        axes[1].set_title('Hessian Response')
        plt.show()
        

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap='gray')
        ax.scatter([y for x, y in maxima_coords], [x for x, y in maxima_coords], 
                color='red', s=20, label="Detected Centers")
        ax.set_title("Synapse Centers Overlaid on Image")
        ax.legend()
        plt.show()
    
    
    return smoothed_intensities, derive, maxima_coords   

def worm_segmentation(img):
    
    # Apply Gaussian filter to smooth the image
    img = ski.filters.meijering(img, sigmas=range(8, 14, 2), black_ridges=False) # quand on baisse le sigma max, on garde seulement les vaisseaux fins

    binary_mask = img > np.mean(img)
    
    # Remove small objects from the binary mask
    cleaned_mask = remove_small_objects(binary_mask, option=2, min_size_value=30)
    
    # transform it in boolean
    cleaned_mask = cleaned_mask.astype(bool)
    
    # Fill small holes inside the worm
    worm_mask = ski.morphology.remove_small_holes(cleaned_mask, area_threshold=50)
    
    # Apply binary closing to fill small gaps
    worm_mask = ski.morphology.binary_closing(worm_mask, ski.morphology.disk(20))
    
    # keep the largest connected component (the worm)    
    labeled_mask = ski.measure.label(worm_mask)
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1  # +1 to skip background label
    worm_mask = (labeled_mask == largest_component).astype(np.uint8)
    
    return worm_mask


def combine_maxima(all_maxima, maxima1, maxima2):
    
    return list(set(maxima1) | set(maxima2))
   
def is_a_roll_worm(worm_mask):
    
    # get the major axis of the worm
    labeled_mask = ski.measure.label(worm_mask)
    components = ski.measure.regionprops(labeled_mask)
    major_axis = []
    minor_axis = []
    for component in components:
        major_axis.append(component.major_axis_length)
        minor_axis.append(component.minor_axis_length)
    # get the maximum major axis and minor axis
    max_major_axis = np.max(major_axis)
    min_minor_axis = np.min(minor_axis)
    
    if max_major_axis / min_minor_axis > 1.5:
        return False
    else:
        return True   
    
def get_synapse_using_graph(image):
    # Apply preprocessing to the image
    
    img, local_max = preprocess_image_for_graph(image) 

    # Get a segmentation mask
    worm_mask = worm_segmentation(img)

    if is_a_roll_worm(worm_mask):
        print("Image is a roll worm. Skipping...")
        return [], [], 0
    else:
        # Get the synapses graph to get only the synapses
        maxima, G, median_width = get_synapses_graph(worm_mask, local_max)
        maxima = list(map(tuple, maxima))   
        return maxima, G, median_width
    
def get_preprocess_images(method = 1, recompute=False, X=None, pkl_name=DEFAULT_PKL_NAME):
    """
    Apply preprocessing to images.
    
    Parameters
    ----------
    recompute : bool
        If True, recompute preprocessing. If False, load from file.
    X : ndarray
        Array of images to preprocess
    pkl_name : str
        Base name for the preprocessing pkl file
        
    Returns
    -------
    ndarray
        Preprocessed images
    """
    preprocess_file = f'{Path(pkl_name).stem}.pkl' 
        
    
    # Try to load existing preprocessing
    if not recompute:
        try:
            dict_preprocess = joblib.load(DATASET_PKL_DIR / preprocess_file)
            X_preprocessed = dict_preprocess['X_preprocessed']
            X_intensity = dict_preprocess['X_intensity']    
            X_derivative_intensity = dict_preprocess['X_derivative_intensity']
            maxima_coords = dict_preprocess['maxima_coords']
            mask_synapses = dict_preprocess['mask_synapses']
            print('Preprocessing loaded from file.')
            return X_preprocessed, X_intensity, X_derivative_intensity, maxima_coords, mask_synapses
        except FileNotFoundError:
            print('Preprocessing file not found. Recomputing...')
            recompute = True
    
    # if name already exists, add a number to the name
    i = 1
    while (DATASET_PKL_DIR / preprocess_file).exists():
        preprocess_file = f'{Path(pkl_name).stem}_preprocessing_{i}.pkl'
        i += 1
    
    # Validate input
    if recompute and X is None:
        raise ValueError("Input images (X) must be provided when recomputing preprocessing")
    
    print('Preprocessing images...')
    X_preprocessed = np.zeros_like(X, dtype=np.float64)
    X_intensity = np.zeros((len(X), MAX_LENGTH_OF_FEATURES), dtype=np.float64)
    X_derivative_intensity = np.zeros((len(X), MAX_LENGTH_OF_FEATURES), dtype=np.float64)
    maxima_coords = [None] * len(X)
    mask_synapses = [None] * len(X)
    Graphs = [None] * len(X)
    median_width = [None] * len(X)
    
    
    for im_num, image in enumerate(X):
        
        print(f'Processing image {im_num+1}/{len(X)}')
        
        original_image = image.copy()
            
        if method == 1:
            # Create mask for synapses
            mask_synapses[im_num] = create_mask_synapse(image)
            # apply mask to original image
            X_preprocessed[im_num] = original_image * mask_synapses[im_num]
            X_intensity[im_num], X_derivative_intensity[im_num], maxima_coords[im_num] = get_high_intensity_pixels(mask_synapses[im_num], image)
        elif method == 2:
            try:
                maxima_coords[im_num], Graphs[im_num], median_width[im_num] = get_synapse_using_graph(original_image)
                    
                # creat a mask with disk of radius 5 around each maxima
                mask_synapses[im_num] = np.zeros_like(image)
                for coord in maxima_coords[im_num]:
                    rr, cc = ski.draw.disk(coord, 5, shape=original_image.shape)
                    mask_synapses[im_num][rr, cc] = 1
                        
                # apply mask to original image
                X_preprocessed[im_num] = original_image * mask_synapses[im_num]
            except Exception as e:
                print(f"Error in image {im_num}: {e}")
                maxima_coords[im_num] = []
                mask_synapses[im_num] = np.zeros_like(image)
                X_preprocessed[im_num] = original_image * mask_synapses[im_num]
                X_intensity[im_num], X_derivative_intensity[im_num], maxima_coords[im_num] = get_high_intensity_pixels(mask_synapses[im_num], image)
        elif method == 3:
            # ---------- GET ALL PEAKS ----------
            worm_mask = worm_segmentation(image)    
            masked_img = image.copy()
            masked_img = masked_img * worm_mask
                
            # Find local maxima in the masked image
            all_maxima = ski.feature.peak_local_max(masked_img, min_distance=5, threshold_abs=0, exclude_border=False)
                
            # ---------- FIRST METHOD ----------
            mask_synapses_first_method = create_mask_synapse(image)
            _ ,_ , maxima_first_method = get_high_intensity_pixels(mask_synapses_first_method, image)
            
            # ---------- SECOND METHOD ----------
            maxima_second_method = get_synapse_using_graph(original_image)
                
            # ---------- COMBINAISON ----------
            maxima_coords[im_num] = combine_maxima(all_maxima, maxima_first_method, maxima_second_method)
                
            # creat a mask with disk of radius 5 around each maxima
            mask_synapses[im_num] = np.zeros_like(image)
            for coord in maxima_coords[im_num]:
                rr, cc = ski.draw.disk(coord, 5, shape=original_image.shape)
                mask_synapses[im_num][rr, cc] = 1
                
            # apply mask to original image
            X_preprocessed[im_num] = original_image * mask_synapses[im_num]
        else:
            raise ValueError("Invalid method. Choose 1, 2, or 3.")
    
        
    # Save preprocessing results
    DATASET_PKL_DIR.mkdir(exist_ok=True)
    dict_preprocess = {'X_preprocessed': X_preprocessed, 'X_intensity': X_intensity, 'X_derivative_intensity': X_derivative_intensity, 'maxima_coords': maxima_coords, 'mask_synapses': mask_synapses}
    joblib.dump(dict_preprocess, preprocess_file)
    
    #joblib.dump(X_preprocessed, preprocess_file)
    shutil.move(preprocess_file, DATASET_PKL_DIR)
    print(f'Preprocessing done and saved to {DATASET_PKL_DIR / preprocess_file}')
    
    # return le dictionnaire ? 
    
    return X_preprocessed, X_intensity, X_derivative_intensity, maxima_coords, mask_synapses, Graphs, median_width

