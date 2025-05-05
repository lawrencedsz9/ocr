"""
grabcut_core_fromcv2.py

Core implementation of the GrabCut algorithm based on OpenCV's implementation.
This file contains the main algorithm for interactive foreground extraction
using iterated graph cuts.
"""

import numpy as np
import cv2
import maxflow
from grabcut_utils_fromcv2 import GMM, calc_smoothness_weights, normalized_rgb
from grabcut_utils_fromcv2 import GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def grabcut(img, mask, rect=None, n_iter=5, gamma=50, max_threads=4):
    """
    Interactive foreground/background segmentation using GrabCut algorithm.
    
    Args:
        img: Input 8-bit 3-channel image
        mask: Input/output 8-bit mask (initial segmentation)
            - GC_BGD (0) for background pixels
            - GC_FGD (1) for foreground pixels
            - GC_PR_BGD (2) for possible background pixels
            - GC_PR_FGD (3) for possible foreground pixels
        rect: Rectangle containing the object of interest (x, y, width, height)
        n_iter: Number of algorithm iterations
        gamma: Parameter for smoothness term
        max_threads: Maximum number of threads for parallel processing
    
    Returns:
        Updated mask
    """
    # Start timing
    start_time = time.time()
    
    # Initialize mask if a rectangle is provided
    if rect is not None:
        x, y, w, h = rect
        
        # Initialize the mask
        mask[:] = GC_BGD
        mask[y:y+h, x:x+w] = GC_PR_FGD
    
    # Check if mask contains all 4 states
    if np.all(mask != GC_FGD) and np.all(mask != GC_PR_FGD):
        print("Mask does not have any foreground pixels!")
        return mask
    
    # Create GMM models for background and foreground
    bgd_gmm = GMM()
    fgd_gmm = GMM()
    
    # Prepare image data
    h, w = img.shape[:2]
    img_array = np.asarray(img, dtype=np.float64)
    
    # Get indices for each region
    bgd_idx = np.where((mask == GC_BGD) | (mask == GC_PR_BGD))
    fgd_idx = np.where((mask == GC_FGD) | (mask == GC_PR_FGD))
    
    # Extract pixel values for background and foreground
    bgd_pixels = img_array[bgd_idx]
    fgd_pixels = img_array[fgd_idx]
    
    print(f"Initial pixel counts - BGD: {len(bgd_pixels)}, FGD: {len(fgd_pixels)}")
    
    # Initialize GMM models
    bgd_gmm.initialize(bgd_pixels)
    fgd_gmm.initialize(fgd_pixels)
    
    print(f"GMM initialization completed in {time.time() - start_time:.2f} seconds")
    
    # Calculate edge weights for graph
    edge_weights = calc_smoothness_weights(img_array, gamma)
    
    print(f"Edge weights calculated in {time.time() - start_time:.2f} seconds")
    
    # Reshape image for easier processing
    img_flat = img_array.reshape(-1, 3)
    
    # Component assignments
    bgd_components = np.zeros(len(bgd_pixels), dtype=np.int32)
    fgd_components = np.zeros(len(fgd_pixels), dtype=np.int32)
    
    # Main algorithm iterations
    for i in range(n_iter):
        iter_start = time.time()
        print(f"Iteration {i+1}/{n_iter}")
        
        # 1. Assign GMM components
        print("  Assigning GMM components...")
        bgd_components = bgd_gmm.assign_components(bgd_pixels)
        fgd_components = fgd_gmm.assign_components(fgd_pixels)
        
        # 2. Update GMM parameters
        print("  Updating GMM parameters...")
        bgd_gmm.update_parameters(bgd_pixels)
        fgd_gmm.update_parameters(fgd_pixels)
        
        # 3. Construct graph and calculate terminal weights (source/sink)
        print("  Creating graph and calculating weights...")
        graph = create_graph(img_array, mask, bgd_gmm, fgd_gmm, edge_weights, max_threads)
        
        # 4. Run maxflow algorithm
        print("  Running maxflow algorithm...")
        graph.maxflow()
        
        # 5. Update segmentation based on min cut
        print("  Updating segmentation mask...")
        mask_changed = update_mask(mask, graph, w, h)
        
        # 6. Update pixel lists
        bgd_idx = np.where((mask == GC_BGD) | (mask == GC_PR_BGD))
        fgd_idx = np.where((mask == GC_FGD) | (mask == GC_PR_FGD))
        
        bgd_pixels = img_array[bgd_idx]
        fgd_pixels = img_array[fgd_idx]
        
        print(f"  Updated pixel counts - BGD: {len(bgd_pixels)}, FGD: {len(fgd_pixels)}")
        print(f"  Iteration completed in {time.time() - iter_start:.2f} seconds")
        
        # Check for convergence
        if not mask_changed:
            print(f"Converged after {i+1} iterations")
            break
    
    # Clean up the mask (convert probable regions to definite regions)
    final_mask = np.zeros_like(mask, dtype=np.uint8)
    final_mask[mask == GC_BGD] = 0
    final_mask[mask == GC_PR_BGD] = 0
    final_mask[mask == GC_FGD] = 1
    final_mask[mask == GC_PR_FGD] = 1
    
    print(f"GrabCut completed in {time.time() - start_time:.2f} seconds")
    
    return final_mask

def process_pixel_chunk(chunk_data):
    """
    Process a chunk of pixels to calculate terminal weights.
    This function is used for parallel processing.
    
    Args:
        chunk_data: Tuple containing (chunk_idx, img_flat, mask_flat, bgd_gmm, fgd_gmm)
    
    Returns:
        Tuple of (chunk_idx, bgd_weights, fgd_weights)
    """
    chunk_idx, img_flat, mask_flat, bgd_gmm, fgd_gmm = chunk_data
    
    bgd_weights = np.zeros(len(chunk_idx), dtype=np.float64)
    fgd_weights = np.zeros(len(chunk_idx), dtype=np.float64)
    
    for i, idx in enumerate(chunk_idx):
        pixel = img_flat[idx]
        
        # Skip definite background/foreground pixels for terminal weights
        if mask_flat[idx] == GC_BGD or mask_flat[idx] == GC_FGD:
            continue
        
        # Calculate negative log probabilities (energy)
        bgd_energy = bgd_gmm.calc_pixel_energy(pixel)
        fgd_energy = fgd_gmm.calc_pixel_energy(pixel)
        
        bgd_weights[i] = bgd_energy
        fgd_weights[i] = fgd_energy
    
    return chunk_idx, bgd_weights, fgd_weights

def create_graph(img, mask, bgd_gmm, fgd_gmm, edge_weights, max_threads=4):
    """
    Create the graph for mincut/maxflow calculation.
    
    Args:
        img: Input image
        mask: Current segmentation mask
        bgd_gmm: Background GMM model
        fgd_gmm: Foreground GMM model
        edge_weights: Tuple of (leftW, upleftW, upW, uprightW)
        max_threads: Maximum number of threads for parallel processing
    
    Returns:
        Graph object ready for maxflow calculation
    """
    h, w = img.shape[:2]
    n_nodes = w * h
    n_edges = 2*(4*w*h - 3*w - 3*h + 2)  # Estimate number of edges
    
    # Create new graph
    graph = maxflow.Graph[float](n_nodes, n_edges)
    graph.add_nodes(n_nodes)
    
    leftW, upleftW, upW, uprightW = edge_weights
    
    # Flatten arrays for easier processing
    img_flat = img.reshape(-1, 3)
    mask_flat = mask.reshape(-1)
    
    # Divide work for parallel processing
    all_indices = np.arange(n_nodes)
    chunk_size = max(1, n_nodes // max_threads)
    chunks = [all_indices[i:i+chunk_size] for i in range(0, n_nodes, chunk_size)]
    
    # Calculate terminal weights in parallel
    terminal_weights = {}
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i, chunk_idx in enumerate(chunks):
            futures.append(executor.submit(
                process_pixel_chunk, 
                (chunk_idx, img_flat, mask_flat, bgd_gmm, fgd_gmm)
            ))
        
        for future in as_completed(futures):
            chunk_idx, bgd_weights, fgd_weights = future.result()
            for i, idx in enumerate(chunk_idx):
                terminal_weights[idx] = (bgd_weights[i], fgd_weights[i])
    
    # Add terminal edges (to source and sink)
    for y in range(h):
        for x in range(w):
            node_idx = y * w + x
            
            # Hard constraints (infinite weights)
            if mask[y, x] == GC_BGD:
                # Definite background - strong connection to sink
                graph.add_tedge(node_idx, 0, 9*1e10)
            elif mask[y, x] == GC_FGD:
                # Definite foreground - strong connection to source
                graph.add_tedge(node_idx, 9*1e10, 0)
            else:
                # Probable background/foreground
                b_weight, f_weight = terminal_weights.get(node_idx, (0, 0))
                graph.add_tedge(node_idx, f_weight, b_weight)
    
    # Add non-terminal edges (between pixels)
    for y in range(h):
        for x in range(w):
            node_idx = y * w + x
            
            # Right neighbor
            if x < w - 1:
                right_idx = node_idx + 1
                graph.add_edge(node_idx, right_idx, leftW[y, x+1], leftW[y, x+1])
            
            # Bottom neighbor
            if y < h - 1:
                bottom_idx = node_idx + w
                graph.add_edge(node_idx, bottom_idx, upW[y+1, x], upW[y+1, x])
            
            # Bottom-left neighbor
            if x > 0 and y < h - 1:
                bottom_left_idx = node_idx + w - 1
                graph.add_edge(node_idx, bottom_left_idx, upleftW[y+1, x-1], upleftW[y+1, x-1])
            
            # Bottom-right neighbor
            if x < w - 1 and y < h - 1:
                bottom_right_idx = node_idx + w + 1
                graph.add_edge(node_idx, bottom_right_idx, uprightW[y+1, x+1], uprightW[y+1, x+1])
    
    return graph

def update_mask(mask, graph, w, h):
    """
    Update the segmentation mask based on min cut results.
    
    Args:
        mask: Current segmentation mask
        graph: Graph after maxflow calculation
        w, h: Width and height of the image
    
    Returns:
        bool: True if mask was changed, False otherwise
    """
    mask_changed = False
    
    for y in range(h):
        for x in range(w):
            node_idx = y * w + x
            
            # Skip definite background/foreground pixels
            if mask[y, x] == GC_BGD or mask[y, x] == GC_FGD:
                continue
            
            # Check if pixel belongs to source (foreground) or sink (background)
            if graph.get_segment(node_idx):  # sink segment
                if mask[y, x] != GC_BGD and mask[y, x] != GC_PR_BGD:
                    mask[y, x] = GC_PR_BGD
                    mask_changed = True
            else:  # source segment
                if mask[y, x] != GC_FGD and mask[y, x] != GC_PR_FGD:
                    mask[y, x] = GC_PR_FGD
                    mask_changed = True
    
    return mask_changed

def interactive_grabcut(img, rect=None, mask=None, n_iter=5, gamma=50, max_threads=4):
    """
    Interactive wrapper for the GrabCut algorithm.
    
    Args:
        img: Input image
        rect: Rectangle containing foreground object (x, y, width, height)
        mask: Optional initial mask
        n_iter: Number of iterations
        gamma: Parameter for smoothness term
        max_threads: Maximum number of threads for parallel processing
    
    Returns:
        Final binary mask where 1 = foreground, 0 = background
    """
    # Make a copy of the input image
    img_copy = img.copy()
    h, w = img.shape[:2]
    
    # Initialize mask
    if mask is None:
        mask = np.zeros((h, w), dtype=np.uint8)
        # Use rectangle to initialize mask if provided
        if rect is not None:
            x, y, rw, rh = rect
            mask[:] = GC_BGD
            mask[y:y+rh, x:x+rw] = GC_PR_FGD
    else:
        # If mask is provided, make sure it's the right size
        if mask.shape[:2] != (h, w):
            raise ValueError("Mask size doesn't match image size")
    
    # Run GrabCut algorithm
    result_mask = grabcut(img_copy, mask, rect, n_iter, gamma, max_threads)
    
    # Return binary mask
    return result_mask * 255 