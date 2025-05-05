"""
grabcut_utils_fromcv2.py

Utility functions for the GrabCut algorithm, based on OpenCV's implementation.
These functions support the main GrabCut algorithm in grabcut_core_fromcv2.py.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans

# Constants for GrabCut algorithm
GC_BGD = 0      # Hard background
GC_FGD = 1      # Hard foreground
GC_PR_BGD = 2   # Probable background
GC_PR_FGD = 3   # Probable foreground

# For GMM model
GMM_COMPONENT_COUNT = 5

def normalized_rgb(img):
    """
    Convert image to normalized RGB color space.
    
    Args:
        img: Input BGR image
    
    Returns:
        Normalized RGB image
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sum_channels = np.sum(rgb, axis=2, keepdims=True)
    
    # Avoid division by zero
    sum_channels[sum_channels == 0] = 1
    
    normalized = rgb / sum_channels
    return normalized

def calc_beta(img):
    """
    Calculate beta parameter for edge weights.
    
    Args:
        img: Input image
    
    Returns:
        beta: Parameter for edge weights calculation
    """
    h, w = img.shape[:2]
    beta = 0
    count = 0
    
    # Calculate squared color differences for neighboring pixels
    for y in range(h):
        for x in range(w-1):
            diff = np.sum((img[y, x] - img[y, x+1])**2)
            beta += diff
            count += 1
    
    for y in range(h-1):
        for x in range(w):
            diff = np.sum((img[y, x] - img[y+1, x])**2)
            beta += diff
            count += 1
    
    # Avoid division by zero
    if count == 0:
        return 0
    
    # Calculate beta as described in the paper
    beta = 1.0 / (2 * beta / count)
    return beta

def calc_smoothness_weights(img, gamma=50):
    """
    Calculate smoothness weights for the graph.
    
    Args:
        img: Input image
        gamma: Weight for neighboring penalties
    
    Returns:
        leftW, upleftW, upW, uprightW: Edge weights in different directions
    """
    h, w = img.shape[:2]
    
    # Calculate beta from the image
    beta = calc_beta(img)
    
    # Initialize weight matrices
    leftW = np.zeros((h, w))
    upleftW = np.zeros((h, w))
    upW = np.zeros((h, w))
    uprightW = np.zeros((h, w))
    
    # Calculate weights for each pixel's neighbors
    for y in range(h):
        for x in range(w):
            # Left neighbor
            if x > 0:
                diff = np.sum((img[y, x] - img[y, x-1])**2)
                leftW[y, x] = gamma * np.exp(-beta * diff)
            
            # Up neighbor
            if y > 0:
                diff = np.sum((img[y, x] - img[y-1, x])**2)
                upW[y, x] = gamma * np.exp(-beta * diff)
            
            # Up-left neighbor
            if x > 0 and y > 0:
                diff = np.sum((img[y, x] - img[y-1, x-1])**2)
                upleftW[y, x] = gamma * np.exp(-beta * diff) / np.sqrt(2)
            
            # Up-right neighbor
            if x < w-1 and y > 0:
                diff = np.sum((img[y, x] - img[y-1, x+1])**2)
                uprightW[y, x] = gamma * np.exp(-beta * diff) / np.sqrt(2)
    
    return leftW, upleftW, upW, uprightW

class GMM:
    """
    Gaussian Mixture Model for the GrabCut algorithm.
    """
    def __init__(self, component_count=GMM_COMPONENT_COUNT):
        self.component_count = component_count
        self.weights = np.zeros(component_count)
        self.means = np.zeros((component_count, 3))
        self.covs = np.zeros((component_count, 3, 3))
        self.cov_invs = np.zeros((component_count, 3, 3))
        self.cov_dets = np.zeros(component_count)
        self.pixel_count = 0
        self.components = None
    
    def initialize(self, pixels):
        """
        Initialize the GMM with K-means clustering.
        
        Args:
            pixels: Array of pixel colors
        """
        if pixels.size == 0:
            return False
        
        # Use K-means for initial clustering
        kmeans = KMeans(n_clusters=self.component_count, n_init=10).fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        self.components = labels
        
        # Calculate weights, means, and covariances for each component
        for k in range(self.component_count):
            mask = (labels == k)
            count = np.sum(mask)
            if count == 0:
                continue
            
            self.weights[k] = count / float(len(pixels))
            self.means[k] = centers[k]
            
            # Calculate covariance
            diff = pixels[mask] - self.means[k]
            self.covs[k] = np.dot(diff.T, diff) / count
            
            # Add a small value to diagonal to ensure invertibility
            self.covs[k] += np.eye(3) * 0.01
            
            # Calculate inverse and determinant
            self.cov_invs[k] = np.linalg.inv(self.covs[k])
            self.cov_dets[k] = np.linalg.det(self.covs[k])
        
        self.pixel_count = len(pixels)
        return True
    
    def update_parameters(self, pixels):
        """
        Update GMM parameters based on current component assignments.
        
        Args:
            pixels: Array of pixel colors
        """
        if len(pixels) == 0:
            return False
            
        for k in range(self.component_count):
            mask = (self.components == k)
            count = np.sum(mask)
            
            if count == 0:
                # Handle empty component
                continue
            
            # Update weight
            self.weights[k] = count / float(len(pixels))
            
            # Update mean
            self.means[k] = np.mean(pixels[mask], axis=0)
            
            # Update covariance
            diff = pixels[mask] - self.means[k]
            self.covs[k] = np.dot(diff.T, diff) / count
            
            # Add a small value to diagonal to ensure invertibility
            self.covs[k] += np.eye(3) * 0.01
            
            # Update inverse and determinant
            self.cov_invs[k] = np.linalg.inv(self.covs[k])
            self.cov_dets[k] = np.linalg.det(self.covs[k])
        
        return True
    
    def assign_components(self, pixels):
        """
        Assign each pixel to the most likely GMM component.
        
        Args:
            pixels: Array of pixel colors
        
        Returns:
            Array of component assignments
        """
        if len(pixels) == 0:
            return np.array([])
            
        probs = np.zeros((len(pixels), self.component_count))
        
        for k in range(self.component_count):
            # Avoid components with zero weight
            if self.weights[k] < 1e-10 or self.cov_dets[k] < 1e-10:
                continue
                
            # Calculate probability for each pixel for this component
            diff = pixels - self.means[k]
            
            # Mahalanobis distance calculation
            mahala_dist = np.sum(diff @ self.cov_invs[k] * diff, axis=1)
            
            # Log probability
            log_prob = -0.5 * (np.log(2 * np.pi) * 3 + np.log(self.cov_dets[k]) + mahala_dist)
            probs[:, k] = log_prob + np.log(self.weights[k])
        
        # Get component with maximum probability for each pixel
        self.components = np.argmax(probs, axis=1)
        return self.components
    
    def calc_pixel_energy(self, pixel):
        """
        Calculate the energy (negative log likelihood) of a pixel.
        
        Args:
            pixel: Single pixel color
        
        Returns:
            Energy value
        """
        min_energy = float('inf')
        
        for k in range(self.component_count):
            # Skip components with zero weight
            if self.weights[k] < 1e-10 or self.cov_dets[k] < 1e-10:
                continue
                
            # Calculate Mahalanobis distance
            diff = pixel - self.means[k]
            mahala_dist = diff @ self.cov_invs[k] @ diff.T
            
            # Calculate energy
            energy = -np.log(self.weights[k]) + 0.5 * (np.log(self.cov_dets[k]) + mahala_dist)
            min_energy = min(min_energy, energy)
        
        if min_energy == float('inf'):
            return 0
            
        return min_energy 