"""
Project: Text Segmentation using Graph Cut Algorithm

Description:
This project extracts text from images by separating text (foreground) from background. 
It uses OpenCV's GrabCut algorithm, which is based on graph cuts - a powerful method that 
models the image as a network where each pixel is connected to its neighbors. 
The algorithm finds the optimal way to "cut" this network into two parts: text and background.

How It Works:
1. We first convert the image to grayscale and enhance its contrast
2. We create an initial guess of what might be text using simple thresholding
3. The GrabCut algorithm then refines this initial guess by:
   - Building a graph where pixels are connected to their neighbors
   - Creating models of what text and background look like
   - Finding the best separation between text and background
4. Finally, we clean up the result by removing small isolated spots and
   connecting nearby text regions that should be together

Key functionalities:
- Image preprocessing with adaptive contrast enhancement
- Initial segmentation using Otsu's thresholding
- Graph-based segmentation using OpenCV's GrabCut implementation
- Connected component analysis to filter text-like regions
- Morphological post-processing to refine text shapes

Applications:
- Extracting text from documents with complex backgrounds
- Detecting text in natural scene images (street signs, store fronts, etc.)
- Preparing images for text recognition (OCR) systems
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pytesseract
from ocr_utils import OCRProcessor

# Add this after the import statements
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed

class TextSegmentationGrabCut:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.segmentation_mask = None
        
        # Initialize OCR processor
        self.ocr_processor = OCRProcessor(tesseract_cmd=pytesseract.pytesseract.tesseract_cmd)
        
    def preprocess_image(self):
        """Preprocess the image for segmentation"""
        print(f"Reading image from: {self.image_path}")
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read the image. File exists: {os.path.exists(self.image_path)}")
        
        print(f"Image shape: {self.original_image.shape}")
            
        if max(self.original_image.shape) > 1000:
            scale_factor = 1000 / max(self.original_image.shape)
            new_width = int(self.original_image.shape[1] * scale_factor)
            new_height = int(self.original_image.shape[0] * scale_factor)
            self.original_image = cv2.resize(self.original_image, (new_width, new_height))
            print(f"Resized to {new_width}x{new_height} for computational efficiency")
        
    def create_initial_mask(self):
        """Create initial mask for GrabCut"""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        height, width = gray.shape
        rect = (0, 0, width, height)
        
        # Create initial mask for GrabCut
        # GrabCut mask values:
        # 0 = background, 1 = foreground, 2 = probable background, 3 = probable foreground
        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        
        
        mask[binary == 255] = cv2.GC_PR_FGD  
        
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.erode(binary, kernel, iterations=2)
        mask[fg_mask == 255] = cv2.GC_FGD  
        
       
        self.binary = binary
        
        return mask, rect
    
    def apply_graphcut(self, mask, rect):
        """Apply GrabCut algorithm"""
        print("Applying GrabCut (Graph Cut) algorithm...")
        
        try:
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            mask_copy = mask.copy()  
            cv2.grabCut(self.original_image, mask_copy, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            
            foreground_mask = np.where((mask_copy == cv2.GC_FGD) | (mask_copy == cv2.GC_PR_FGD), 255, 0).astype('uint8')

            self.segmentation_mask = self.postprocess_mask(foreground_mask)
            
        except Exception as e:
            print(f"Error in GraphCut algorithm: {e}")
            # Fallback to the binary mask if GraphCut fails
            print("Falling back to binary thresholding...")
            _, binary = cv2.threshold(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), 
                                    0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.segmentation_mask = self.postprocess_mask(binary)
    
    def postprocess_mask(self, mask):
        """Clean up the segmentation mask"""
      
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        min_area = 30
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
               
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = w / float(h) if h > 0 else 0
                
                if 0.1 <= aspect_ratio <= 20 and h >= 5:
                    filtered_mask[labels == i] = 255
        kernel = np.ones((3, 3), np.uint8)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return filtered_mask
    
    def extract_text(self, preprocessing_methods=None):
        """Extract text from the segmented image using OCR"""
        if self.segmentation_mask is None:
            raise ValueError("Segmentation must be performed before text extraction")
        
        # Create masked image for OCR
        mask_3d = np.zeros(self.original_image.shape, dtype=np.uint8)
        mask_3d[:, :, 0] = self.segmentation_mask
        mask_3d[:, :, 1] = self.segmentation_mask
        mask_3d[:, :, 2] = self.segmentation_mask
        
        masked_img = cv2.bitwise_and(self.original_image, mask_3d)
        
        # Save masked image for debugging
        cv2.imwrite("segmented_text.png", masked_img)
        print("Saved segmented text image to: segmented_text.png")
        
        # Use OCR processor to extract text
        if preprocessing_methods is None:
            # Try different preprocessing methods and use the best one
            print("Trying different preprocessing methods for OCR...")
            results = self.ocr_processor.try_all_preprocessing_combinations(masked_img, top_n=3)
            
            if results:
                best_text, best_methods, _ = results[0]
                print(f"Best preprocessing methods: {best_methods}")
                print(f"Extracted text:\n{best_text}")
                return best_text
            else:
                print("Failed to extract text with any preprocessing method")
                return ""
        else:
            # Use specified preprocessing methods
            print(f"Using specified preprocessing methods: {preprocessing_methods}")
            text = self.ocr_processor.extract_text(masked_img, preprocessing_methods)
            print(f"Extracted text:\n{text}")
            return text
    
    def visualize_results(self):
        """Visualize the segmentation process and results"""
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(221)
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Binary image
        plt.subplot(222)
        plt.imshow(self.binary, cmap='gray')
        plt.title('Initial Binary Image')
        plt.axis('off')
        
        # Segmentation mask
        plt.subplot(223)
        plt.imshow(self.segmentation_mask, cmap='gray')
        plt.title('Graph Cut Segmentation')
        plt.axis('off')
        
        # Overlay result
        plt.subplot(224)
        overlay = self.original_image.copy()
        overlay[self.segmentation_mask == 255] = [0, 255, 0]  
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title('Text Regions (Green)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process(self):
        """Run the complete Graph Cut segmentation process"""
        self.preprocess_image()
        mask, rect = self.create_initial_mask()
        self.apply_graphcut(mask, rect)
        self.visualize_results()
        
        # Extract text from the segmented image
        text = self.extract_text()
        print("\nExtracted Text:")
        print("="*50)
        print(text)
        print("="*50)
        
        return self.segmentation_mask, text

if __name__ == "__main__":
    image_path = input("Enter the path to the image file: ")
    
    try:
        print(f"Processing image: {image_path}")
        segmenter = TextSegmentationGrabCut(image_path)
        segmenter.process()
        print("Processing completed successfully")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc()) 