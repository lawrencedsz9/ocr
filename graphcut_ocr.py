#!/usr/bin/env python
"""
GraphCut OCR Script

This script uses GraphCut-based text segmentation to enhance OCR accuracy.
It combines the TextSegmentationGrabCut class with the OCRProcessor to achieve better results.
"""

import os
import sys
import cv2
import numpy as np
import argparse
import pytesseract
from ocr_utils import OCRProcessor
from opencv_graphcut import TextSegmentationGrabCut
import matplotlib.pyplot as plt

def show_results(original_image, segmented_image, extracted_text):
    """Display the original image, segmented image, and extracted text in one figure"""
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Segmented image
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Text')
    plt.axis('off')
    
    # Extracted text
    plt.subplot(212)
    plt.text(0.05, 0.5, extracted_text, fontsize=12, wrap=True)
    plt.title('Extracted Text')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_image(image_path, lang='eng', show_visualization=True, save_results=False):
    """Process the image using GraphCut-based OCR"""
    print(f"Processing image: {image_path}")
    
    try:
        # Step 1: First segment the image using GraphCut for text extraction
        print("Applying GraphCut text segmentation...")
        segmenter = TextSegmentationGrabCut(image_path)
        segmenter.preprocess_image()
        mask, rect = segmenter.create_initial_mask()
        segmenter.apply_graphcut(mask, rect)
        
        # Step 2: Create a masked image with only the text
        original_image = segmenter.original_image.copy()
        mask_3d = np.zeros(original_image.shape, dtype=np.uint8)
        mask_3d[:, :, 0] = segmenter.segmentation_mask
        mask_3d[:, :, 1] = segmenter.segmentation_mask
        mask_3d[:, :, 2] = segmenter.segmentation_mask
        segmented_image = cv2.bitwise_and(original_image, mask_3d)
        
        # Save segmented image if requested
        if save_results:
            segmented_path = os.path.splitext(image_path)[0] + "_segmented.png"
            cv2.imwrite(segmented_path, segmented_image)
            print(f"Saved segmented image to: {segmented_path}")
        
        # Step 3: Apply OCR with additional processing to improve results
        print("Performing OCR on segmented image...")
        ocr = OCRProcessor(lang=lang)
        
        # Try multiple preprocessing methods
        results = []
        
        # Method 1: Just segmentation with basic processing
        text1 = ocr.extract_text(segmented_image, ['grayscale', 'threshold'])
        results.append((text1, "Basic", len(text1)))
        
        # Method 2: Segmentation with contrast enhancement
        text2 = ocr.extract_text(segmented_image, ['grayscale', 'contrast', 'threshold'])
        results.append((text2, "Contrast", len(text2)))
        
        # Method 3: Segmentation with sharpening
        text3 = ocr.extract_text(segmented_image, ['grayscale', 'sharpen', 'threshold'])
        results.append((text3, "Sharpening", len(text3)))
        
        # Method 4: Segmentation with adaptive thresholding
        text4 = ocr.extract_text(segmented_image, ['grayscale', 'adaptive_threshold'])
        results.append((text4, "Adaptive", len(text4)))
        
        # Sort by text length (more text generally means better quality)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Show the best result
        best_text, best_method, _ = results[0]
        print(f"\nBest result (using {best_method}):")
        print("=" * 40)
        print(best_text)
        print("=" * 40)
        
        # Save text if requested
        if save_results:
            text_path = os.path.splitext(image_path)[0] + "_ocr.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"--- GraphCut OCR Results ---\n\n")
                f.write(f"Best method: {best_method}\n\n")
                f.write(best_text)
                f.write("\n\n--- Alternative Results ---\n\n")
                for text, method, _ in results[1:]:
                    f.write(f"Method: {method}\n")
                    f.write(text)
                    f.write("\n\n")
            print(f"Saved OCR results to: {text_path}")
        
        # Display results if requested
        if show_visualization:
            show_results(original_image, segmented_image, best_text)
        
        return best_text, segmented_image
    
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return None, None

def main():
    parser = argparse.ArgumentParser(description='GraphCut OCR: Text extraction with GraphCut segmentation')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--lang', default='eng', help='OCR language (e.g., eng, fra, deu)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--save', action='store_true', help='Save results to disk')
    
    args = parser.parse_args()
    
    # Verify file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Process the image
    process_image(args.image_path, args.lang, not args.no_viz, args.save)

if __name__ == "__main__":
    main() 