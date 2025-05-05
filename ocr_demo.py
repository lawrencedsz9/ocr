#!/usr/bin/env python
"""
OCR Demonstration Script

This script demonstrates the functionality of the OCRProcessor class from ocr_utils.py.
It allows you to test different preprocessing methods on any image to find the best approach
for text extraction.
"""

import cv2
import numpy as np
import os
import argparse
import sys
import pytesseract
from PIL import Image
from ocr_utils import OCRProcessor

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description='OCR Demonstration Tool')
    parser.add_argument('image_path', help='Path to the image file to process')
    parser.add_argument('--lang', default='eng', help='Language for OCR (e.g., eng, fra, deu)')
    parser.add_argument('--method', choices=['auto', 'basic', 'custom'], default='auto',
                        help='Preprocessing method: auto (try all), basic, or custom')
    parser.add_argument('--custom', 
                        help='Custom preprocessing methods, comma-separated (e.g., grayscale,threshold,sharpen)')
    parser.add_argument('--tesseract', default=r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                        help='Path to Tesseract OCR executable')
    parser.add_argument('--save-images', action='store_true', 
                        help='Save intermediate preprocessed images')
    parser.add_argument('--psm', type=int, default=6, 
                        help='Page Segmentation Mode (0-13)')
    
    args = parser.parse_args()
    
    # Verify image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Configure Tesseract path
    try:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract
        print(f"Using Tesseract executable: {args.tesseract}")
        print(f"Testing Tesseract...")
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except Exception as e:
        print(f"Error configuring Tesseract: {e}")
        alt_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'/usr/bin/tesseract',
            r'/usr/local/bin/tesseract'
        ]
        
        # Try alternative paths
        for path in alt_paths:
            try:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"Found alternative Tesseract path: {path}")
                    version = pytesseract.get_tesseract_version()
                    print(f"Tesseract version: {version}")
                    break
            except:
                continue
        else:
            print("Error: Tesseract not found. Please install Tesseract OCR and set the correct path.")
            sys.exit(1)
    
    # Initialize OCR processor
    ocr = OCRProcessor(tesseract_cmd=pytesseract.pytesseract.tesseract_cmd, 
                      lang=args.lang, psm=args.psm)
    
    # Read image
    try:
        print(f"Reading image: {args.image_path}")
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError("Failed to read image")
        
        print(f"Image shape: {image.shape}")
        
        # Create output directory for preprocessed images
        if args.save_images:
            output_dir = "preprocessed_images"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving preprocessed images to: {output_dir}")
    
    except Exception as e:
        print(f"Error reading image: {e}")
        sys.exit(1)
    
    # Process image according to the selected method
    try:
        if args.method == 'auto':
            print("\nTrying multiple preprocessing combinations automatically...")
            results = ocr.try_all_preprocessing_combinations(image)
            
            print("\n--- Results Summary ---")
            for i, (text, methods, text_len) in enumerate(results):
                print(f"\nMethod {i+1}: {methods} - {text_len} characters")
                print("-" * 40)
                print(text[:200] + ("..." if len(text) > 200 else ""))
                
                if args.save_images:
                    # Save the preprocessed image for this method
                    preprocessed = ocr.preprocess_image(image, methods)
                    filename = f"{output_dir}/method_{i+1}_{'_'.join(methods)}.png"
                    cv2.imwrite(filename, preprocessed)
                    print(f"Saved preprocessed image to: {filename}")
            
            # Print full text of best result
            best_text = results[0][0] if results else ""
            print("\n\n=== BEST RESULT ===")
            print(best_text)
            
        elif args.method == 'basic':
            methods = ['grayscale', 'threshold']
            print(f"\nUsing basic preprocessing methods: {methods}")
            
            if args.save_images:
                preprocessed = ocr.preprocess_image(image, methods)
                filename = f"{output_dir}/basic_preprocessing.png"
                cv2.imwrite(filename, preprocessed)
                print(f"Saved preprocessed image to: {filename}")
            
            text = ocr.extract_text(image, methods)
            print("\n--- Extracted Text ---")
            print(text)
            
        elif args.method == 'custom':
            if not args.custom:
                print("Error: '--custom' parameter is required with custom method")
                sys.exit(1)
                
            methods = args.custom.split(',')
            print(f"\nUsing custom preprocessing methods: {methods}")
            
            if args.save_images:
                preprocessed = ocr.preprocess_image(image, methods)
                filename = f"{output_dir}/custom_preprocessing.png"
                cv2.imwrite(filename, preprocessed)
                print(f"Saved preprocessed image to: {filename}")
            
            text = ocr.extract_text(image, methods)
            print("\n--- Extracted Text ---")
            print(text)
    
    except Exception as e:
        import traceback
        print(f"Error processing OCR: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 