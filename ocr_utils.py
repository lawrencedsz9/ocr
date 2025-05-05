"""
OCR Utilities Module

This module provides utilities for Optical Character Recognition (OCR) using pytesseract
with enhanced preprocessing techniques for better text extraction from images.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import os
import tempfile

class OCRProcessor:
    """
    A class to handle OCR processing with various preprocessing techniques to improve text recognition.
    """
    
    def __init__(self, tesseract_cmd=None, lang='eng', oem=3, psm=6):
        """
        Initialize the OCR processor.
        
        Args:
            tesseract_cmd (str): Path to tesseract executable
            lang (str): Language(s) for OCR, comma-separated for multiple languages
            oem (int): OCR Engine Mode (0-3)
            psm (int): Page Segmentation Mode (0-13)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.lang = lang
        self.config = f'--oem {oem} --psm {psm}'
        
        # Print the Tesseract version and path for debugging
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Using Tesseract version: {version}")
            print(f"Tesseract command path: {pytesseract.pytesseract.tesseract_cmd}")
        except Exception as e:
            print(f"Warning: Could not get Tesseract version. Error: {e}")
    
    def preprocess_image(self, image, methods=None):
        """
        Apply preprocessing methods to enhance image for better OCR.
        
        Args:
            image: PIL Image or numpy array
            methods (list): List of preprocessing methods to apply
                Options: 'grayscale', 'threshold', 'adaptive_threshold', 'denoise',
                         'contrast', 'sharpen', 'deskew', 'erode', 'dilate'
        
        Returns:
            Preprocessed image (numpy array)
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img = np.array(image)
            # Convert RGB to BGR for OpenCV
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        if methods is None:
            methods = ['grayscale', 'threshold']
        
        # Apply selected preprocessing methods
        for method in methods:
            if method == 'grayscale':
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            elif method == 'threshold':
                if len(img.shape) == 2:  # Ensure grayscale
                    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            elif method == 'adaptive_threshold':
                if len(img.shape) == 2:  # Ensure grayscale
                    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
            
            elif method == 'denoise':
                if len(img.shape) == 2:  # For grayscale
                    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
                else:  # For color images
                    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            
            elif method == 'contrast':
                # Convert to PIL Image for enhancement
                if len(img.shape) == 2:  # For grayscale
                    pil_img = Image.fromarray(img)
                else:  # For color images
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(2.0)  # Increase contrast
                
                # Convert back to numpy array
                if len(img.shape) == 2:  # For grayscale
                    img = np.array(pil_img)
                else:  # For color images
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            elif method == 'sharpen':
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                img = cv2.filter2D(img, -1, kernel)
            
            elif method == 'deskew':
                if len(img.shape) == 2:  # Ensure grayscale
                    # Find all non-zero points
                    coords = np.column_stack(np.where(img > 0))
                    if len(coords) > 0:
                        # Find the minimum area rectangle
                        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
                        # Adjust the angle
                        if angle < -45:
                            angle = -(90 + angle)
                        else:
                            angle = -angle
                        # Rotate the image
                        (h, w) = img.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, 
                                           borderMode=cv2.BORDER_REPLICATE)
            
            elif method == 'erode':
                kernel = np.ones((2, 2), np.uint8)
                img = cv2.erode(img, kernel, iterations=1)
            
            elif method == 'dilate':
                kernel = np.ones((2, 2), np.uint8)
                img = cv2.dilate(img, kernel, iterations=1)
            
            elif method == 'graphcut':
                # Apply GraphCut-based text segmentation
                try:
                    # Import here to avoid circular imports
                    from opencv_graphcut import TextSegmentationGrabCut
                    
                    # Save the image temporarily
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    temp_filename = temp_file.name
                    temp_file.close()
                    
                    cv2.imwrite(temp_filename, img)
                    
                    # Create segmenter instance
                    segmenter = TextSegmentationGrabCut(temp_filename)
                    segmenter.preprocess_image()
                    mask, rect = segmenter.create_initial_mask()
                    segmenter.apply_graphcut(mask, rect)
                    
                    # Create a masked image with only the text
                    if len(img.shape) == 3:  # Color image
                        mask_3d = np.zeros(img.shape, dtype=np.uint8)
                        mask_3d[:, :, 0] = segmenter.segmentation_mask
                        mask_3d[:, :, 1] = segmenter.segmentation_mask
                        mask_3d[:, :, 2] = segmenter.segmentation_mask
                        img = cv2.bitwise_and(img, mask_3d)
                    else:  # Grayscale image
                        img = cv2.bitwise_and(img, img, mask=segmenter.segmentation_mask)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                    
                    print("Applied GraphCut text segmentation")
                except Exception as e:
                    print(f"Error applying GraphCut segmentation: {e}")
                    # Continue with original image if segmentation fails
                    pass
        
        return img
    
    def extract_text(self, image, preprocessing_methods=None):
        """
        Extract text from an image using OCR.
        
        Args:
            image: PIL Image, numpy array, or file path
            preprocessing_methods (list): List of preprocessing methods to apply
        
        Returns:
            Extracted text (str)
        """
        # Handle different input types
        if isinstance(image, str):
            # Image path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image from path: {image}")
        elif isinstance(image, Image.Image):
            # PIL Image
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # Numpy array
            img = image.copy()
        else:
            raise TypeError("Unsupported image type. Expected PIL Image, numpy array, or file path.")
        
        # Apply preprocessing
        processed_img = self.preprocess_image(img, preprocessing_methods)
        
        # Convert to PIL Image for pytesseract
        if len(processed_img.shape) == 2:  # Grayscale
            pil_img = Image.fromarray(processed_img)
        else:  # Color
            pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        
        # Extract text
        try:
            text = pytesseract.image_to_string(pil_img, lang=self.lang, config=self.config)
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            
            # Try an alternative approach: save to temporary file and process
            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_filename = temp_file.name
                temp_file.close()
                
                cv2.imwrite(temp_filename, processed_img)
                text = pytesseract.image_to_string(temp_filename, lang=self.lang, config=self.config)
                
                # Clean up temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                return text
            except Exception as e2:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                raise Exception(f"OCR Failed. Original error: {e}, Secondary error: {e2}")
    
    def extract_text_with_bounding_boxes(self, image, preprocessing_methods=None):
        """
        Extract text with bounding box information.
        
        Args:
            image: PIL Image, numpy array, or file path
            preprocessing_methods (list): List of preprocessing methods to apply
        
        Returns:
            Dictionary containing text, bounding boxes, and confidence scores
        """
        # Handle different input types
        if isinstance(image, str):
            # Image path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image from path: {image}")
        elif isinstance(image, Image.Image):
            # PIL Image
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # Numpy array
            img = image.copy()
        else:
            raise TypeError("Unsupported image type. Expected PIL Image, numpy array, or file path.")
        
        # Apply preprocessing
        processed_img = self.preprocess_image(img, preprocessing_methods)
        
        # Extract text and bounding boxes
        try:
            d = pytesseract.image_to_data(processed_img, lang=self.lang, config=self.config, 
                                        output_type=pytesseract.Output.DICT)
            
            result = {
                'text': [],
                'boxes': [],
                'confidence': []
            }
            
            n_boxes = len(d['text'])
            for i in range(n_boxes):
                # Filter empty text and low confidence
                if int(d['conf'][i]) > 0 and d['text'][i].strip():
                    result['text'].append(d['text'][i])
                    result['boxes'].append((d['left'][i], d['top'][i], 
                                           d['left'][i] + d['width'][i], 
                                           d['top'][i] + d['height'][i]))
                    result['confidence'].append(d['conf'][i])
            
            return result
        
        except Exception as e:
            print(f"OCR Error: {e}")
            return {'text': [], 'boxes': [], 'confidence': []}
    
    def extract_text_from_regions(self, image, regions, preprocessing_methods=None):
        """
        Extract text from specific regions of an image.
        
        Args:
            image: PIL Image, numpy array, or file path
            regions (list): List of regions as (x1, y1, x2, y2) tuples
            preprocessing_methods (list): List of preprocessing methods to apply
        
        Returns:
            List of extracted text from each region
        """
        # Handle different input types
        if isinstance(image, str):
            # Image path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image from path: {image}")
        elif isinstance(image, Image.Image):
            # PIL Image
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # Numpy array
            img = image.copy()
        else:
            raise TypeError("Unsupported image type. Expected PIL Image, numpy array, or file path.")
        
        results = []
        
        for region in regions:
            x1, y1, x2, y2 = region
            
            # Extract region
            region_img = img[y1:y2, x1:x2]
            
            # Apply preprocessing
            processed_region = self.preprocess_image(region_img, preprocessing_methods)
            
            # Extract text
            try:
                text = pytesseract.image_to_string(processed_region, lang=self.lang, config=self.config)
                results.append(text.strip())
            except Exception as e:
                print(f"OCR Error in region {region}: {e}")
                results.append("")
        
        return results
    
    def extract_text_with_graphcut(self, image, additional_methods=None):
        """
        Extract text using GraphCut-based segmentation followed by OCR.
        
        Args:
            image: PIL Image, numpy array, or file path
            additional_methods (list): Additional preprocessing methods to apply after GraphCut
        
        Returns:
            Extracted text (str)
        """
        # Use graphcut as the first preprocessing method
        methods = ['graphcut']
        
        # Add additional preprocessing methods if specified
        if additional_methods:
            methods.extend(additional_methods)
        else:
            # Default additional methods for text cleanup
            methods.extend(['grayscale', 'threshold'])
        
        return self.extract_text(image, methods)
    
    def try_all_preprocessing_combinations(self, image, top_n=3):
        """
        Try different preprocessing combinations and return the top N results.
        
        Args:
            image: PIL Image, numpy array, or file path
            top_n (int): Number of top results to return
        
        Returns:
            List of tuples (text, preprocessing_methods, text_length)
        """
        preprocessing_options = [
            ['grayscale', 'threshold'],
            ['grayscale', 'adaptive_threshold'],
            ['grayscale', 'contrast', 'threshold'],
            ['grayscale', 'sharpen', 'threshold'],
            ['grayscale', 'denoise', 'threshold'],
            ['grayscale', 'deskew', 'threshold'],
            ['grayscale', 'contrast', 'sharpen', 'threshold'],
            ['grayscale', 'denoise', 'sharpen', 'threshold'],
            ['grayscale', 'erode', 'threshold'],
            ['grayscale', 'dilate', 'threshold'],
            # Adding GraphCut-based options
            ['graphcut', 'grayscale', 'threshold'],
            ['graphcut', 'grayscale', 'adaptive_threshold'],
            ['graphcut', 'grayscale', 'sharpen', 'threshold']
        ]
        
        results = []
        
        for methods in preprocessing_options:
            try:
                text = self.extract_text(image, methods)
                # Use text length as a simple heuristic for OCR quality
                text_length = len(text.strip())
                results.append((text, methods, text_length))
                print(f"Preprocessing methods {methods}: extracted {text_length} characters")
            except Exception as e:
                print(f"Error with preprocessing methods {methods}: {e}")
        
        # Sort by text length (more text is generally better)
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:top_n] 