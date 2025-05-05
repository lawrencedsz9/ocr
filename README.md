# OCR Image Editor

A Python application that combines image segmentation with optical character recognition (OCR). While it excels at text extraction, it can process any type of image to separate foreground objects from backgrounds.

## Features

- **Image Loading**: Open and process standard image formats (JPG, PNG, BMP)
- **Selection Tools**: Draw rectangles to select specific areas of text or objects
- **GrabCut Segmentation**: Precisely extract foreground elements from background for any image type
- **OCR Text Extraction**: Convert image text to editable text using Tesseract OCR
- **Multiple Languages**: Support for English, French, German, and Spanish
- **Processing Modes**:
  - Automatic: Tries multiple methods for optimal results
  - High Contrast: Enhances visibility
  - Text Cleanup: Reduces noise and improves clarity
  - Small Text: Optimized for extracting small details

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Pytesseract
- Tesseract OCR engine installed on your system

## Installation

1. Install Tesseract OCR engine:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Update the Tesseract path in `segment.py` if necessary:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

## Usage

1. Run the application:
   ```
   python segment.py
   ```

2. **Loading Images**:
   - Click "Open Image" to load an image file

3. **General Image Segmentation**:
   - Draw a rectangle around any object you want to extract
   - Click "Run GrabCut on Selection" to separate the object from background
   - Select visibility options if needed
   - Save the processed image using "Save Result"

4. **Text Extraction (Optional)**:
   - After segmentation, click "Extract Text (OCR)" to perform text recognition
   - Use "Copy Text" or "Save Text" to use the extracted content

5. **Adjusting Results**:
   - Select different processing modes for optimal results
   - Change languages for non-English text
   - Use "Reset Image" to start over

## How GrabCut Works

The GrabCut segmentation algorithm separates foreground from background using these steps:

1. User provides initial hint by drawing a rectangle around the object of interest
2. Algorithm builds statistical models of foreground and background colors and textures
3. Multiple iterations refine the segmentation boundary using graph cuts optimization
4. Morphological operations clean up the result by removing noise and filling gaps
5. Visibility enhancements improve the final output for better visualization

## Applications

This tool can be used for various image processing tasks:

- **Text Extraction**: Isolate text from complex backgrounds before OCR
- **Object Isolation**: Extract objects from images for further analysis or editing
- **Background Removal**: Create transparent backgrounds for product photos
- **Image Enhancement**: Focus on specific image elements by removing distractions
- **Content Preparation**: Prepare images for presentations, documents, or websites

## Tips for Best Results

- Ensure good lighting and contrast in your images
- Select areas with clear boundaries between foreground and background
- For complex objects, make sure your selection rectangle contains the entire object
- Try different visibility options if the segmented result is not clear
- For text extraction, use the OCR modes appropriate to your text type
- For text on complex backgrounds, always use GrabCut segmentation first

## Acknowledgments

The OCR implementation is based on the Tesseract OCR engine.

