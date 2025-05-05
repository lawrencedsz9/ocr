# OCR Image Editor

This application provides a simple and effective interface for extracting text from images using OCR (Optical Character Recognition). It offers multiple processing modes and options to enhance text extraction quality.

## Features

- **Enhanced OCR Text Extraction**: Extract text from images with multiple preprocessing techniques for improved accuracy
- **Multiple Processing Modes**: Choose from different processing modes optimized for various text types:
  - Automatic: Tries multiple methods and selects the best result
  - High Contrast: Enhances text contrast for better recognition
  - Text Cleanup: Removes noise and enhances text clarity
  - Small Text: Optimized for extracting small text
- **Multi-language Support**: OCR capabilities in multiple languages (English, French, German, Spanish, and more)
- **Modern User Interface**: Intuitive controls with visual feedback
- **Export Options**: Copy or save extracted text to file

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Pillow
- matplotlib
- pytesseract (for OCR functionality)
- Tesseract OCR engine

## Installation

1. Clone or download this repository:
   ```
   git clone <repository-url>
   cd ocr-image-editor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - Download and install from https://github.com/tesseract-ocr/tesseract
   - Make sure Tesseract is in your system PATH or configure pytesseract to point to your Tesseract installation
   - For Windows users: The default path is `C:\Program Files\Tesseract-OCR\tesseract.exe` (you may need to adjust this in the code)

## Usage

1. Run the application:
   ```
   python enhanced_image_editor.py
   ```

2. Basic workflow:
   - Open an image using the "Open Image" button
   - Select the OCR language (default: English)
   - Choose a processing mode based on your image type
   - Click "Extract Text (OCR)" to perform text recognition
   - Copy or save the extracted text

3. OCR Processing Modes:
   - **Automatic**: Tries multiple methods and selects the best result
   - **High Contrast**: Best for images with low contrast between text and background
   - **Text Cleanup**: Best for noisy images, removes artifacts and enhances text
   - **Small Text**: Optimized for extracting small or fine text

## OCR Preprocessing Methods

The OCR system applies various preprocessing methods depending on the selected mode:

- **Basic Thresholding**: Simple binarization using Otsu's method
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for improving contrast
- **Adaptive Thresholding**: Local adaptive thresholding for varying backgrounds
- **Denoising**: Non-local means denoising to remove noise
- **Morphological Operations**: Clean up text regions
- **Image Upscaling**: Improves recognition of small text

The 'auto' mode automatically tests multiple methods to find the one that extracts the most text.

## Acknowledgments

The OCR implementation is based on the Tesseract OCR engine.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 