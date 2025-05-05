# Enhanced Image Editor with Advanced GrabCut Segmentation

This application provides an interactive interface for image segmentation and editing with advanced capabilities. It leverages a sophisticated implementation of the GrabCut algorithm for interactive foreground extraction, allowing users to easily separate objects from their background.

## Features

- **Advanced GrabCut-based Segmentation**: High-quality foreground extraction using an implementation based on the original ["GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts"](https://doi.org/10.1145/1015706.1015720) paper
- **Interactive Brush Tools**: Refine segmentation with foreground and background brushes
- **Multi-threaded Processing**: Parallel computation for better performance on multi-core systems
- **Enhanced OCR Text Extraction**: Extract text from images with multiple preprocessing techniques for improved accuracy
- **Multi-language Support**: OCR capabilities in multiple languages (English, French, German, Spanish, and more)
- **Background Removal**: Create images with transparent backgrounds
- **Modern User Interface**: Intuitive controls with real-time visual feedback

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- PyMaxflow
- scikit-learn
- Pillow
- matplotlib
- pytesseract (for OCR functionality)
- Tesseract OCR engine (for OCR functionality)

## Installation

1. Clone or download this repository:
   ```
   git clone <repository-url>
   cd enhanced-image-editor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. For OCR functionality:
   - Install Tesseract OCR from https://github.com/tesseract-ocr/tesseract
   - Make sure Tesseract is in your system PATH or configure pytesseract to point to your Tesseract installation

## Usage

1. Run the application:
   ```
   python run_editor.py
   ```

2. Basic workflow:
   - Open an image
   - Use the rectangle tool to select the initial region of interest
   - Refine the selection using foreground (green) and background (red) brushes
   - Click "Apply GrabCut" to perform segmentation
   - Continue refining as needed
   - Save the result as a PNG with transparency

3. For text extraction:
   - Select the text area using the segmentation tools
   - Choose the appropriate language
   - Select preprocessing method (auto, basic, or advanced)
   - Click "Extract Text (OCR)"
   - Copy or save the extracted text

4. Standalone OCR demo:
   ```
   python ocr_demo.py <image_path> [options]
   ```
   Options:
   - `--lang`: Language code (default: eng)
   - `--method`: Preprocessing method (auto, basic, custom)
   - `--custom`: Custom preprocessing methods (comma-separated)
   - `--save-images`: Save preprocessed images
   - `--psm`: Page Segmentation Mode (0-13)

## Implementation Details

This implementation features three main components:

1. **GrabCut Algorithm Core**: A sophisticated implementation based on:
   - Gaussian Mixture Models (GMMs) for color modeling
   - Graph cuts with maxflow/mincut optimization
   - Multi-threaded processing for performance

2. **Enhanced OCR System**: Advanced text extraction with:
   - Multiple preprocessing techniques (grayscale, thresholding, denoising, contrast enhancement, etc.)
   - Automatic selection of optimal preprocessing methods
   - Region-specific text extraction
   - Text extraction with bounding boxes

3. **User Interface**:
   - Interactive canvas with drawing tools
   - Real-time visual feedback
   - Progress indicators for computationally intensive operations

## Files

- `run_editor.py`: Main script to launch the application
- `enhanced_image_editor.py`: The main application GUI and interaction logic
- `grabcut_core_fromcv2.py`: Core implementation of the GrabCut algorithm
- `grabcut_utils_fromcv2.py`: Utility functions for the GrabCut algorithm
- `ocr_utils.py`: Advanced OCR processing utilities
- `opencv_graphcut.py`: Text-focused segmentation using graph cuts
- `ocr_demo.py`: Standalone script for testing OCR with different preprocessing methods

## OCR Preprocessing Methods

The enhanced OCR system supports several preprocessing methods that can be combined for optimal results:

- **grayscale**: Convert image to grayscale
- **threshold**: Apply Otsu's thresholding
- **adaptive_threshold**: Apply adaptive Gaussian thresholding
- **denoise**: Apply non-local means denoising
- **contrast**: Enhance image contrast
- **sharpen**: Apply sharpening filter
- **deskew**: Correct text skew angle
- **erode**: Apply morphological erosion
- **dilate**: Apply morphological dilation

The 'auto' mode automatically tests combinations of these methods to find the one that extracts the most text.

## Acknowledgments

This implementation is based on the original GrabCut paper and takes inspiration from OpenCV's implementation:

- Carsten Rother, Vladimir Kolmogorov, and Andrew Blake. "GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts." ACM Transactions on Graphics (SIGGRAPH), 2004.
- OpenCV's GrabCut implementation
- The OCR implementation is based on Tesseract OCR engine

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Note: The PyMaxflow package requires Visual C++ build tools on Windows. If you encounter installation issues, please refer to the PyMaxflow documentation for more information.* 