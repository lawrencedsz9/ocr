import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pytesseract
import os

# Configure Tesseract path - modify as needed for your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ImageTextExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Image Editor")
        self.root.minsize(1000, 700)
        
        # Configure the grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = Frame(self.root)
        self.main_frame.pack(fill=BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=0)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Create left panel for controls
        self.left_panel = Frame(self.main_frame, width=250)
        self.left_panel.pack(side=LEFT, fill=Y, padx=5, pady=5)
        
        # Create canvas frame
        self.canvas_frame = Frame(self.main_frame)
        self.canvas_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbars to canvas frame
        self.h_scrollbar = Scrollbar(self.canvas_frame, orient=HORIZONTAL)
        self.h_scrollbar.pack(side=BOTTOM, fill=X)
        
        self.v_scrollbar = Scrollbar(self.canvas_frame)
        self.v_scrollbar.pack(side=RIGHT, fill=Y)
        
        # Create canvas with scrollbars
        self.canvas = Canvas(self.canvas_frame, 
                          xscrollcommand=self.h_scrollbar.set,
                          yscrollcommand=self.v_scrollbar.set,
                          bg="gray80",
                          cursor="cross")
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Connect scrollbars to canvas
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        # Image variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.tk_image = None
        self.extracted_text = ""
        self.processing_thread = None
        self.is_processing = False
        
        # Selection rectangle variables
        self.rect_id = None
        self.start_x = self.start_y = 0
        self.rect = (0, 0, 0, 0)  # (x, y, width, height)
        
        # Setup UI components
        self._create_ui()
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Display welcome message
        self.show_welcome()
    
    def _create_ui(self):
        """Set up all UI components"""
        Label(self.left_panel, text="OCR Image Editor", font=("Arial", 14, "bold")).pack(pady=10)
        
        # File operations frame
        file_frame = LabelFrame(self.left_panel, text="File Operations")
        file_frame.pack(fill=X, padx=5, pady=5, ipady=5)
        Button(file_frame, text="Open Image", command=self.open_image).pack(fill=X, padx=5, pady=2)
        Button(file_frame, text="Save Result", command=self.save_result).pack(fill=X, padx=5, pady=2)
        
        # Selection frame
        selection_frame = LabelFrame(self.left_panel, text="Selection")
        selection_frame.pack(fill=X, padx=5, pady=5, ipady=5)
        Label(selection_frame, text="Draw a rectangle around text").pack(anchor=W, padx=5, pady=2)
        Button(selection_frame, text="Reset Image", command=self.reset_image).pack(fill=X, padx=5, pady=2)
        Button(selection_frame, text="Run GrabCut on Selection", command=self.run_grabcut).pack(fill=X, padx=3, pady=2)
        
        # OCR frame
        ocr_frame = LabelFrame(self.left_panel, text="OCR Tools")
        ocr_frame.pack(fill=X, padx=5, pady=5, ipady=5)
        
        # Language selection
        lang_frame = Frame(ocr_frame)
        lang_frame.pack(fill=X, padx=5, pady=2)
        Label(lang_frame, text="Language:").pack(side=LEFT)
        self.lang_var = StringVar(value="eng")
        lang_combo = OptionMenu(lang_frame, self.lang_var, "eng", "fra", "deu", "spa")
        lang_combo.pack(side=RIGHT, fill=X, expand=True, padx=5)
        
        # Processing options frame
        processing_frame = Frame(ocr_frame)
        processing_frame.pack(fill=X, padx=5, pady=5)
        
        # OCR Mode
        Label(processing_frame, text="OCR Mode:").pack(anchor=W)
        self.ocr_mode_var = StringVar(value="auto")
        modes = [
            ("Automatic", "auto"),
            ("High Contrast", "contrast"),
            ("Text Cleanup", "cleanup"),
            ("Small Text", "small")
        ]
        
        for text, mode in modes:
            Radiobutton(processing_frame, text=text, value=mode, 
                       variable=self.ocr_mode_var).pack(anchor=W, padx=15)
        
        # OCR buttons
        Button(ocr_frame, text="Extract Text (OCR)", command=self.extract_text).pack(fill=X, padx=5, pady=2)
        Button(ocr_frame, text="Copy Text", command=self.copy_text).pack(fill=X, padx=5, pady=2)
        Button(ocr_frame, text="Save Text", command=self.save_text).pack(fill=X, padx=5, pady=2)
        
        # Status frame
        status_frame = LabelFrame(self.left_panel, text="Status")
        status_frame.pack(fill=X, padx=5, pady=5, ipady=5)
        
        self.status_var = StringVar(value="Ready")
        Label(status_frame, textvariable=self.status_var).pack(fill=X, padx=5, pady=2)
        
        # OCR Text display
        ocr_display_frame = LabelFrame(self.left_panel, text="Extracted Text")
        ocr_display_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        self.text_display = scrolledtext.ScrolledText(ocr_display_frame, wrap=WORD, height=10)
        self.text_display.pack(fill=BOTH, expand=True, padx=5, pady=5)
    
    def show_welcome(self):
        """Display welcome message on canvas"""
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="OCR Image Editor", 
                             font=("Arial", 24, "bold"), fill="navy")
        self.canvas.create_text(400, 350, 
                              text="Open an image to get started\n" + 
                              "Draw a rectangle around text\n" +
                              "Then use OCR to extract text",
                              font=("Arial", 12), fill="black")
    
    def open_image(self):
        """Open an image file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), 
                      ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.image_path = file_path
            
            # Load the image
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", f"Could not read the image: {file_path}")
                return
            
            # Save a copy for display
            self.display_image = self.original_image.copy()
            
            # Display the image
            self.update_display()
            
            # Clear any existing selection
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None
            self.rect = (0, 0, 0, 0)
            
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error opening image: {str(e)}")
    
    def update_display(self):
        """Update the canvas with the current display image"""
        if self.display_image is not None:
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img_rgb)
            
            # Create PhotoImage
            self.tk_image = ImageTk.PhotoImage(pil_img)
            
            # Clear canvas and display new image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)
            
            # Set canvas scrollregion to image size
            h, w = self.display_image.shape[:2]
            self.canvas.config(scrollregion=(0, 0, w, h))
            
            # Redraw selection rectangle if it exists
            if hasattr(self, 'rect') and self.rect != (0, 0, 0, 0):
                x, y, width, height = self.rect
                self.rect_id = self.canvas.create_rectangle(
                    x, y, x+width, y+height, 
                    outline="red", width=2, dash=(4, 4))
    
    def on_mouse_down(self, event):
        """Handle mouse button press"""
        if self.display_image is None:
            return
            
        # Get the mouse position relative to the canvas
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Save start coordinates
        self.start_x = x
        self.start_y = y
        
        # Clear any existing rectangle
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            
        # Create a new rectangle
        self.rect_id = self.canvas.create_rectangle(x, y, x, y, outline="red", width=2)
        
        # Update status
        self.status_var.set(f"Started selection at ({int(x)}, {int(y)})")
    
    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        if not self.rect_id or self.display_image is None:
            return
            
        # Get the current mouse position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Update the rectangle coordinates
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, x, y)
        
        # Update status
        width = abs(x - self.start_x)
        height = abs(y - self.start_y)
        self.status_var.set(f"Selection size: {int(width)}x{int(height)}")
    
    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if not self.rect_id or self.display_image is None:
            return
            
        # Get final position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Calculate rectangle coordinates (x, y, width, height)
        x1 = min(self.start_x, x)
        y1 = min(self.start_y, y)
        width = abs(x - self.start_x)
        height = abs(y - self.start_y)
        
        # Check for minimum size
        if width < 5 or height < 5:
            self.status_var.set("Selection too small, please try again")
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            return
        
        # Ensure coordinates are within image bounds
        h, w = self.display_image.shape[:2]
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        width = min(width, w - x1)
        height = min(height, h - y1)
        
        # Store the rectangle for later use
        self.rect = (int(x1), int(y1), int(width), int(height))
        
        # Update the canvas rectangle to reflect bounds checking
        self.canvas.coords(self.rect_id, x1, y1, x1 + width, y1 + height)
        
        # Update status
        self.status_var.set(f"Selection: ({int(x1)}, {int(y1)}, {int(width)}x{int(height)})")
    
    def reset_image(self):
        """Reset to the original image and clear any selection"""
        if self.original_image is not None:
            # Reset display image to original
            self.display_image = self.original_image.copy()
            
            # Update the display
            self.update_display()
            
            # Clear any selection
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None
            self.rect = (0, 0, 0, 0)
            
            self.status_var.set("Reset to original image")
    
    def run_grabcut(self):
        """Run GrabCut segmentation on the selected region with optimal accuracy for text"""
        if self.display_image is None or self.display_image.size == 0:
            messagebox.showinfo("Info", "Please open an image first")
            return

        if self.rect == (0, 0, 0, 0):
            messagebox.showinfo("Info", "Please select a region first")
            return

        try:
            original = self.display_image.copy()
            x, y, w, h = self.rect
            h_img, w_img = self.display_image.shape[:2]

            # Ensure valid rectangle size
            if w < 10 or h < 10:
                messagebox.showinfo("Info", "Selection too small. Please select a larger area.")
                return

            # Ensure selection is within image bounds
            if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
                messagebox.showinfo("Info", "Selection is out of image bounds. Please select a valid region.")
                return

            # Create mask for GrabCut
            mask = np.zeros(self.display_image.shape[:2], np.uint8)
            border = 5
            # Clamp border region to image bounds
            x0 = max(0, x - border)
            y0 = max(0, y - border)
            x1 = min(w_img, x + w + border)
            y1 = min(h_img, y + h + border)
            mask[:] = cv2.GC_PR_BGD
            mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
            mask[y:y+h, x:x+w] = cv2.GC_FGD

            # Check that mask has both foreground and background
            if np.count_nonzero(mask == cv2.GC_FGD) == 0 or np.count_nonzero(mask == cv2.GC_PR_BGD) == 0:
                messagebox.showinfo("Info", "Selection does not contain enough foreground or background. Please select a different region.")
                return

            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            self.status_var.set("Running GrabCut segmentation...")
            rect = (x, y, w, h)
            cv2.grabCut(self.display_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

            # Additional refinement iterations
            for _ in range(5):
                cv2.grabCut(self.display_image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                temp_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
                kernel = np.ones((3, 3), np.uint8)
                temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask[temp_mask == 255] = cv2.GC_FGD
                mask[temp_mask == 0] = cv2.GC_BGD

            result_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result_mask, connectivity=8)
            min_size = (w * h) // 200
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    result_mask[labels == i] = 0

            rgba = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = result_mask
            white_bg = np.ones_like(original) * 255
            segmented = cv2.bitwise_and(original, original, mask=result_mask)
            inv_mask = cv2.bitwise_not(result_mask)
            background = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
            result = cv2.add(segmented, background)
            black_bg = np.zeros_like(original)
            black_bg_result = cv2.add(segmented, cv2.bitwise_and(black_bg, black_bg, mask=inv_mask))
            option1 = rgba
            option2 = result
            option3 = black_bg_result
            self.display_image = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
            self.update_display()
            self.segmentation_options = {
                "transparent": option1,
                "white": option2,
                "black": option3,
                "original": original
            }
            visibility_choice = messagebox.askquestion("Segmentation Result", 
                                              "Is the segmented result visible enough?\n\n" +
                                              "Select 'No' to see other background options.")
            if visibility_choice == 'no':
                self.display_image = option2.copy()
                self.update_display()
                bg_choice = messagebox.askquestion("Background Options", 
                                                "Is this white background better?\n\n" +
                                                "Select 'No' to try black background.")
                if bg_choice == 'no':
                    self.display_image = option3.copy()
                    self.update_display()
            self.status_var.set("GrabCut segmentation completed")
        except Exception as e:
            messagebox.showerror("Error", f"Error during segmentation: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def extract_text(self):
        """Extract text from the image"""
        if self.display_image is None:
            messagebox.showinfo("Info", "Please open an image first")
            return
        
        try:
            # Ask if we should use just the selection or the whole image
            use_selection = True
            if self.rect == (0, 0, 0, 0):
                use_selection = False
            else:
                use_selection = messagebox.askyesno("Extract Text", 
                                                "Extract text only from the selected region?\n\n" +
                                                "Yes = Use the selected region\n" +
                                                "No = Use the entire image")
            
            # Get image region to process
            if use_selection:
                x, y, w, h = self.rect
                image_region = self.display_image[y:y+h, x:x+w]
            else:
                image_region = self.display_image
            
            # Get language and OCR mode
            lang = self.lang_var.get()
            ocr_mode = self.ocr_mode_var.get()
            
            # Process based on selected mode
            if ocr_mode == "auto":
                # Try multiple methods
                results = self.try_multiple_ocr_methods(image_region, lang)
                if results:
                    best_text = results[0]
                    self.update_text_display(best_text)
                    return
                else:
                    # Default to grayscale + threshold if no methods worked
                    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
                    _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            elif ocr_mode == "contrast":
                # Enhanced contrast
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(gray)
                _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif ocr_mode == "cleanup":
                # Denoise and clean
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
                processed = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                processed = cv2.adaptiveThreshold(
                    processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                kernel = np.ones((1, 1), np.uint8)
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
                
            elif ocr_mode == "small":
                # Resize to improve clarity for small text (2x)
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
                processed = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                processed = cv2.filter2D(processed, -1, kernel)
                
            else:  # Default processing
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
                _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save enhanced image for debugging
            cv2.imwrite("ocr_enhanced.png", processed)
            
            # Perform OCR
            config = '--oem 3 --psm 6'  # Assume a single uniform block of text
            text = pytesseract.image_to_string(processed, lang=lang, config=config)
            
            # If no text was found, try with another PSM mode
            if not text.strip():
                config = '--oem 3 --psm 11'  # Sparse text
                text = pytesseract.image_to_string(processed, lang=lang, config=config)
            
            # Format result
            if text.strip():
                extracted_text = "Extracted Text:\n\n" + text
            else:
                extracted_text = "No text could be extracted from the image.\n\n" + \
                              "Tips:\n" + \
                              "- Try a different OCR mode\n" + \
                              "- Make sure the text is clear and in focus\n" + \
                              "- Try using a different language if the text is not in English"
            
            # Update text display
            self.update_text_display(extracted_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"OCR processing error: {str(e)}\n\nPlease make sure Tesseract is properly installed.")
    
    def try_multiple_ocr_methods(self, image, lang):
        """Try multiple OCR processing methods and return the best results"""
        results = []
        
        # Method 1: Basic thresholding
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text1 = pytesseract.image_to_string(binary, lang=lang, config='--psm 6')
            if text1.strip():
                results.append(text1)
        except Exception:
            pass
            
        # Method 2: CLAHE contrast enhancement
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            text2 = pytesseract.image_to_string(enhanced, lang=lang, config='--psm 6')
            if text2.strip():
                results.append(text2)
        except Exception:
            pass
            
        # Method 3: Resize + adaptive thresholding
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            adaptive = cv2.adaptiveThreshold(
                resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            text3 = pytesseract.image_to_string(adaptive, lang=lang, config='--psm 6')
            if text3.strip():
                results.append(text3)
        except Exception:
            pass
            
        # Method 4: Denoising + morphology
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            kernel = np.ones((1, 1), np.uint8)
            morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            text4 = pytesseract.image_to_string(morphed, lang=lang, config='--psm 6')
            if text4.strip():
                results.append(text4)
        except Exception:
            pass
        
        # Sort results by text length (longer text usually means better recognition)
        results.sort(key=len, reverse=True)
        
        return results
    
    def update_text_display(self, text):
        """Update text display with extracted text"""
        self.text_display.delete(1.0, END)
        self.text_display.insert(END, text)
        self.extracted_text = text
        self.status_var.set("OCR completed")
    
    def copy_text(self):
        """Copy extracted text to clipboard"""
        if self.extracted_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.extracted_text)
            self.status_var.set("Text copied to clipboard")
        else:
            messagebox.showinfo("Info", "No text to copy")
    
    def save_text(self):
        """Save extracted text to file"""
        if not self.extracted_text:
            messagebox.showinfo("Info", "No text to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.extracted_text)
                self.status_var.set(f"Text saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving text: {str(e)}")
    
    def save_result(self):
        """Save the current image"""
        if self.display_image is None:
            messagebox.showinfo("Info", "Please open an image first")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Save image
                cv2.imwrite(file_path, self.display_image)
                self.status_var.set(f"Image saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {str(e)}")

# Run the GUI
if __name__ == "__main__":
    root = Tk()
    app = ImageTextExtractor(root)
    root.mainloop()
