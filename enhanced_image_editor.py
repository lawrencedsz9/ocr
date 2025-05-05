"""
Enhanced Image Editor with Advanced GrabCut Segmentation

Features:
- Interactive GrabCut-based image segmentation
- Brush tools for refining foreground/background
- OCR text extraction with multiple processing modes
- Background removal with alpha channel support
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from grabcut_utils_fromcv2 import GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD
from grabcut_core_fromcv2 import interactive_grabcut
import threading
import time
from ocr_utils import OCRProcessor
from opencv_graphcut import TextSegmentationGrabCut

# Configure Tesseract path - modify as needed for your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class EnhancedImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Image Editor")
        self.root.minsize(1000, 700)
        
        # Configure the grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(0, weight=0)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Create left panel for controls
        self.left_panel = ttk.Frame(self.main_frame, width=200)
        self.left_panel.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        
        # Create canvas frame
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)
        
        # Create scrollable canvas
        self.canvas_container = ttk.Frame(self.canvas_frame)
        self.canvas_container.grid(row=0, column=0, sticky="nsew")
        self.canvas_container.columnconfigure(0, weight=1)
        self.canvas_container.rowconfigure(0, weight=1)
        
        # Add scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(self.canvas_container, 
                               xscrollcommand=self.h_scrollbar.set,
                               yscrollcommand=self.v_scrollbar.set,
                               bg="gray80")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Connect scrollbars to canvas
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.tk_image = None
        self.mask = None
        self.drawing = False
        self.last_x, self.last_y = 0, 0
        self.rect_start_x, self.rect_start_y = 0, 0
        self.rect_end_x, self.rect_end_y = 0, 0
        self.rect_id = None
        self.mode = "rectangle"
        self.brush_size = 10
        self.extracted_text = ""
        self.processing_thread = None
        self.is_processing = False
        
        # Initialize OCR processor
        self.ocr_processor = OCRProcessor(tesseract_cmd=pytesseract.pytesseract.tesseract_cmd)
        
        # Setup UI components
        self._create_ui()
        
        # Bind canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Display welcome message
        self.show_welcome()
    
    def _create_ui(self):
        """Set up all UI components"""
        ttk.Label(self.left_panel, text="Image Editor", font=("Arial", 14, "bold")).pack(pady=10)
        
        # File operations frame
        file_frame = ttk.LabelFrame(self.left_panel, text="File Operations")
        file_frame.pack(fill="x", padx=5, pady=5, ipady=5)
        ttk.Button(file_frame, text="Open Image", command=self.open_image).pack(fill="x", padx=5, pady=2)
        ttk.Button(file_frame, text="Save Result", command=self.save_result).pack(fill="x", padx=5, pady=2)
        
        # Segmentation frame
        seg_frame = ttk.LabelFrame(self.left_panel, text="Segmentation Tools")
        seg_frame.pack(fill="x", padx=5, pady=5, ipady=5)
        
        # Mode selection
        self.mode_var = tk.StringVar(value="rectangle")
        ttk.Radiobutton(seg_frame, text="Rectangle Selection", value="rectangle", 
                        variable=self.mode_var, command=self.set_mode).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(seg_frame, text="Foreground Brush", value="foreground", 
                        variable=self.mode_var, command=self.set_mode).pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(seg_frame, text="Background Brush", value="background", 
                        variable=self.mode_var, command=self.set_mode).pack(anchor="w", padx=5, pady=2)
        
        # Brush size
        brush_frame = ttk.Frame(seg_frame)
        brush_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(brush_frame, text="Brush Size:").pack(side=tk.LEFT)
        self.brush_size_var = tk.IntVar(value=10)
        brush_scale = ttk.Scale(brush_frame, from_=1, to=50, orient=tk.HORIZONTAL, 
                               variable=self.brush_size_var, command=self.update_brush_size)
        brush_scale.pack(side=tk.RIGHT, fill="x", expand=True, padx=5)
        
        # GrabCut buttons
        ttk.Button(seg_frame, text="Apply GrabCut", command=self.apply_grabcut).pack(fill="x", padx=5, pady=2)
        ttk.Button(seg_frame, text="Reset Selection", command=self.reset_selection).pack(fill="x", padx=5, pady=2)
        
        # OCR frame
        ocr_frame = ttk.LabelFrame(self.left_panel, text="OCR Tools")
        ocr_frame.pack(fill="x", padx=5, pady=5, ipady=5)
        
        # Language selection
        lang_frame = ttk.Frame(ocr_frame)
        lang_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(lang_frame, text="Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="eng")
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, width=10)
        lang_combo['values'] = ('eng', 'fra', 'deu', 'spa')
        lang_combo.pack(side=tk.RIGHT, fill="x", expand=True, padx=5)
        
        # OCR description
        ttk.Label(ocr_frame, text="Enhanced OCR with automatic image clarity improvements", 
                 wraplength=200, justify="center").pack(padx=5, pady=5)
        
        # OCR buttons
        ttk.Button(ocr_frame, text="Extract Text (OCR)", command=self.extract_text).pack(fill="x", padx=5, pady=2)
        ttk.Button(ocr_frame, text="Copy Text", command=self.copy_text).pack(fill="x", padx=5, pady=2)
        ttk.Button(ocr_frame, text="Save Text", command=self.save_text).pack(fill="x", padx=5, pady=2)
        
        # Progress bar 
        self.progress_frame = ttk.LabelFrame(self.left_panel, text="Progress")
        self.progress_frame.pack(fill="x", padx=5, pady=5, ipady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, 
                                           length=100, mode='indeterminate', 
                                           variable=self.progress_var)
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.progress_frame, textvariable=self.status_var).pack(padx=5)
        
        # OCR Text display
        ocr_display_frame = ttk.LabelFrame(self.left_panel, text="Extracted Text")
        ocr_display_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.text_display = scrolledtext.ScrolledText(ocr_display_frame, wrap=tk.WORD, height=10)
        self.text_display.pack(fill="both", expand=True, padx=5, pady=5)
    
    def show_welcome(self):
        """Display welcome message on canvas"""
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="Enhanced Image Editor", 
                                font=("Arial", 24, "bold"), fill="navy")
        self.canvas.create_text(400, 350, 
                               text="Open an image to get started\n" + 
                               "Use rectangle tool to select initial region\n" + 
                               "Then use brush tools to refine selection",
                               font=("Arial", 12), fill="black")
    
    def open_image(self):
        """Open an image file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), 
                       ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.image_path = file_path
                
                # Load the image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    messagebox.showerror("Error", f"Could not read the image: {file_path}")
                    return
                
                # Resize large images for display
                h, w = self.original_image.shape[:2]
                if max(h, w) > 1500:
                    scale = 1500 / max(h, w)
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    self.original_image = cv2.resize(self.original_image, (new_w, new_h))
                
                # Initialize mask
                h, w = self.original_image.shape[:2]
                self.mask = np.zeros((h, w), dtype=np.uint8)
                self.mask[:] = GC_PR_BGD
                
                # Reset selection
                self.rect_start_x, self.rect_start_y = 0, 0
                self.rect_end_x, self.rect_end_y = 0, 0
                
                # Display the image
                self.display_image = self.original_image.copy()
                self.update_display()
                
                # Set canvas scrollregion
                h, w = self.original_image.shape[:2]
                self.canvas.config(scrollregion=(0, 0, w, h))
                
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
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            # Redraw rectangle if in rectangle mode
            if (self.mode == "rectangle" and 
                self.rect_start_x != self.rect_end_x and 
                self.rect_start_y != self.rect_end_y):
                self.rect_id = self.canvas.create_rectangle(
                    self.rect_start_x, self.rect_start_y, 
                    self.rect_end_x, self.rect_end_y, 
                    outline="yellow", width=2
                )
    
    def set_mode(self):
        """Set the current editing mode"""
        self.mode = self.mode_var.get()
        
        cursor_types = {
            "rectangle": "crosshair",
            "foreground": "pencil",
            "background": "dot"
        }
        
        self.canvas.config(cursor=cursor_types.get(self.mode, "arrow"))
        
        # Update status
        mode_descriptions = {
            "rectangle": "Draw a rectangle around the object",
            "foreground": "Mark foreground areas with brush",
            "background": "Mark background areas with brush"
        }
        self.status_var.set(mode_descriptions.get(self.mode, ""))
    
    def update_brush_size(self, event=None):
        """Update brush size from slider"""
        self.brush_size = self.brush_size_var.get()
    
    def on_mouse_down(self, event):
        """Handle mouse button press"""
        if self.original_image is None:
            return
        
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        
        if self.mode == "rectangle":
            self.rect_start_x, self.rect_start_y = event.x, event.y
            self.rect_end_x, self.rect_end_y = event.x, event.y
            
            # Remove existing rectangle
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            
            # Draw new rectangle
            self.rect_id = self.canvas.create_rectangle(
                self.rect_start_x, self.rect_start_y, 
                self.rect_end_x, self.rect_end_y, 
                outline="yellow", width=2
            )
    
    def on_mouse_move(self, event):
        """Handle mouse movement while button is pressed"""
        if not self.drawing or self.original_image is None:
            return
        
        if self.mode == "rectangle":
            self.rect_end_x, self.rect_end_y = event.x, event.y
            
            # Update rectangle
            if self.rect_id:
                self.canvas.coords(self.rect_id, 
                                   self.rect_start_x, self.rect_start_y, 
                                   self.rect_end_x, self.rect_end_y)
        
        elif self.mode in ["foreground", "background"]:
            x, y = event.x, event.y
            
            # Make sure coordinates are within image bounds
            h, w = self.original_image.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                # Draw on display image
                if self.mode == "foreground":
                    color = (0, 255, 0)  # Green for foreground
                    mask_value = GC_FGD
                else:
                    color = (0, 0, 255)  # Red for background
                    mask_value = GC_BGD
                
                # Draw line on the display image
                cv2.line(self.display_image, (self.last_x, self.last_y), (x, y), color, self.brush_size)
                
                # Draw circle at current position
                cv2.circle(self.display_image, (x, y), self.brush_size // 2, color, -1)
                
                # Update mask
                cv2.circle(self.mask, (x, y), self.brush_size // 2, mask_value, -1)
                
                # Update display
                self.update_display()
            
            self.last_x, self.last_y = x, y
    
    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if self.original_image is None:
            return
        
        self.drawing = False
        
        if self.mode == "rectangle":
            # Ensure rectangle has positive width and height
            x1 = min(self.rect_start_x, self.rect_end_x)
            y1 = min(self.rect_start_y, self.rect_end_y)
            x2 = max(self.rect_start_x, self.rect_end_x)
            y2 = max(self.rect_start_y, self.rect_end_y)
            
            self.rect_start_x, self.rect_start_y = x1, y1
            self.rect_end_x, self.rect_end_y = x2, y2
            
            # Make sure coordinates are within image bounds
            h, w = self.original_image.shape[:2]
            self.rect_start_x = max(0, min(self.rect_start_x, w-1))
            self.rect_start_y = max(0, min(self.rect_start_y, h-1))
            self.rect_end_x = max(0, min(self.rect_end_x, w-1))
            self.rect_end_y = max(0, min(self.rect_end_y, h-1))
            
            # Update rectangle
            if self.rect_id:
                self.canvas.coords(self.rect_id, 
                                  self.rect_start_x, self.rect_start_y, 
                                  self.rect_end_x, self.rect_end_y)
            
            # Update mask with rectangle
            self.mask[:] = GC_BGD  # Set all to background
            # Set rectangle area to probable foreground
            self.mask[self.rect_start_y:self.rect_end_y, 
                      self.rect_start_x:self.rect_end_x] = GC_PR_FGD
    
    def apply_grabcut(self):
        """Apply GrabCut algorithm on the image"""
        if self.original_image is None:
            messagebox.showinfo("Info", "Please open an image first")
            return
        
        if self.is_processing:
            messagebox.showinfo("Info", "Processing is already in progress")
            return
        
        # Check if a rectangle has been drawn or if brush tools have been used
        if self.mode == "rectangle" and (self.rect_start_x == self.rect_end_x or 
                                        self.rect_start_y == self.rect_end_y):
            messagebox.showinfo("Info", "Please select a region using the rectangle tool")
            return
        
        # Get rectangle coordinates
        rect = (
            self.rect_start_x, self.rect_start_y, 
            self.rect_end_x - self.rect_start_x, 
            self.rect_end_y - self.rect_start_y
        ) if self.mode == "rectangle" else None
        
        # Start processing in a separate thread
        self.is_processing = True
        self.progress_bar.start(10)
        self.status_var.set("Applying GrabCut algorithm...")
        
        self.processing_thread = threading.Thread(
            target=self.process_grabcut, 
            args=(rect,)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_grabcut(self, rect=None):
        """Process GrabCut in a separate thread"""
        try:
            # Apply GrabCut algorithm
            result_mask = interactive_grabcut(
                self.original_image, 
                rect=rect, 
                mask=self.mask.copy(),
                n_iter=5,
                gamma=50,
                max_threads=4
            )
            
            # Apply mask to the image
            mask_3d = np.zeros(self.original_image.shape, dtype=np.uint8)
            mask_3d[:, :, 0] = result_mask
            mask_3d[:, :, 1] = result_mask
            mask_3d[:, :, 2] = result_mask
            
            # Set background to red for visualization
            self.display_image = self.original_image.copy()
            red_bg = self.display_image.copy()
            red_bg[:, :, 0] = 0    # Blue channel
            red_bg[:, :, 1] = 0    # Green channel
            red_bg[:, :, 2] = 255  # Red channel
            
            # Create masked image (foreground from original, background as red)
            foreground = cv2.bitwise_and(self.original_image, mask_3d)
            background = cv2.bitwise_and(red_bg, cv2.bitwise_not(mask_3d))
            self.display_image = cv2.add(foreground, background)
            
            # Update mask from result
            _, binary_mask = cv2.threshold(result_mask, 128, 255, cv2.THRESH_BINARY)
            self.mask = np.zeros_like(self.mask)
            self.mask[binary_mask == 0] = GC_BGD
            self.mask[binary_mask == 255] = GC_FGD
            
            # Update display in the main thread
            self.root.after(0, self.update_display_after_processing)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error in GrabCut processing: {str(e)}"))
        finally:
            self.is_processing = False
            self.root.after(0, self.stop_progress)
    
    def update_display_after_processing(self):
        """Update display after processing"""
        self.update_display()
        self.status_var.set("GrabCut completed")
    
    def stop_progress(self):
        """Stop progress bar"""
        self.progress_bar.stop()
    
    def reset_selection(self):
        """Reset selection and mask"""
        if self.original_image is None:
            return
        
        # Reset mask
        h, w = self.original_image.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.mask[:] = GC_PR_BGD
        
        # Reset rectangle
        self.rect_start_x, self.rect_start_y = 0, 0
        self.rect_end_x, self.rect_end_y = 0, 0
        
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        
        # Reset display image
        self.display_image = self.original_image.copy()
        self.update_display()
        
        self.status_var.set("Selection reset")
    
    def extract_text(self):
        """Extract text from the segmented area"""
        if self.original_image is None:
            messagebox.showinfo("Info", "Please open an image first")
            return
        
        try:
            # Check if OCR can be performed
            if not self.is_ocr_available():
                messagebox.showwarning("Warning", "Tesseract OCR engine not found. Please install it and set the path.")
                return
            
            # Get language
            lang = self.lang_var.get()
            
            # Show waiting status
            self.progress_bar.start(10)
            self.status_var.set("Performing OCR...")
            
            # Create masked image for OCR - only use foreground areas
            mask_3d = np.zeros(self.original_image.shape, dtype=np.uint8)
            mask_3d[:, :, 0] = (self.mask == GC_FGD) | (self.mask == GC_PR_FGD)
            mask_3d[:, :, 1] = (self.mask == GC_FGD) | (self.mask == GC_PR_FGD)
            mask_3d[:, :, 2] = (self.mask == GC_FGD) | (self.mask == GC_PR_FGD)
            
            # Apply mask to focus only on selected foreground
            masked_img = cv2.bitwise_and(self.original_image, mask_3d * 255)
            
            # Run OCR in a separate thread
            self.is_processing = True
            self.processing_thread = threading.Thread(
                target=self.process_ocr,
                args=(masked_img, lang)
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in OCR processing: {str(e)}")
            self.stop_progress()
    
    def is_ocr_available(self):
        """Check if Tesseract OCR is available"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def process_ocr(self, image, lang):
        """Process OCR with image enhancement optimized for text recognition"""
        try:
            # Step 1: Save original masked image for reference
            cv2.imwrite("ocr_input_masked.png", image)
            
            # Step 2: Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Step 3: Resize to improve clarity for small text (2x)
            resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Step 4: Apply adaptive thresholding to handle uneven lighting
            adaptive = cv2.adaptiveThreshold(
                resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Step 5: Denoise to remove speckles and improve character clarity
            denoised = cv2.fastNlMeansDenoising(adaptive, None, 10, 7, 21)
            
            # Step 6: Morphological operations to enhance text
            kernel = np.ones((1, 1), np.uint8)
            morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
            # Save enhanced image for debugging
            cv2.imwrite("ocr_enhanced.png", morphed)
            
            # Step 7: Perform OCR on the enhanced image
            # Use a more appropriate PSM mode for text regions
            # PSM 6: Assume a single uniform block of text
            # OEM 3: Default, based on what is available
            config = '--oem 3 --psm 6'
            text = pytesseract.image_to_string(morphed, lang=lang, config=config)
            
            # If no text was found, try with another PSM mode
            if not text.strip():
                # PSM 11: Sparse text - Find as much text as possible with no specific order
                config = '--oem 3 --psm 11'
                text = pytesseract.image_to_string(morphed, lang=lang, config=config)
            
            # Format result
            if text.strip():
                extracted_text = "Extracted Text:\n\n" + text
            else:
                extracted_text = "No text could be extracted from the selected region.\n\n" + \
                               "Tips:\n" + \
                               "- Select an area with clearer text\n" + \
                               "- Make sure the text is well-lit and in focus\n" + \
                               "- Try using a different language if the text is not in English"
            
            # Update text display in main thread
            self.root.after(0, lambda: self.update_text_display(extracted_text))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                f"OCR processing error: {str(e)}\n\nPlease make sure Tesseract is properly installed."))
        finally:
            self.is_processing = False
            self.root.after(0, self.stop_progress)
    
    def update_text_display(self, text):
        """Update text display with extracted text"""
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, text)
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
        """Save the segmented result"""
        if self.original_image is None:
            messagebox.showinfo("Info", "Please open an image first")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Create transparency mask (255 for foreground, 0 for background)
                alpha = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
                alpha[(self.mask == GC_FGD) | (self.mask == GC_PR_FGD)] = 255
                
                # Create RGBA image
                bgr = self.original_image.copy()
                rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = alpha
                
                # Save image with transparency
                cv2.imwrite(file_path, rgba)
                
                self.status_var.set(f"Result saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving result: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedImageEditor(root)
    root.mainloop() 