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
        self.root.title("OCR Segmentation Comparator")
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
        Label(self.left_panel, text="OCR Segmentation Comparator", font=("Arial", 14, "bold")).pack(pady=10)
        
        # File operations frame
        file_frame = LabelFrame(self.left_panel, text="File Operations")
        file_frame.pack(fill=X, padx=5, pady=5, ipady=5)
        Button(file_frame, text="Open Image", command=self.open_image).pack(fill=X, padx=5, pady=2)
        Button(file_frame, text="Save Current View", command=self.save_result).pack(fill=X, padx=5, pady=2)
        Button(file_frame, text="Reset Image", command=self.reset_image).pack(fill=X, padx=5, pady=2)
        
        # --- MODIFIED: Segmentation Frame ---
        segment_frame = LabelFrame(self.left_panel, text="Segmentation Tools")
        segment_frame.pack(fill=X, padx=5, pady=5, ipady=5)
        
        Label(segment_frame, text="1. Draw a rectangle on the image").pack(anchor=W, padx=5)
        Label(segment_frame, text="2. Choose a segmentation method:").pack(anchor=W, padx=5)
        
        self.segment_method_var = StringVar(value="grabcut")
        methods = [
            ("GrabCut", "grabcut"),
            ("Binary Threshold", "binary"),
            ("Otsu's Threshold", "otsu"),
            ("Adaptive Threshold", "adaptive")
        ]
        for text, mode in methods:
            Radiobutton(segment_frame, text=text, value=mode, 
                        variable=self.segment_method_var).pack(anchor=W, padx=15)
        
        Button(segment_frame, text="3. Apply Segmentation", command=self.apply_segmentation).pack(fill=X, padx=5, pady=5)
        
        # OCR frame
        ocr_frame = LabelFrame(self.left_panel, text="OCR Tools")
        ocr_frame.pack(fill=X, padx=5, pady=5, ipady=5)
        Button(ocr_frame, text="Extract Text from Current View", command=self.extract_text).pack(fill=X, padx=5, pady=2)
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
    
    # --- New Dispatcher Function ---
    def apply_segmentation(self):
        """Calls the appropriate segmentation function based on user's choice."""
        if self.original_image is None or self.rect == (0, 0, 0, 0):
            messagebox.showinfo("Info", "Please open an image and select a region first.")
            return

        method = self.segment_method_var.get()
        
        # Always work on a fresh copy of the original image
        img_copy = self.original_image.copy()
        
        if method == "grabcut":
            self.run_grabcut(img_copy)
        elif method == "binary":
            self.run_binary_threshold(img_copy)
        elif method == "otsu":
            self.run_otsu_threshold(img_copy)
        elif method == "adaptive":
            self.run_adaptive_threshold(img_copy)

    # --- New Segmentation Functions ---
    def run_binary_threshold(self, image):
        """Performs simple binary thresholding on the selected region."""
        self.status_var.set("Running Binary Threshold...")
        self.root.update_idletasks()

        try:
            x, y, w, h = self.rect
            roi = image[y:y+h, x:x+w]
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Using a fixed threshold of 127. You can adjust this value.
            _, mask = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Create a white background and place the segmented foreground on it
            result_roi = cv2.bitwise_and(roi, roi, mask=mask)
            background = np.full(roi.shape, 255, dtype=np.uint8)
            inv_mask = cv2.bitwise_not(mask)
            bg_part = cv2.bitwise_and(background, background, mask=inv_mask)
            final_roi = cv2.add(result_roi, bg_part)

            # Place the processed ROI back into the main image
            self.display_image = final_roi
            
            self.update_display()
            output_filename = "result_binary.png"
            cv2.imwrite(output_filename, self.display_image)
            self.status_var.set(f"Binary Threshold done. Saved to {output_filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Binary Threshold failed: {e}")
            self.status_var.set("Ready")

    def run_otsu_threshold(self, image):
        """Performs Otsu's thresholding on the selected region."""
        self.status_var.set("Running Otsu's Threshold...")
        self.root.update_idletasks()
        try:
            x, y, w, h = self.rect
            roi = image[y:y+h, x:x+w]
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            result_roi = cv2.bitwise_and(roi, roi, mask=mask)
            background = np.full(roi.shape, 255, dtype=np.uint8)
            inv_mask = cv2.bitwise_not(mask)
            bg_part = cv2.bitwise_and(background, background, mask=inv_mask)
            final_roi = cv2.add(result_roi, bg_part)

            self.display_image = final_roi
            
            self.update_display()
            output_filename = "result_otsu.png"
            cv2.imwrite(output_filename, self.display_image)
            self.status_var.set(f"Otsu's Threshold done. Saved to {output_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Otsu's Threshold failed: {e}")
            self.status_var.set("Ready")
            
    def run_adaptive_threshold(self, image):
        """Performs adaptive thresholding on the selected region."""
        self.status_var.set("Running Adaptive Threshold...")
        self.root.update_idletasks()
        try:
            x, y, w, h = self.rect
            roi = image[y:y+h, x:x+w]
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Parameters for adaptive threshold can be tuned
            mask = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            result_roi = cv2.bitwise_and(roi, roi, mask=mask)
            background = np.full(roi.shape, 255, dtype=np.uint8)
            inv_mask = cv2.bitwise_not(mask)
            bg_part = cv2.bitwise_and(background, background, mask=inv_mask)
            final_roi = cv2.add(result_roi, bg_part)

            self.display_image = final_roi
            
            self.update_display()
            output_filename = "result_adaptive.png"
            cv2.imwrite(output_filename, self.display_image)
            self.status_var.set(f"Adaptive Threshold done. Saved to {output_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Adaptive Threshold failed: {e}")
            self.status_var.set("Ready")

    def run_grabcut(self, image):
        """Run GrabCut segmentation on the selected region."""
        self.status_var.set("Running GrabCut...")
        self.root.update_idletasks()
        try:
            mask = np.zeros(image.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(image, mask, self.rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            
            # The mask is now modified. 0 and 2 are background, 1 and 3 are foreground.
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Create a white background image
            white_bg = np.full(image.shape, 255, dtype=np.uint8)

            # Use the mask to create the full-sized result image first
            full_result_image = white_bg * (1 - mask2[:, :, np.newaxis]) + image * mask2[:, :, np.newaxis]

# Now, crop this result to show ONLY the selected rectangle
            x, y, w, h = self.rect
            self.display_image = full_result_image[y:y+h, x:x+w]
            
            self.update_display()
            output_filename = "result_grabcut.png"
            cv2.imwrite(output_filename, self.display_image)
            self.status_var.set(f"GrabCut done. Saved to {output_filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during GrabCut segmentation: {str(e)}")
            self.status_var.set("Ready")

    # --- UNCHANGED HELPER FUNCTIONS BELOW ---

    def show_welcome(self):
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="OCR Segmentation Comparator", 
                                font=("Arial", 24, "bold"), fill="navy")
        self.canvas.create_text(400, 350, 
                                 text="1. Open an image\n" + 
                                      "2. Draw a rectangle around the text\n" +
                                      "3. Choose a segmentation method and apply it\n" +
                                      "4. Use OCR to extract text from the result",
                                 font=("Arial", 12), fill="black", justify=CENTER)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")]
        )
        if not file_path: return
        try:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", f"Could not read the image: {file_path}")
                return
            self.reset_image() # Use reset_image to set up display
        except Exception as e:
            messagebox.showerror("Error", f"Error opening image: {str(e)}")
    
    def update_display(self):
        if self.display_image is not None:
            img_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            self.tk_image = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)
            h, w = self.display_image.shape[:2]
            self.canvas.config(scrollregion=(0, 0, w, h))
            if self.rect != (0, 0, 0, 0):
                x, y, width, height = self.rect
                self.rect_id = self.canvas.create_rectangle(
                    x, y, x + width, y + height, 
                    outline="red", width=2, dash=(4, 4))
    
    def on_mouse_down(self, event):
        if self.display_image is None: return
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)
        self.status_var.set(f"Started selection at ({int(self.start_x)}, {int(self.start_y)})")
    
    def on_mouse_drag(self, event):
        if not self.rect_id or self.display_image is None: return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, x, y)
        width = abs(x - self.start_x)
        height = abs(y - self.start_y)
        self.status_var.set(f"Selection size: {int(width)}x{int(height)}")
    
    def on_mouse_up(self, event):
        if not self.rect_id or self.display_image is None: return
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        x1 = min(self.start_x, x)
        y1 = min(self.start_y, y)
        width = abs(x - self.start_x)
        height = abs(y - self.start_y)
        if width < 5 or height < 5:
            self.status_var.set("Selection too small, please try again")
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.rect = (0, 0, 0, 0)
            return
        h_img, w_img = self.display_image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        width = min(width, w_img - x1)
        height = min(height, h_img - y1)
        self.rect = (int(x1), int(y1), int(width), int(height))
        self.canvas.coords(self.rect_id, x1, y1, x1 + width, y1 + height)
        self.status_var.set(f"Selection: ({self.rect[0]}, {self.rect[1]}, {self.rect[2]}x{self.rect[3]})")
    
    def reset_image(self):
        if self.original_image is not None:
            self.display_image = self.original_image.copy()
            if self.rect_id: self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.rect = (0, 0, 0, 0)
            self.update_display()
            self.status_var.set("Image reset to original.")
    
    def extract_text(self):
        if self.display_image is None:
            messagebox.showinfo("Info", "No image to process.")
            return
        
        # We will now always process the full (potentially segmented) view
        image_to_process = self.display_image
        self.status_var.set("Extracting text...")
        self.root.update_idletasks()
        try:
            # Using a simple config. The pre-processing from segmentation is key.
            config = '--oem 3 --psm 6'
            text = pytesseract.image_to_string(image_to_process, lang='eng', config=config)
            
            if not text.strip():
                text = "No text could be extracted. The segmentation may have removed the text or made it illegible. Try a different method or reset the image."
            
            self.update_text_display(text)
        except Exception as e:
            messagebox.showerror("Error", f"OCR processing error: {str(e)}\n\nPlease make sure Tesseract is properly installed.")
            self.status_var.set("OCR failed.")

    def update_text_display(self, text):
        self.text_display.delete(1.0, END)
        self.text_display.insert(END, text)
        self.extracted_text = text
        self.status_var.set("OCR completed")
    
    def copy_text(self):
        if self.extracted_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.extracted_text)
            self.status_var.set("Text copied to clipboard")
        else:
            messagebox.showinfo("Info", "No text to copy")
    
    def save_text(self):
        if not self.extracted_text:
            messagebox.showinfo("Info", "No text to save")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.extracted_text)
                self.status_var.set(f"Text saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving text: {str(e)}")
    
    def save_result(self):
        if self.display_image is None:
            messagebox.showinfo("Info", "No image to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            try:
                cv2.imwrite(file_path, self.display_image)
                self.status_var.set(f"Image saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {str(e)}")

# Run the GUI
if __name__ == "__main__":
    root = Tk()
    app = ImageTextExtractor(root)
    root.mainloop()