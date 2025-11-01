"""Tkinter-based face viewer for labeling and confirmation."""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from typing import List, Callable, Optional
import config


class FaceViewer:
    """Simple Tkinter interface for viewing and confirming faces."""
    
    def __init__(self, title: str = "Face Viewer"):
        """Initialize the face viewer window."""
        self.root = tk.Tk()
        self.root.title(title)
        self.result = None
        self.callback = None
        
    def show_single_face(self, face_image: np.ndarray, message: str = "") -> bool:
        """
        Show a single face and get confirmation.
        
        Args:
            face_image: Face image array (RGB)
            message: Message to display
            
        Returns:
            True if user confirms, False otherwise
        """
        self.result = None
        
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Message label
        if message:
            label = ttk.Label(self.root, text=message, padding=10)
            label.pack()
        
        # Display image
        img = Image.fromarray(face_image.astype('uint8'))
        img = img.resize(config.PREVIEW_IMAGE_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        img_label = ttk.Label(self.root, image=photo)
        img_label.image = photo  # Keep a reference
        img_label.pack(pady=10)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Yes", 
                  command=lambda: self._set_result(True)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="No", 
                  command=lambda: self._set_result(False)).pack(side=tk.LEFT, padx=5)
        
        # Center window
        self.root.update_idletasks()
        self._center_window()
        
        # Wait for result
        self.root.wait_variable(self.result_var)
        
        return self.result
    
    def show_face_grid(self, face_images: List[np.ndarray], 
                       message: str = "", 
                       allow_individual_selection: bool = False) -> Optional[List[int]]:
        """
        Show multiple faces in a grid.
        
        Args:
            face_images: List of face image arrays (RGB)
            message: Message to display
            allow_individual_selection: If True, allow selecting individual faces
            
        Returns:
            If allow_individual_selection is True: list of selected indices
            Otherwise: True if all confirmed, False if rejected
        """
        self.result = None
        self.selected_indices = []
        
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Message label
        if message:
            label = ttk.Label(self.root, text=message, padding=10, wraplength=600)
            label.pack()
        
        # Create scrollable frame
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display images in grid
        cols = config.PREVIEW_GRID_COLUMNS
        checkboxes = []
        
        for idx, face_image in enumerate(face_images):
            row = idx // cols
            col = idx % cols
            
            frame = ttk.Frame(scrollable_frame, padding=5)
            frame.grid(row=row, column=col, padx=5, pady=5)
            
            # Image
            img = Image.fromarray(face_image.astype('uint8'))
            img = img.resize(config.PREVIEW_IMAGE_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            img_label = ttk.Label(frame, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack()
            
            # Checkbox for individual selection
            if allow_individual_selection:
                var = tk.BooleanVar(value=True)
                checkbox = ttk.Checkbutton(frame, text=f"Face {idx+1}", variable=var)
                checkbox.pack()
                checkboxes.append((idx, var))
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        if allow_individual_selection:
            ttk.Button(button_frame, text="Confirm Selection", 
                      command=lambda: self._set_selection_result(checkboxes)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", 
                      command=lambda: self._set_result(None)).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Button(button_frame, text="Confirm All", 
                      command=lambda: self._set_result(True)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Reject All", 
                      command=lambda: self._set_result(False)).pack(side=tk.LEFT, padx=5)
        
        # Set window size
        self.root.geometry("800x600")
        self._center_window()
        
        # Wait for result
        self.root.wait_variable(self.result_var)
        
        if allow_individual_selection:
            return self.selected_indices if self.result else None
        else:
            return self.result
    
    def ask_person_info(self, default_name: str = "", default_surname: str = "",
                       default_instagram: str = "") -> Optional[dict]:
        """
        Show a form to input person information.
        
        Args:
            default_name: Default name
            default_surname: Default surname
            default_instagram: Default Instagram username
            
        Returns:
            Dictionary with 'name', 'surname', 'instagram' or None if cancelled
        """
        self.result = None
        
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Title
        title_label = ttk.Label(self.root, text="Enter Person Information", 
                               font=('Arial', 14, 'bold'), padding=10)
        title_label.pack()
        
        # Form frame
        form_frame = ttk.Frame(self.root, padding=20)
        form_frame.pack()
        
        # Name
        ttk.Label(form_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_entry = ttk.Entry(form_frame, width=30)
        name_entry.insert(0, default_name)
        name_entry.grid(row=0, column=1, pady=5)
        name_entry.focus()
        
        # Surname
        ttk.Label(form_frame, text="Surname:").grid(row=1, column=0, sticky=tk.W, pady=5)
        surname_entry = ttk.Entry(form_frame, width=30)
        surname_entry.insert(0, default_surname)
        surname_entry.grid(row=1, column=1, pady=5)
        
        # Instagram
        ttk.Label(form_frame, text="Instagram:").grid(row=2, column=0, sticky=tk.W, pady=5)
        instagram_entry = ttk.Entry(form_frame, width=30)
        instagram_entry.insert(0, default_instagram)
        instagram_entry.grid(row=2, column=1, pady=5)
        
        ttk.Label(form_frame, text="(optional)", 
                 font=('Arial', 9, 'italic')).grid(row=2, column=2, sticky=tk.W, padx=5)
        
        def submit():
            name = name_entry.get().strip()
            surname = surname_entry.get().strip()
            instagram = instagram_entry.get().strip()
            
            if not name or not surname:
                messagebox.showerror("Error", "Name and Surname are required!")
                return
            
            self.result = {
                'name': name,
                'surname': surname,
                'instagram': instagram if instagram else None
            }
            self.result_var.set(1)
        
        def cancel():
            self.result = None
            self.result_var.set(1)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Submit", 
                  command=submit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=cancel).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to submit
        self.root.bind('<Return>', lambda e: submit())
        
        # Center window
        self.root.geometry("400x250")
        self._center_window()
        
        # Wait for result
        self.root.wait_variable(self.result_var)
        
        return self.result
    
    def _set_result(self, value):
        """Set the result and close the dialog."""
        self.result = value
        self.result_var.set(1)
    
    def _set_selection_result(self, checkboxes):
        """Set the selection result from checkboxes."""
        self.selected_indices = [idx for idx, var in checkboxes if var.get()]
        self.result = True
        self.result_var.set(1)
    
    def _center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def close(self):
        """Close the viewer window."""
        self.root.destroy()
    
    def __enter__(self):
        """Context manager entry."""
        self.result_var = tk.IntVar()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
