# Import necessary libraries
import cv2
import numpy as np
import imutils
import tkinter as tk
from tkinter import filedialog
from imutils.object_detection import non_max_suppression
from PIL import Image, ImageTk



class PedestrianDetectionApp:
    def __init__(self, root):
        # Initialize the Tkinter window
        self.root = root
        self.root.title("Pedestrian Detector App")
        
        # Title label
        self.label = tk.Label(root, text="Pedestrian Detector ", font=("algerian", 25, "bold"))
        self.label.pack(pady=10)


        # Button to load an image
        self.load_button = tk.Button(root, text="Load Image", font=("amasis mt pro medium", 15 , "bold"), command=self.load_image)
        self.load_button.pack(pady=10)

        # Button to detect pedestrians
        self.detect_button = tk.Button(root, text="Detect Pedestrian", font=("amasis mt pro medium", 15 , "bold"), command=self.detect_pedestrians)
        self.detect_button.pack(pady=10)        

        # Canvas to display the loaded image
        self.canvas = tk.Canvas(root, width=600, height=450)
        self.canvas.pack(pady=10)

    def load_image(self):
        # Open a file dialog to select an image file
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if self.file_path:
            # Read the selected image
            img = cv2.imread(self.file_path)
            # Uncomment the line below to display the loaded image
            self.display_image(img)

    def detect_pedestrians(self):
        # Check if a file was selected
        if hasattr(self, 'file_path') and self.file_path:
            # Read the selected image
            img = cv2.imread(self.file_path)
            # Resize the image
            img = imutils.resize(img, width=500, height= 600)
            
            # Initialize HOG descriptor and detector
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect pedestrians in the image
            rects, weights = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)

             # Apply non-maximum suppression to reduce overlapping bounding boxes   
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            
            # Draw bounding boxes and labels on the image
            b = 1
            for (x, y, w, h) in pick:
                cv2.rectangle(img, (x, y), (w, h), (0, 0, 100), 2)
                cv2.rectangle(img, (x, y - 20), (w, y), (0, 0, 255), -1)
                cv2.putText(img, f'P{b}', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                b += 1
            
            # Display the result
            self.display_image(img)

    def convert_to_tkinter_image(self, image):
        # Convert the image to a Tkinter-compatible format
        image = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(image=image)
        return img_tk

    def display_image(self, img):
        # Convert the image from BGR to RGB for displaying with tkinter
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to a tkinter PhotoImage
        img_tk = self.convert_to_tkinter_image(img_rgb)

        # Create an image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        # Save a reference to the image to prevent it from being garbage collected
        self.canvas.image = img_tk




if __name__ == "__main__":
    # Run the main function if the script is executed
    # Create the main Tkinter window
    root = tk.Tk()

    # Create an instance of the PedestrianDetectionApp class
    app = PedestrianDetectionApp(root)

    # Start the Tkinter event loop
    root.mainloop()
  
