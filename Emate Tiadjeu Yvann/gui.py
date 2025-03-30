import sys
import numpy as np
import cv2
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30

# GTSRB Class ID to Name Mapping (Top 5 shown as example - expand this dictionary)
CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "Entry prohibited",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signal",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

class TrafficSignRecognizer:
    def __init__(self, root, model_file):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        self.model = tf.keras.models.load_model(model_file)
        
        # GUI Setup
        self.label = Label(root, text="Select an image to predict", font=("Arial", 14))
        self.label.pack(pady=10)
        
        self.image_label = Label(root)
        self.image_label.pack()
        
        self.predict_button = Button(root, text="Select Image", command=self.load_image)
        self.predict_button.pack(pady=10)
        
        self.result_label = Label(root, text="", font=("Arial", 12))
        self.result_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.ppm;*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        
        image = cv2.imread(file_path)
        if image is None:
            self.result_label.config(text="Error: Could not read the image file.")
            return
        
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        self.predict(image, file_path)

    def predict(self, image, file_path):
        # Make prediction
        prediction = self.model.predict(np.array([image]))
        predicted_category = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Get class name or default to 'Unknown'
        class_name = CLASS_NAMES.get(predicted_category, f"Unknown (Class {predicted_category})")

        # Display image
        img = Image.open(file_path)
        img = img.resize((200, 200), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=img)
        self.image_label.image = img
        
        # Update result text
        result_text = f"Predicted Category: {class_name}\nConfidence: {confidence:.2f}%"
        self.result_label.config(text=result_text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python gui.py model.h5")
    
    model_file = sys.argv[1]
    root = tk.Tk()
    app = TrafficSignRecognizer(root, model_file)
    root.mainloop()