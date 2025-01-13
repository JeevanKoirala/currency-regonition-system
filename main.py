import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import json

class CurrencyRecognitionSystem:
    def __init__(self):
        self.CURRENCIES = {
            "USD": {
                "denominations": {
                    1: {"color_range": [(40, 40, 40), (80, 255, 255)], "pattern": "washington"},
                    5: {"color_range": [(40, 40, 40), (80, 255, 255)], "pattern": "lincoln"},
                    10: {"color_range": [(15, 40, 40), (35, 255, 255)], "pattern": "hamilton"},
                    20: {"color_range": [(40, 40, 40), (80, 255, 255)], "pattern": "jackson"},
                    50: {"color_range": [(15, 40, 40), (35, 255, 255)], "pattern": "grant"},
                    100: {"color_range": [(85, 40, 40), (115, 255, 255)], "pattern": "franklin"}
                },
                "features": ["security_thread", "color_shift", "watermark"]
            },
            "EUR": {
                "denominations": {
                    5: {"color_range": [(0, 40, 40), (10, 255, 255)], "pattern": "classical"},
                    10: {"color_range": [(0, 40, 40), (20, 255, 255)], "pattern": "romanesque"},
                    20: {"color_range": [(100, 40, 40), (130, 255, 255)], "pattern": "gothic"},
                    50: {"color_range": [(20, 40, 40), (40, 255, 255)], "pattern": "renaissance"},
                    100: {"color_range": [(40, 40, 40), (60, 255, 255)], "pattern": "baroque"},
                    200: {"color_range": [(60, 40, 40), (80, 255, 255)], "pattern": "modern"},
                    500: {"color_range": [(130, 40, 40), (150, 255, 255)], "pattern": "contemporary"}
                },
                "features": ["hologram", "security_thread", "watermark"]
            }
        }
        
        self.load_denomination_templates()
        self.model = self.create_model()
        self.initialize_gui()

    def load_denomination_templates(self):
        self.templates = {}
        template_dir = "currency_templates"
        
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
            print(f"Please add currency templates to {template_dir}")
            return

        for currency in self.CURRENCIES:
            self.templates[currency] = {}
            for denom in self.CURRENCIES[currency]["denominations"]:
                template_path = f"{template_dir}/{currency}_{denom}.jpg"
                if os.path.exists(template_path):
                    template = cv2.imread(template_path, 0)
                    self.templates[currency][denom] = template

    def create_model(self):
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(self.CURRENCIES), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def detect_security_features(self, frame):
        features = {}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        features['security_thread'] = lines is not None and len(lines) > 0

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        features['watermark'] = np.mean(thresh) < 127

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        features['color_shift'] = self.detect_color_shift(hsv)

        return features

    def detect_color_shift(self, hsv):
        lower_bounds = np.array([85, 40, 40])
        upper_bounds = np.array([115, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bounds, upper_bounds)
        return np.sum(mask) > 1000

    def detect_denomination(self, frame, currency):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        best_match = None
        highest_confidence = 0
        
        for denom, features in self.CURRENCIES[currency]["denominations"].items():
            confidence = 0
            
            lower_bound = np.array(features["color_range"][0])
            upper_bound = np.array(features["color_range"][1])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            color_score = np.sum(mask) / (frame.shape[0] * frame.shape[1])
            
            if currency in self.templates and denom in self.templates[currency]:
                template = self.templates[currency][denom]
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                pattern_score = np.max(result)
            else:
                pattern_score = 0
                
            security_features = self.detect_security_features(frame)
            security_score = sum(security_features.values()) / len(security_features)
            
            confidence = (color_score + pattern_score + security_score) / 3
            
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_match = denom
        
        return best_match, highest_confidence

    def process_frame(self, frame):
        processed = cv2.resize(frame, (224, 224))
        processed = processed / 255.0
        
        prediction = self.model.predict(np.expand_dims(processed, axis=0))[0]
        currency_idx = np.argmax(prediction)
        currency = list(self.CURRENCIES.keys())[currency_idx]
        
        denomination, confidence = self.detect_denomination(frame, currency)
        
        security_features = self.detect_security_features(frame)
        
        return {
            'currency': currency,
            'denomination': denomination,
            'confidence': confidence,
            'security_features': security_features
        }

    def draw_results(self, frame, results):
        text = f"{results['currency']} {results['denomination']}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        conf_text = f"Confidence: {results['confidence']*100:.2f}%"
        cv2.putText(frame, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        y_pos = 90
        for feature, present in results['security_features'].items():
            feature_text = f"{feature}: {'Present' if present else 'Not detected'}"
            cv2.putText(frame, feature_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise Exception("Could not read image")
        
        results = self.process_frame(frame)
        self.draw_results(frame, results)
        
        return frame, results

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.process_frame(frame)
            self.draw_results(frame, results)
            
            cv2.imshow('Currency Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def process_webcam(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.process_frame(frame)
            self.draw_results(frame, results)
            
            cv2.imshow('Currency Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def initialize_gui(self):
        self.root = tk.Tk()
        self.root.title("Currency Recognition System")
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Button(main_frame, text="Process Image", 
                  command=self.gui_process_image).grid(row=0, column=0, pady=5)
        ttk.Button(main_frame, text="Process Video", 
                  command=self.gui_process_video).grid(row=1, column=0, pady=5)
        ttk.Button(main_frame, text="Start Webcam", 
                  command=self.process_webcam).grid(row=2, column=0, pady=5)
        
        self.result_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.result_var).grid(row=3, column=0, pady=5)
        
        self.result_var.set("Select an option to begin")

    def gui_process_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if file_path:
            try:
                frame, results = self.process_image(file_path)
                self.result_var.set(
                    f"Detected: {results['currency']} {results['denomination']}\n"
                    f"Confidence: {results['confidence']*100:.2f}%"
                )
                cv2.imshow('Result', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                self.result_var.set(f"Error: {str(e)}")

    def gui_process_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            try:
                self.process_video(file_path)
            except Exception as e:
                self.result_var.set(f"Error: {str(e)}")

    def run(self):
        self.root.mainloop()

def main():
    app = CurrencyRecognitionSystem()
    app.run()

if __name__ == "__main__":
    main()
