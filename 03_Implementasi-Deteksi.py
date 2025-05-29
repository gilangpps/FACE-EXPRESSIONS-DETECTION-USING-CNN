import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

class EmotionDetectorApp:
    def __init__(self, window, window_title, model_path):
        self.window = window
        self.window.title(window_title)
        
        # Load model CNN
        self.model = load_model(model_path)
        self.emotions = ['angry', 'happy', 'sad', 'surprise']
        self.img_size = (48, 48)
        
        # Load face detector Haarcascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Setup webcam
        self.cap = cv2.VideoCapture(0)
        
        # Create a label in Tkinter window to display video frames
        self.label = Label(window)
        self.label.pack()
        
        # Label untuk menampilkan emosi prediksi
        self.emotion_label = Label(window, text="", font=("Helvetica", 20))
        self.emotion_label.pack(pady=10)
        
        # Start update loop
        self.update()
        
        # Handle window closing event
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            emotion_text = "No face detected"
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, self.img_size)
                face_img = face_img.astype('float32') / 255.0
                face_img = np.expand_dims(face_img, axis=0)
                face_img = np.expand_dims(face_img, axis=-1)
                
                preds = self.model.predict(face_img)
                emotion_idx = np.argmax(preds)
                emotion_text = f"{self.emotions[emotion_idx]} ({preds[0][emotion_idx]*100:.1f}%)"
                
                # Draw rectangle dan label di frame asli
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, emotion_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                break  # hanya prediksi wajah pertama yang terdeteksi
            
            # Update label text
            self.emotion_label.config(text=emotion_text)
            
            # Convert ke RGB dan ke PIL Image utk Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        
        # Update frame tiap 10ms
        self.window.after(10, self.update)
    
    def on_closing(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    EmotionDetectorApp(tk.Tk(), "Realtime Facial Expression Detection", "emotion_detection_model.h5")
