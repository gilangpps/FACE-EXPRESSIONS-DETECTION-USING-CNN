import os
import cv2
import numpy as np
from datetime import datetime

class ManualDatasetCollector:
    def __init__(self, data_dir='manual_emotion_dataset'):
        self.data_dir = data_dir
        self.img_size = (48, 48)

        # Hanya 4 ekspresi yang digunakan
        self.emotions = {
            'a': 'angry',
            's': 'sad',
            'h': 'happy',
            'u': 'surprise',
        }

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._prepare_dirs()

    def _prepare_dirs(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for emotion in self.emotions.values():
            os.makedirs(os.path.join(self.data_dir, emotion), exist_ok=True)

    def collect(self):
        cap = cv2.VideoCapture(0)
        print("🎥 Tekan tombol berikut untuk menyimpan ekspresi wajah:")
        for key, emotion in self.emotions.items():
            print(f"'{key.upper()}' - {emotion.capitalize()}")
        print("Tekan 'Q' untuk keluar.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            face_resized = None
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, self.img_size)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Detected Face", face_resized)

            cv2.imshow("Webcam Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if chr(key) in self.emotions and face_resized is not None:
                emotion = self.emotions[chr(key)]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(self.data_dir, emotion, f"{emotion}_{timestamp}.png")
                cv2.imwrite(filename, face_resized)
                print(f"[✔] {emotion.capitalize()} saved as {filename}")

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Dataset collection selesai.")

if __name__ == "__main__":
    collector = ManualDatasetCollector()
    collector.collect()
