import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import pickle
import mediapipe as mp
import numpy as np
import threading

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KeadConnect - Sign Language Recognition")
        self.root.geometry("960x780")
        self.root.configure(bg="#1e1e2f")

        self.cap = cv2.VideoCapture(0)
        self.recognizing = False
        self.model_loaded = False
        self.predicted_text = ""

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.labels_dict = {i: chr(65 + i) for i in range(26)}  # A–Z

        self.build_ui()
        self.update_frame()

    def build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#1e1e2f")
        header.pack(pady=20)

        tk.Label(header, text="🤟 KeadConnect", font=("Helvetica", 28, "bold"), fg="#50fa7b", bg="#1e1e2f").pack()
        tk.Label(self.root, text="Sign Language Recognition", font=("Helvetica", 22, "bold"), fg="#f8f8f2", bg="#1e1e2f").pack(pady=5)
        tk.Label(self.root, text="Bridge the gap with gesture recognition and language detection.",
                 font=("Helvetica", 13), bg="#1e1e2f", fg="#bd93f9").pack()

        # Controls
        controls = tk.Frame(self.root, bg="#1e1e2f")
        controls.pack(pady=20)

        self.language_var = tk.StringVar()
        language_menu = ttk.Combobox(controls, textvariable=self.language_var,
                                     values=["English", "Hindi", "ASL"], width=25, font=("Helvetica", 11))
        language_menu.set("🌐 Choose Language")
        language_menu.grid(row=0, column=0, padx=10)

        style = ttk.Style()
        style.theme_use('default')
        style.configure('TButton', background="#ff79c6", foreground="#000", font=("Helvetica", 11, "bold"))
        ttk.Button(controls, text="▶ Start Recognition", command=self.start_recognition).grid(row=0, column=1, padx=10)

        # Webcam Feed
        self.video_label = tk.Label(self.root, bg="#ffffff", bd=4, relief="groove")
        self.video_label.pack(pady=20)

        # Predicted Text Output Box
        self.output_label = tk.Label(self.root, text="✋ Start Signing...", font=("Helvetica", 20, "bold"),
                                     bg="#282a36", fg="#ffffff", width=40, height=2, relief="ridge", bd=2)
        self.output_label.pack(pady=10)

        # Feature Icons
        features = tk.Frame(self.root, bg="#1e1e2f")
        features.pack(pady=30)

        icons = ["📁", "📊", "📝", "🎯", "🔐"]
        labels = ["Translate", "Analytics", "Comments", "Performance", "Privacy"]

        for i in range(5):
            item = tk.Frame(features, bg="#1e1e2f")
            item.grid(row=0, column=i, padx=20)
            tk.Label(item, text=icons[i], font=("Arial", 24), bg="#1e1e2f", fg="#ffb86c").pack()
            tk.Label(item, text=labels[i], font=("Arial", 10, "bold"), bg="#1e1e2f", fg="#f8f8f2").pack()

    def load_model(self):
        try:
            model_dict = pickle.load(open('model.p', 'rb'))
            self.model = model_dict['model']
            self.model_loaded = True
            print("✅ Model loaded successfully.")
        except Exception as e:
            print("❌ Error loading model.p:", e)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.recognizing and self.model_loaded:
                data_aux = []
                x_, y_ = [], []

                results = self.hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x)
                            y_.append(lm.y)

                        for lm in hand_landmarks.landmark:
                            data_aux.append(lm.x - min(x_))
                            data_aux.append(lm.y - min(y_))

                    if len(data_aux) == 42:
                        prediction = self.model.predict([np.asarray(data_aux)])
                        predicted_char = self.labels_dict.get(int(prediction[0]), "")

                        if len(self.predicted_text) == 0 or predicted_char != self.predicted_text[-1]:
                            self.predicted_text += predicted_char
                            self.output_label.config(text=self.predicted_text)

                        # Draw box and label
                        h, w, _ = frame.shape
                        x1 = int(min(x_) * w) - 10
                        y1 = int(min(y_) * h) - 10
                        x2 = int(max(x_) * w) + 10
                        y2 = int(max(y_) * h) + 10

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(frame, predicted_char, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def start_recognition(self):
        print("🟢 Recognition started for:", self.language_var.get())
        if not self.model_loaded:
            threading.Thread(target=self.load_model).start()
        self.recognizing = True

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
