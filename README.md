# Sign Language Detection – IBM Project

## 📌 Overview
This project is a **Sign Language Detection System** built using **Python, OpenCV, and a Machine Learning Classifier**.  
It allows users to:
- Collect custom hand gesture datasets.
- Train a classifier model.
- Recognize and predict sign language gestures in real-time.
- Interact via a web interface for login and gesture recognition.

---

## 📂 Project Structure
```
IBM/
│── app.py                  # Flask web app entry point
│── main.py                 # Script to run real-time sign detection
│── collect_image.py        # Capture images for dataset creation
│── creating_dataset.py     # Prepare dataset for training
│── training_classifier.py  # Train the ML classifier
│── inference_classifier.py # Run inference with the trained model
│── requirements.txt        # Python dependencies
│
├── static/                 # Static assets (images, backgrounds)
│    ├── asl.avif
│    ├── asl2.webp
│    ├── background.jpg
│    ├── login_backround.jpg
│
├── templates/              # HTML templates for the Flask app
│    ├── login.html
│    ├── main.html
│    ├── recognition.html
```

---

## ⚙️ Installation
1. **Clone or Download** this repository.
2. **Navigate to the project folder**:
   ```bash
   cd IBM
   ```
3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   source venv/bin/activate # Mac/Linux
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1️⃣ **Run Web App**
```bash
python app.py
```
- Opens the **Sign Language Detection** web interface in the browser.

### 2️⃣ **Collect Dataset**
```bash
python collect_image.py
```
- Capture images for each sign gesture.

### 3️⃣ **Create Dataset**
```bash
python creating_dataset.py
```
- Converts collected images into training-ready format.

### 4️⃣ **Train Classifier**
```bash
python training_classifier.py
```
- Trains the ML model and saves it.

### 5️⃣ **Run Real-Time Detection**
```bash
python main.py
```
- Starts camera-based real-time sign recognition.

---

## 🛠 Tech Stack
- **Python 3.x**
- **Flask**
- **OpenCV**
- **scikit-learn**
- **NumPy**
- **Pandas**

---

## ✨ Features
✅ Real-time gesture recognition  
✅ Web-based UI for interaction  
✅ Easy dataset collection and training  
✅ Extensible for new sign gestures  
