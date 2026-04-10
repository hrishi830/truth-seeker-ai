# Truth-Seeker AI 🎥

A Multimodal Deepfake Detection System that analyzes both **video and audio** to identify manipulated media.

---

##  Overview

Truth-Seeker AI is designed to detect deepfake content by combining:

*  Video frame analysis
*  Audio signal processing
*  Fusion model for final prediction

This project demonstrates a complete pipeline for identifying fake media using machine learning techniques.

---

##  Features

* Detects deepfake videos
* Analyzes both audio and visual components
* Fusion-based prediction system
* User-friendly frontend interface
* Backend API for processing

---

##  Tech Stack

* **Backend:** Python, FastAPI
* **Frontend:** React (Vite + TypeScript)
* **ML Models:** TensorFlow / PyTorch
* **Libraries:** OpenCV, NumPy, Librosa

---

##  Project Structure

```
truth-seeker-ai/
│
├── backend/        # ML models and API
├── frontend/       # UI interface
├── weights/        # Pretrained models (not included in repo)
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/hrishi830/truth-seeker-ai.git
cd truth-seeker-ai
```

---

### 2. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run Backend

```bash
python main.py
```

---

### 4. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

---

##  Dataset

Due to size limitations, the dataset is not included in this repository.

You can download the dataset from:

 LAV-DF Dataset (Kaggle):
https://www.kaggle.com/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df

---

##  Notes

* Model weights and datasets are excluded due to large size
* This project is intended for demonstration and academic purposes
* Sample inputs can be used for testing

---

## Future Improvements

* Real-time detection system
* Improved model accuracy
* Deployment using cloud services

---

# Author

**Hrishi**

---

##  Acknowledgement

This project was developed as part of an academic project and later refined for demonstration and portfolio purposes.
