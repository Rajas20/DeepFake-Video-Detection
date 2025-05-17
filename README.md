# DeepFake Video Detection

A Deep Learning-based system for detecting deepfake videos using facial region preprocessing and a CNN + LSTM architecture.

This project involves extracting facial frames from videos, preprocessing them into a standardized format, and then feeding them into a deep learning model to detect whether a video is real or fake.

---


##  Features

-  Extracts frames from videos
-  Detects and crops faces using `face_recognition`
-  Converts faces into standardized `112x112` videos
-  Ready for training on CNN + LSTM for temporal analysis
-  Modular design for preprocessing, model training, and evaluation

---

## 🛠 Tech Stack

| Technology      | Purpose                         |
|-----------------|----------------------------------|
| Python          | Programming language            |
| OpenCV          | Video and image processing      |
| face_recognition| Face detection in frames        |
| NumPy           | Numerical computations          |
| tqdm            | Progress bar                    |
| PyTorch / TensorFlow | (Future) Model development |
| Git + GitHub    | Version control and collaboration |

---



