# 🌿 Palm Leaf Nutrient Stress Detection using YOLOv8

## 📌 Project Overview

This project focuses on building an **AI-powered object detection system** to identify nutrient deficiencies in palm leaves.
The model not only classifies the type of deficiency but also **detects and highlights the exact affected region** on the leaf using bounding boxes.

---

## 🎯 Objective

* Detect early-stage nutrient stress in palm leaves
* Identify deficiency type (e.g., Nitrogen, Magnesium, etc.)
* Localize affected areas using bounding boxes
* Assist in faster and more accurate agricultural analysis

---

## 🚀 Features

* ✅ Object Detection using YOLOv8
* ✅ Multi-class detection (5 classes)
* ✅ Bounding box visualization
* ✅ Confidence score for predictions
* ✅ Streamlit web app for real-time predictions

---

## 🧠 Classes

The model detects the following nutrient conditions:

* Healthy
* Boron
* Kalium
* Magnesium
* Nitrogen

---

## 📊 Dataset Details

* **Total Images:** ~14,000
* **Annotation Format:** YOLO (bounding boxes)
* **Tools Used:** Roboflow, LabelImg
* **Preprocessing:**

  * Image resizing (640×640)
  * Data augmentation (flip, rotation, brightness)
* **Dataset Split:**

  * Train: 70%
  * Validation: 20%
  * Test: 10%

---

## 🏗️ Model Architecture

* Model: **YOLOv8 (Ultralytics)**
* Training Framework: PyTorch
* Image Size: 640
* Epochs: 80–100

---

## 📈 Performance Metrics

* **Precision:** ~0.95+
* **Recall:** ~0.95
* **mAP@0.5:** ~0.96
* **mAP@0.5:0.95:** ~0.89

> The model demonstrates strong performance in detecting and classifying nutrient deficiencies.

---

## 🖥️ Project Structure

```
Palm-Leaf-Detection/
│
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│
├── runs/
│   ├── detect/
│
├── best.pt
├── app.py
├── data.yaml
└── README.md
```

---

## ⚙️ Installation

```bash
pip install ultralytics streamlit opencv-python
```

---

## 🏋️‍♂️ Training

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640
)
```

---

## 🔍 Inference (Prediction)

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("image.jpg", show=True)
```

---

## 🌐 Streamlit Web App

Run the app:

```bash
streamlit run app.py
```

### Features:

* Upload leaf image
* Get real-time detection
* View bounding boxes and confidence scores

---

## 🔄 Workflow

```
Data Collection → Annotation → Training → Auto-labeling → Correction → Retraining → Deployment
```

---

## ⚠️ Challenges Faced

* Class imbalance in early dataset
* Duplicate class naming issues
* Large-scale annotation (14,000 images)
* Improving bounding box precision

---

## 🚀 Future Improvements

* Improve detection of smaller stress regions
* Add more diverse dataset (lighting, angles)
* Deploy mobile application
* Real-time camera-based detection

---

## 📌 Conclusion

This project successfully demonstrates an **end-to-end AI pipeline** for agricultural problem detection using computer vision.
The model can assist in **early detection of nutrient deficiencies**, improving crop health and productivity.

---

## 👤 Author

**Md Tajuddin**
Aspiring Machine Learning Engineer

---

## 📬 Contact

Feel free to connect for collaboration or queries.

---
