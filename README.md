# LDCE-Net: Liver Fibrosis Classification from Ultrasound Images

## 🧠 Overview
LDCE-Net is a custom, lightweight deep learning model designed from scratch to classify liver fibrosis stages (Normal, Fibrosis, Cirrhosis) using grayscale ultrasound images. It is optimized to run on CPU-only systems and designed specifically for use in real-time clinical environments.

- ⚡ Fast and resource-efficient
- ✅ Trained from scratch (no pre-trained models)
- 🏥 Designed for clinical deployment (no GPU required)
- 🎛️ Simple GUI using Streamlit
- 📊 Training includes stratified sampling, dropout, and regularization to prevent overfitting

---

## 📂 Project Structure
```
LDCE-Net/
├── ldce_net_model.py         # Model definition (LDCE-Net with depthwise + attention)
├── train.py                  # Full training pipeline with plots and metrics
├── app.py                    # Streamlit GUI for image classification
├── Liver Ultrasounds.zip     # Dataset (3 class folders: F0-Normal, F1-Fibrosis, F2-Cirrhosis)
├── ldce_model.pt             # Trained model weights
├── plots/                    # Accuracy, loss, and confusion matrix plots
├── README.md                 # Project documentation
```

---

## 🧪 Training the Model
Unzip `Liver Ultrasounds.zip` into a folder and ensure it contains subfolders like:
```
Liver Ultrasounds/
├── F0-Normal/
├── F1-Fibrosis/
└── F2-Cirrhosis/
```
Then run:
```bash
python train.py
```
This will:
- Train the model with stratified sampling
- Save best weights to `ldce_model.pt`
- Output plots to `plots/`

---

## 📊 Sample Training Metrics
- **Best validation accuracy**: ~87.6%
- **Train accuracy**: steadily rises to ~85%
- **No overfitting**: due to dropout + augmentations

---

## 🚀 GUI with Streamlit
```bash
streamlit run app.py
```
Upload any grayscale ultrasound image. The model will predict:
- F0-Normal
- F1-Fibrosis
- F2-Cirrhosis

---

