# 🔊 What’s That Sound? (ESC-50 Classifier): Environmental Audio Classifier

**🌍 What’s That Sound?** is a deep learning pipeline that classifies environmental sounds into 50 categories using the [ESC-50 dataset](https://github.com/karoldvl/ESC-50). It converts audio signals into mel-spectrograms and uses a fine-tuned ConvNeXt-Tiny model to predict the most likely sound class.

> 🎧 Upload a `.wav` file, hear it back, and instantly get the predicted sound.

---

## 🚀 Features

- End-to-end audio classification pipeline
- Mel-spectrogram generation with Librosa
- Custom CNN, ResNet18, and ConvNeXt-Tiny models
- Transfer learning for improved accuracy
- 5-fold cross-validation evaluation
- Interactive **Streamlit** web app with real-time predictions and emojis
- 📽️ Demo video included

---

## 🧠 Objectives

- Build a robust model for real-world environmental sound classification
- Benchmark traditional vs. modern architectures
- Create a demo-ready, user-friendly interface
- Make it reproducible, understandable, and visually engaging

---

## 📁 Project Structure

- `data/` – ESC-50 dataset (download from Google Drive)
- `models/` – Trained models (.pth files, also in Drive)
- `notebooks/` – Project stages: data prep, training, evaluation, cross-validation
- `src/` – Python modules: dataset, models, training scripts
- `streamlit_app.py` – Streamlit app for real-time prediction
- `requirements.txt` – All project dependencies
- `README.md` – Project documentation

---

### 📦 Data & Model Files

Due to storage size, the `data/` and `models/` folders are hosted externally.

📁 Download them here:  
🔗 **[Google Drive – Project Assets](https://drive.google.com/drive/folders/1QTr_30UcbgM_xz4RT47lgUCAaZ2G-uiO?usp=drive_link)**

After downloading:
- Place `ESC-50-master` inside the `data/` folder
- Place model weights (e.g. `convnext_audio_classifier.pth`) inside the `models/` folder

---

## 🔍 Key Technologies

- **Python**
- **PyTorch** + **TorchVision**
- **Librosa**
- **ConvNeXt-Tiny** (pretrained on ImageNet)
- **Streamlit** for deployment

---

## 🧪 Experiments & Results

| Model          | Val Accuracy |
|----------------|--------------|
| Custom CNN     | 48.0%        |
| ResNet18       | ~75.0%       |
| ConvNeXt-Tiny  | **84.5%** ✅ |

### 📊 5-Fold Cross-Validation

| Fold | Val Accuracy |
|------|---------------|
| 1    | 80.5%         |
| 2    | 82.5%         |
| 3    | 83.75%        |
| 4    | 87.25%        |
| 5    | 79.75%        |

**📈 Average Accuracy: 82.75%**

---

## 🌐 Streamlit App

**Try it locally:**

```bash
streamlit run streamlit_app.py

