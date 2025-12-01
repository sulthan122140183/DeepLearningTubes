# 🎯 Face Recognition - Tugas Besar Deep Learning

Demo aplikasi pengenalan wajah dengan Streamlit menggunakan MobileNetV2 dan MediaPipe Face Detection.

## 📁 Struktur Folder

```
DeepLearningTubes/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── data/
│   ├── Train/                    # Raw training data (70 classes)
│   └── processed/
│       └── Train/                # Preprocessed faces (auto-generated)
├── notebooks/
│   ├── preprocessing.ipynb        # Face extraction pipeline
│   ├── training.ipynb             # Model training
│   └── final_MobileNetV2_model.h5 # Final trained model
├── models/
│   ├── recognizer.h5             # Model copy for inference
│   └── class_indices.json         # Class mapping
├── scripts/
│   └── create_demo_model.py       # Utility scripts
├── tools/
│   └── extract_pdf.py             # Helper tools
├── requirements.txt               # Dependencies
├── run_app.ps1                    # Launcher script (Windows)
└── README.md                      # This file
```

## 🚀 Quick Start (Windows PowerShell)

### Setup Virtual Environment

```powershell
# Create venv
python -m venv .venv

# Activate venv
. .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run Application

**Option 1: Using launcher script**
```powershell
.\run_app.ps1
```

**Option 2: Manual activation**
```powershell
. .venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

App akan berjalan di: **http://localhost:8502**

## 📊 Features

✅ **Face Detection** - MediaPipe dengan confidence 0.2  
✅ **Face Recognition** - MobileNetV2 (69 classes)  
✅ **Dynamic Class Loading** - Auto-load dari `data/processed/Train` atau `data/Train`  
✅ **Training Data Viewer** - Browse training samples  
✅ **Model Info** - View architecture & metrics  

## 🎓 Preprocessing & Training

### Data Preprocessing
Jalankan `notebooks/preprocessing.ipynb` untuk extract wajah:
- Input: Raw images dari `data/Train/`
- Output: Cropped faces → `data/processed/Train/`
- Face detection: MediaPipe dengan `min_confidence=0.2`

### Model Training
Jalankan `notebooks/training.ipynb` untuk train:
- Base model: MobileNetV2 (ImageNet pretrained, frozen)
- Input size: 224×224
- Augmentation: Rotation 30°, shifts 20%, zoom 20%, horizontal flip
- Normalization: `rescale=1/255`
- Output: `final_MobileNetV2_model.h5`

## ⚙️ Configuration

### Model Loading Priority
1. `notebooks/final_MobileNetV2_model.h5` (primary)
2. `notebooks/final_MobileNetV2_saved_model/` (SavedModel format)
3. `models/recognizer.h5` (legacy)

### Class Loading
- Priority 1: `data/processed/Train/` (preprocessed)
- Priority 2: `data/Train/` (raw training data)
- Total classes: 69 (auto-discovered from folder names)

## 🔧 Dependencies

Key packages:
- **TensorFlow 2.15** - Deep learning
- **Keras 2.15** - High-level API
- **OpenCV 4.8.1** - Image processing
- **MediaPipe** - Face detection
- **Streamlit** - Web UI
- **NumPy 1.26.4** - Numerical computing

See `requirements.txt` for full list.

## 🎨 Preprocessing Details

```python
# Input image
face = image[y1:y2, x1:x2]  # Crop from detection box

# Preprocessing steps
1. Resize → (224, 224)
2. BGR → RGB color space
3. Normalize → [0, 1] by dividing by 255.0
4. Add batch dimension → (1, 224, 224, 3)
```

This matches exactly with training pipeline in `training.ipynb`.

## 📈 Model Architecture

```
Input: (None, 224, 224, 3)
  ↓
MobileNetV2 (ImageNet pretrained, frozen)
  ↓
GlobalAveragePooling2D()
  ↓
Dense(1024, activation='relu')
  ↓
Dense(69, activation='softmax')  # 69 classes
```

## 📝 Troubleshooting

### Issue: "Access denied" when deleting .venv
**Solution**: Kill Python/Streamlit processes first
```powershell
Stop-Process -Name streamlit -Force
Stop-Process -Name python -Force
Remove-Item .venv -Recurse -Force
```

### Issue: Model not found
**Solution**: Copy trained model to correct location
```powershell
# From notebooks/ to models/
Copy-Item notebooks/final_MobileNetV2_model.h5 models/recognizer.h5
```

### Issue: "The name tf.losses.sparse_softmax_cross_entropy is deprecated"
**Solution**: Warning only, app still works. Uses TensorFlow 2.15.

## 📦 Submission Checklist

Before submitting, ensure:

- [ ] `.venv/` and `__pycache__/` are deleted or gitignored
- [ ] `data/processed/Train/` exists with preprocessed faces
- [ ] `notebooks/final_MobileNetV2_model.h5` exists
- [ ] `app/streamlit_app.py` runs without errors
- [ ] `requirements.txt` is up-to-date
- [ ] `README.md` has setup instructions
- [ ] `run_app.ps1` launcher works
- [ ] `.gitignore` includes `venv/`, `*.log`, etc.

## 🏃 Command Reference

```powershell
# Setup
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run
streamlit run app/streamlit_app.py

# Test model loading
python -c "from tensorflow.keras.models import load_model; m = load_model('notebooks/final_MobileNetV2_model.h5'); print('✅ Model loaded:', m.input_shape, '->', m.output_shape)"

# Cleanup
Stop-Process -Name streamlit -Force
Remove-Item .venv -Recurse -Force
```

## 📚 References

- **Tugas**: `Tugas Besar Deep Learning.pdf`
- **Training Notebooks**: `notebooks/training.ipynb`, `notebooks/preprocessing.ipynb`
- **MobileNetV2**: https://arxiv.org/abs/1801.04381
- **MediaPipe**: https://google.github.io/mediapipe/

---

**Created**: Dec 1, 2025  
**Status**: Ready for submission ✅
