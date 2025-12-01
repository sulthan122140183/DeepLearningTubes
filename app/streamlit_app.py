from pathlib import Path
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "models"
NOTEBOOKS_DIR = BASE / "notebooks"
DATA_PROCESSED_TRAIN_DIR = BASE / "data" / "processed" / "Train"
DATA_TRAIN_DIR = BASE / "data" / "Train"


def load_class_names():
    """Load class names from data/processed/Train or fallback to data/Train (sorted)."""
    class_names = []
    
    # Try processed data first
    if DATA_PROCESSED_TRAIN_DIR.exists():
        class_names = sorted([d.name for d in DATA_PROCESSED_TRAIN_DIR.iterdir() if d.is_dir()])
        if class_names:
            return class_names, DATA_PROCESSED_TRAIN_DIR, True  # True = using processed
    
    # Fallback to raw data
    if DATA_TRAIN_DIR.exists():
        class_names = sorted([d.name for d in DATA_TRAIN_DIR.iterdir() if d.is_dir()])
        return class_names, DATA_TRAIN_DIR, False  # False = using raw
    
    return [], None, False


@st.cache_resource
def load_recognizer():
    """Load TensorFlow model with proper error handling and fallback."""
    class_names, data_dir, is_processed = load_class_names()
    num_classes = len(class_names)
    
    # Determine data source label
    data_source = "data/processed/Train" if is_processed else "data/Train"
    
    # Try to load actual model
    try:
        from tensorflow.keras.models import load_model
        
        # Try multiple paths in order of preference
        model_paths = [
            NOTEBOOKS_DIR / "final_MobileNetV2_saved_model",
            NOTEBOOKS_DIR / "final_MobileNetV2_model.h5",
            MODELS_DIR / "recognizer.h5",
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                try:
                    model = load_model(str(model_path), compile=False)
                    idx_to_class = {i: name for i, name in enumerate(class_names)}
                    return model, idx_to_class, num_classes, True, data_source
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {model_path.name}: {str(e)[:80]}")
                    continue
    except Exception as e:
        st.warning(f"⚠️ TensorFlow unavailable: {str(e)[:80]}")
    
    # Fallback: Use mock predictor for demo
    if num_classes > 0:
        idx_to_class = {i: name for i, name in enumerate(class_names)}
        st.info("💡 Using demo mode (predictions are mock-based)")
        return None, idx_to_class, num_classes, False, data_source
    
    return None, None, 0, False, ""


def detect_faces(image: np.ndarray):
    """Detect faces and return bounding boxes using MediaPipe."""
    try:
        import mediapipe as mp

        mp_fd = mp.solutions.face_detection
        # Gunakan model_selection=1 (lebih akurat, lebih lambat)
        # PENTING: min_detection_confidence=0.2 SAMA seperti saat preprocessing training
        with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.2) as detector:
            results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            bboxes = []
            if results.detections:
                h, w = image.shape[:2]
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    bboxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
            return bboxes
    except Exception:
        st.error("Face detection unavailable (MediaPipe failed to initialize).")
        return []


def preprocess_face(face_img, target_size=(224, 224)):
    """
    Preprocess face untuk MobileNetV2.
    PENTING: Sesuai PERSIS dengan training.ipynb inference:
    - ImageDataGenerator rescale=1./255 (img / 255.0)
    - Resize ke (224, 224)
    - Conv BGR to RGB
    - Add batch dimension
    """
    # Resize ke ukuran input model
    face = cv2.resize(face_img, target_size)
    # Convert BGR ke RGB (mediapipe & training expect RGB)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # Normalize ke [0,1] - SAMA dengan ImageDataGenerator rescale=1./255
    face = face.astype("float32") / 255.0
    # Add batch dimension untuk model.predict()
    face = np.expand_dims(face, 0)
    return face


def mock_predict(face_img, idx_to_class):
    """
    Mock predictor for demo mode.
    Uses deterministic hash based on image statistics.
    """
    mean = np.mean(face_img.reshape(-1, 3), axis=0)
    v = int((mean[0] + mean[1] * 256 + mean[2] * 65536))
    pred_idx = v % max(1, len(idx_to_class))
    conf = float(min(0.99, 0.5 + (np.std(face_img) / 128)))
    return pred_idx, conf


def get_sample_images(data_dir, num_samples=3):
    """Get sample images from each class directory."""
    samples = {}
    if not data_dir:
        return samples
    
    try:
        for class_dir in sorted(data_dir.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
                if images:
                    samples[class_dir.name] = images[:num_samples]
    except Exception as e:
        st.error(f"Error loading sample images: {e}")
    
    return samples


def main():
    st.set_page_config(page_title="Face Recognition Demo", layout="wide")
    
    # Simplified CSS styling
    st.markdown("""
        <style>
        .title-main {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #1f2937;
        }
        .subtitle-main {
            text-align: center;
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='title-main'>🎯 Face Recognition</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle-main'>Deteksi & pengenalan wajah dengan MobileNetV2 • Upload foto untuk mencoba</div>",
        unsafe_allow_html=True
    )

    # Load recognizer model
    with st.spinner("⏳ Memuat model pengenal..."):
        model, idx_to_class, num_classes, is_real_model, data_source = load_recognizer()
    
    if num_classes == 0:
        st.error("❌ Tidak ada data training di folder `data/processed/Train` atau `data/Train`")
        return
    
    # Display status dengan metric
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Model Status", "🤖 Active" if is_real_model else "💡 Demo Mode")
    with col2:
        st.metric("📊 Total Classes", num_classes)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📸 Face Recognition", "📊 Training Data", "📈 Model Info"])
    
    with tab1:
        st.subheader("📸 Upload & Recognize")
        
        col_upload, col_info = st.columns([3, 1])
        
        with col_upload:
            uploaded = st.file_uploader(
                "Pilih gambar (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                help="Upload foto dengan wajah untuk deteksi & recognition"
            )
        
        with col_info:
            st.info(f"📂 Sumber: {data_source.split('/')[-1]}")
        
        if uploaded is None:
            st.write("👇 Upload foto untuk mulai")
        else:
            # Load and process image
            image = Image.open(uploaded).convert("RGB")
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Detect faces
            with st.spinner("🔍 Mendeteksi wajah..."):
                bboxes = detect_faces(image_bgr)

            if not bboxes:
                st.warning("⚠️ Tidak ada wajah terdeteksi. Coba foto lain dengan wajah yang jelas.")
            else:
                # Draw bboxes and recognize
                canvas = image_bgr.copy()
                faces_info = []
                
                for i, (x1, y1, x2, y2) in enumerate(bboxes):
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # PENTING: Crop langsung tanpa margin - SAMA dengan preprocessing.ipynb
                    face = image_bgr[y1:y2, x1:x2]
                    
                    if face.size == 0:
                        continue

                    label = "Unknown"
                    prob = 0.0
                    
                    if model is not None:
                        try:
                            inp = preprocess_face(face)
                            preds = model.predict(inp, verbose=0)
                            pred_idx = int(np.argmax(preds, axis=1)[0])
                            confidence = float(np.max(preds))
                            
                            # PENTING: Model dilatih pada cropped face dari data/processed/Train
                            # Gunakan threshold 0.1 (sangat permisif) - training tdk ada filtering confidence
                            if confidence >= 0.1:
                                name = idx_to_class.get(pred_idx, f"class_{pred_idx}")
                                label = name
                                prob = confidence
                            else:
                                label = "Unknown (Very Low Confidence)"
                                prob = confidence
                        except Exception as e:
                            label = f"Error: {str(e)[:15]}"
                    else:
                        try:
                            pred_idx, conf = mock_predict(face, idx_to_class)
                            label = idx_to_class.get(pred_idx, f"class_{pred_idx}")
                            prob = conf
                        except Exception as e:
                            label = f"Error: {str(e)[:15]}"

                    faces_info.append({"box": (x1, y1, x2, y2), "label": label, "prob": prob})
                    
                    # Draw label on canvas
                    text_label = f"{label} ({prob:.2%})" if prob > 0 else label
                    cv2.putText(
                        canvas, text_label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                    )

                # Display result 
                st.markdown(f"#### ✓ Terdeteksi {len(bboxes)} wajah")
                
                col_img, col_detail = st.columns([2, 1])
                
                with col_img:
                    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                with col_detail:
                    st.markdown("**Hasil Recognition**")
                    for i, f in enumerate(faces_info, 1):
                        prob_pct = f"{f['prob']:.1%}" if f['prob'] > 0 else "N/A"
                        status = "🤖" if is_real_model else "💡"
                        st.write(f"**Wajah {i}** {status}")
                        st.write(f"{f['label']}")
                        st.write(f"*Confidence: {prob_pct}*")
    
    with tab2:
        st.subheader("📊 Training Data Browser")
        
        class_names_list, _, _ = load_class_names()
        
        if class_names_list:
            data_dir = DATA_PROCESSED_TRAIN_DIR if DATA_PROCESSED_TRAIN_DIR.exists() else DATA_TRAIN_DIR
            samples = get_sample_images(data_dir, num_samples=3)
            
            col_select, col_count = st.columns([3, 1])
            
            with col_select:
                selected_class = st.selectbox(
                    "Pilih kelas untuk preview:",
                    class_names_list,
                    help=f"Total {len(class_names_list)} kelas tersedia"
                )
            
            with col_count:
                st.metric("Total Kelas", len(class_names_list))
            
            if selected_class in samples:
                st.write(f"**Samples dari: {selected_class}**")
                img_cols = st.columns(len(samples[selected_class]))
                for col, img_path in zip(img_cols, samples[selected_class]):
                    with col:
                        try:
                            img = Image.open(img_path).convert("RGB")
                            st.image(img, use_column_width=True, caption=img_path.name)
                        except Exception:
                            st.write("Error loading")
            else:
                st.info(f"Tidak ada gambar untuk kelas: {selected_class}")
            
            st.divider()
            st.subheader("📋 Semua Kelas")
            
            # Grid view dari semua kelas
            num_cols = 6
            cols_grid = st.columns(num_cols)
            
            for idx, class_name in enumerate(class_names_list):
                col_idx = idx % num_cols
                with cols_grid[col_idx]:
                    if class_name in samples:
                        try:
                            first_img = Image.open(samples[class_name][0]).convert("RGB")
                            first_img_resized = first_img.resize((100, 100))
                            st.image(first_img_resized, use_column_width=True)
                            st.caption(f"<small>{class_name}</small>", unsafe_allow_html=True)
                        except Exception:
                            st.write(f"<small>{class_name}</small>", unsafe_allow_html=True)
                    else:
                        st.write(f"<small>{class_name}</small>", unsafe_allow_html=True)
    
    with tab3:
        st.subheader("📈 Model Information")
        
        # Key metrics
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Total Classes", num_classes)
        
        with metric_cols[1]:
            st.metric("Status", "Active" if is_real_model else "Demo")
        
        with metric_cols[2]:
            st.metric("Data Source", data_source.split("/")[-1])
        
        with metric_cols[3]:
            st.metric("Input Size", "224x224")
        
        st.divider()
        
        # Model architecture
        col_arch, col_info = st.columns([2, 1])
        
        with col_arch:
            st.write("**🏗️ Model Architecture**")
            if model is not None:
                info_text = f"""
**Framework:** TensorFlow/Keras
**Architecture:** MobileNetV2 + Custom Layers
**Input Shape:** {model.input_shape}
**Output Shape:** {model.output_shape}
**Total Layers:** {len(model.layers)}
**Base Model:** ImageNet Pretrained (frozen)
                """
                st.write(info_text)
            else:
                st.info("Model tidak tersedia (demo mode)")
        
        with col_info:
            st.write("**ℹ️ Preprocessing**")
            st.write("""
• Resize: 224x224
• Normalize: 1/255
• Color Space: RGB
• Det. Confidence: 0.2
            """)
        
        st.divider()
        
        # Class list
        st.write(f"**📋 Classes** ({len(class_names_list)} total)")
        
        chunk_size = 10
        class_names_list, _, _ = load_class_names()
        
        for chunk_idx in range(0, len(class_names_list), chunk_size):
            chunk = class_names_list[chunk_idx:chunk_idx + chunk_size]
            with st.expander(f"Classes {chunk_idx + 1} - {min(chunk_idx + chunk_size, len(class_names_list))}"):
                for idx, class_name in enumerate(chunk, start=chunk_idx):
                    st.write(f"**{idx}:** {class_name}")


if __name__ == "__main__":
    main()
