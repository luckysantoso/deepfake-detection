# Nama file: app.py
# Versi dengan perbaikan error ValueError

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

# 1. Impor definisi model dari file yang kita buat sebelumnya
from model_def import MAT

# 2. Konfigurasi Dasar Aplikasi
MODEL_PATH = "convnext-backbone.pth"
DEVICE = torch.device("cpu")
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['ASLI', 'PALSU (DEEPFAKE)']

# 3. Fungsi-fungsi Helper

def load_state(net, ckpt_state_dict):
    """Fungsi helper untuk memuat bobot model dengan aman."""
    net_state_dict = net.state_dict()
    new_state_dict = {k.replace('module.', ''): v for k, v in ckpt_state_dict.items()}
    compatible_state_dict = {k: v for k, v in new_state_dict.items() if k in net_state_dict and net_state_dict[k].shape == v.shape}
    net.load_state_dict(compatible_state_dict, strict=False)
    print(f"Berhasil memuat {len(compatible_state_dict)} kunci dari checkpoint.")

@st.cache_resource
def load_model(model_path):
    """Memuat model dan bobotnya. Di-cache agar tidak loading berulang-ulang."""
    model_config = {
        'net': 'convnext_base.fb_in22k_ft_in1k', 'num_classes': 2, 'M': 4,
        'mid_dims': 256, 'dropout_rate': 0.3, 'drop_final_rate': 0.3,
        'pretrained_backbone': True, 'size': IMAGE_SIZE, 'texture_enhance_ver': 2
    }
    st.info("Menginisialisasi arsitektur model...")
    model = MAT(**model_config)
    try:
        st.info(f"Memuat bobot model dari '{model_path}'...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint.get('state_dict', checkpoint)
        load_state(model, state_dict)
        st.success("Model berhasil dimuat!")
    except FileNotFoundError:
        st.error(f"Error: File model '{model_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_bytes):
    """Menyiapkan gambar untuk dimasukkan ke model."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image, transform(image).unsqueeze(0)

def predict_and_get_attentions(model, image_tensor):
    """Melakukan prediksi dan mengembalikan hasil + attention maps."""
    if model is None:
        return None, None, None
    
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        # Asumsikan model.forward() mengembalikan logits dan attention_maps
        logits, attention_maps = model(image_tensor, return_attentions=True)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    return predicted_class_idx.item(), confidence.item(), attention_maps

def create_attention_heatmap(pil_image, attention_map):
    """Membuat visualisasi heatmap dari satu attention map."""
    # `attention_map` awalnya adalah (H, W).
    # Ubah menjadi (1, 1, H, W) agar sesuai dengan format interpolate.
    attention_map_4d = attention_map.unsqueeze(0).unsqueeze(0)

    # Ubah ukuran attention map agar sesuai dengan gambar asli
    attention_map_resized = torch.nn.functional.interpolate(
        attention_map_4d,
        size=(pil_image.height, pil_image.width), # Gunakan (H, W) dari PIL Image
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy() # .squeeze() akan menghapus dimensi N dan C

    # Normalisasi heatmap
    heatmap = (attention_map_resized - np.min(attention_map_resized)) / (np.max(attention_map_resized) - np.min(attention_map_resized) + 1e-8)
    
    # Buat plot menggunakan matplotlib
    fig, ax = plt.subplots(figsize=(6, 6), frameon=False)
    ax.imshow(pil_image)
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.tight_layout(pad=0) # Hilangkan padding
    
    return fig

# 4. Membuat Antarmuka Pengguna (UI) dengan Streamlit
st.set_page_config(
    page_title="DeepFake Detector", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #FF4B4B;  /* Streamlit primary red */
    }
    .sub-header {
        color: #262730;  /* Streamlit dark text color */
        font-size: 1.2em;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF4B4B;  /* Streamlit primary color */
    }
    .stButton>button {
        /* Using default Streamlit button styles */
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .footer {
        text-align: center;
        color: #6c757d;  /* Standard gray */
        font-size: 0.8em;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1 class='main-header'>DeepFake Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload a facial image to verify its authenticity and visualize the model's attention areas</p>", unsafe_allow_html=True)

# Sidebar with project information
with st.sidebar:
    st.header("About This Project")
    
    with st.expander("Model Information", expanded=True):
        st.markdown("""
        - **Architecture**: Multi-Attention (MAT)
        - **Backbone**: ConvNeXt-Base
        - **Input Size**: 224×224 pixels
        - **Classification**: Binary (Real/Fake)
        """)
    
    with st.expander("How It Works", expanded=True):
        st.markdown("""
        The attention visualization shows which areas of the image most influenced the model's decision:
        - **Red/Yellow (hot)** areas have higher influence
        - **Blue/Green (cold)** areas have lower influence
        
        Multiple attention maps show different aspects the model is focusing on.
        """)
    
    st.markdown("### Team")
    st.info("Deep Learning Group 3 - 2025")
    
    st.markdown("### Resources")
    st.markdown("[GitHub Repository](https://github.com/luckysantoso/deepfake-detection)")

# Main content
model = load_model(MODEL_PATH)

# Create tabs for different sections
tab1, tab2 = st.tabs(["Deepfake Detection", "How to Use"])

with tab2:
    st.markdown("### How to Use This Tool")
    st.markdown("""
    1. Upload a facial image (JPG, JPEG, or PNG format)
    2. Click the "Analyze Image" button
    3. View the prediction result and confidence score
    4. Examine the attention maps to see which areas influenced the decision
    
    ⚠️ **Note**: For best results, use clear facial images with good lighting.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Good Example")
        st.markdown("✅ Clear frontal face image")
        st.markdown("✅ Good lighting")
        st.markdown("✅ Minimal background")
    
    with col2:
        st.markdown("#### Poor Example")
        st.markdown("❌ Multiple faces")
        st.markdown("❌ Blurry image")
        st.markdown("❌ Extreme angles")

with tab1:
    # Create two columns for upload and results
    upload_col, preview_col = st.columns([1, 1])
    
    with upload_col:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader("Choose a facial image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            pil_image, image_tensor = preprocess_image(image_bytes)
            
            st.image(pil_image, use_container_width=True, caption="Preview")
            
            st.markdown("<div style='text-align: center; margin-top: 15px; margin-bottom: 15px;'>", unsafe_allow_html=True)
            analyze_button = st.button('Analyze Image', use_container_width=True, key="analyze_btn")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if analyze_button and model is not None:
                with st.spinner('Analyzing image...'):
                    predicted_idx, confidence, attention_maps = predict_and_get_attentions(model, image_tensor)
                    
                    # Save results to session state
                    st.session_state['prediction_result'] = {
                        'idx': predicted_idx,
                        'confidence': confidence,
                        'attention_maps': attention_maps
                    }
        else:
            st.markdown('<div class="info-box">👆 Upload an image to begin analysis</div>', unsafe_allow_html=True)
            # Clear previous results when no file is uploaded
            if 'prediction_result' in st.session_state:
                del st.session_state['prediction_result']
    
    with preview_col:
        st.markdown("### Analysis Results")
        
        # Display results if available
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            predicted_idx = result['idx']
            confidence = result['confidence']
            
            predicted_class = CLASS_NAMES[predicted_idx]
            
            # More compact result box
            box_color = "#D32F2F" if predicted_class == 'PALSU (DEEPFAKE)' else "#388E3C"
            title = "⚠️ FAKE DETECTED" if predicted_class == 'PALSU (DEEPFAKE)' else "✅ AUTHENTIC"
            message = "This image appears to be manipulated or generated." if predicted_class == 'PALSU (DEEPFAKE)' else "This image appears to be authentic."
            
            st.markdown(f"""
            <div class="result-box" style="border-left: 5px solid {box_color}; padding: 10px;">
                <h3 style="color: {box_color}; margin: 0;">{title}</h3>
                <p style="margin: 5px 0;">{message}</p>
                <div style="display: flex; align-items: center; margin-top: 5px;">
                    <div style="flex-grow: 1;">
                        <div style="background: #f0f0f0; border-radius: 5px; height: 15px;">
                            <div style="background: {box_color}; width: {confidence*100}%; height: 100%; border-radius: 5px;"></div>
                        </div>
                    </div>
                    <div style="margin-left: 10px; font-weight: bold; color: {box_color};">
                        {confidence:.1%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Results will appear here after analysis")
    
    # Attention maps section (in the right column, below confidence score)
    if 'prediction_result' in st.session_state and st.session_state['prediction_result']['attention_maps'] is not None:
        attention_maps = st.session_state['prediction_result']['attention_maps']
        num_maps = min(4, attention_maps.shape[1])  # Limit to 4 maps for 2x2 layout
        
        with preview_col:
            st.markdown("---")
            st.markdown("### Attention Maps Visualization")
            st.markdown("These visualizations show which parts of the image influenced the model's decision the most.")
            # Create 2x2 grid for attention maps with smaller size
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            grid_cols = [col1, col2, col3, col4]
            
            for i in range(num_maps):
                with grid_cols[i]:
                    # Create smaller figure size
                    heatmap_fig = create_attention_heatmap(pil_image, attention_maps[0, i, :, :])
                    # Set a smaller figure size for display
                    st.pyplot(heatmap_fig, use_container_width=True)
                    st.markdown(f"<p style='text-align: center; font-size: 0.9em;'><b>Map #{i+1}</b></p>", unsafe_allow_html=True)
            
            st.markdown("#### Interpretation")
            st.markdown("""
            - **Red/Yellow areas**: Highly influential regions
            - **Blue/Green areas**: Less influential regions
            
            The model examines these areas to detect manipulation patterns.
            """)

# Footer
st.markdown("---")
st.markdown('<p class="footer">This is a demonstration project. Performance may vary outside the original dataset.<br>© 2025 Deep Learning Group 3 - All Rights Reserved</p>', unsafe_allow_html=True)