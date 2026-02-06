import streamlit as st
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import SASNet
import argparse
import os
import urllib.request
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="SASNet - Crowd Counting",
    page_icon="👥",
    layout="wide"
)

# Application title
st.title("👥 SASNet - Crowd Counting System")
st.markdown("### Crowd counting system using deep learning (SASNet - AAAI 2021)")

# Function to clear memory
def clear_memory(device):
    """Clears device memory"""
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    import gc
    gc.collect()

# Function to detect device
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Function to download model if it doesn't exist
def download_model(model_name, model_url, model_path):
    """Downloads model from URL if it doesn't exist locally"""
    if os.path.exists(model_path):
        return True
    
    if not model_url:
        return False
    
    try:
        # Create models directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Downloading {model_name}... This may take a few minutes.")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
            progress_bar.progress(int(percent) / 100)
            status_text.text(f"Downloading {model_name}... {int(percent)}%")
        
        urllib.request.urlretrieve(model_url, model_path, reporthook=show_progress)
        progress_bar.empty()
        status_text.empty()
        return True
    except Exception as e:
        st.error(f"Error downloading {model_name}: {str(e)}")
        return False

# Function to load model
@st.cache_resource
def load_model(model_path, device):
    """Loads the SASNet model"""
    # Create simple args for the model
    class Args:
        def __init__(self):
            self.block_size = 32
    
    args = Args()
    model = SASNet(args=args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to resize image maintaining aspect ratio
def resize_image_if_needed(image, max_size=2048):
    """Resizes image if too large, maintaining aspect ratio"""
    width, height = image.size
    original_size = (width, height)
    
    # If image is larger than max_size in any dimension, resize
    if width > max_size or height > max_size:
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        return image, original_size, True
    
    return image, original_size, False

# Base function to process image (without resizing)
def process_image_base(image, model, device, log_para=1000):
    """Processes an image and returns count and density map (without resizing)"""
    # Transformations (same as original code)
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert PIL to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        pred_map = model(img_tensor)
        pred_map = pred_map.data.cpu().numpy()
    
    # Clear tensor memory
    del img_tensor
    clear_memory(device)
    
    # Calculate count
    count = np.sum(pred_map) / log_para
    
    # Remove extra dimensions from density map
    density_map = pred_map[0, 0]  # [H, W]
    
    # Clear pred_map from memory
    del pred_map
    
    return count, density_map, False, image.size

# Function to process image
def process_image(image, model, device, log_para=1000, max_image_size=2048):
    """Processes an image and returns count and density map"""
    # Resize if necessary to avoid memory issues
    image_resized, original_size, was_resized = resize_image_if_needed(image, max_image_size)
    
    # Use base function
    count, density_map, _, _ = process_image_base(image_resized, model, device, log_para)
    
    # Adjust count if image was resized
    if was_resized:
        scale_factor = (original_size[0] * original_size[1]) / (image_resized.size[0] * image_resized.size[1])
        count = count * scale_factor
    
    return count, density_map, was_resized, original_size

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Model selection
model_option = st.sidebar.selectbox(
    "Select model:",
    ["ShanghaiTech Part A (SHHA)", "ShanghaiTech Part B (SHHB)"],
    help="Part A is better for dense crowds, Part B for sparse crowds"
)

# Map selection to model path
model_paths = {
    "ShanghaiTech Part A (SHHA)": "./models/SHHA.pth",
    "ShanghaiTech Part B (SHHB)": "./models/SHHB.pth"
}

selected_model_path = model_paths[model_option]

# log_para parameter
log_para = st.sidebar.slider(
    "Scale parameter (log_para):",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100,
    help="Density map amplification factor"
)

# Maximum image size (to avoid memory issues)
max_image_size = st.sidebar.slider(
    "Maximum image size (pixels):",
    min_value=512,
    max_value=4096,
    value=2048,
    step=256,
    help="Larger images will be automatically resized to avoid memory issues"
)

# Memory warning
st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Memory Tip:**
- If you encounter memory errors, reduce the maximum image size
- Very large images (>3000px) may cause problems
- The system automatically resizes when necessary
""")

# Model URLs - Try to get from Streamlit Secrets first, then fallback to defaults
# To configure in Streamlit Cloud: Settings → Secrets → Add:
# [model_urls]
# shha_url = "https://your-url.com/SHHA.pth"
# shhb_url = "https://your-url.com/SHHB.pth"
try:
    secrets = st.secrets.get("model_urls", {})
    MODEL_URLS = {
        "ShanghaiTech Part A (SHHA)": secrets.get("shha_url") if secrets.get("shha_url") else None,
        "ShanghaiTech Part B (SHHB)": secrets.get("shhb_url") if secrets.get("shhb_url") else None
    }
except (AttributeError, KeyError, FileNotFoundError):
    # Fallback if secrets are not configured
    MODEL_URLS = {
        "ShanghaiTech Part A (SHHA)": None,
        "ShanghaiTech Part B (SHHB)": None
    }

# Check if model exists, try to download if URL is available
if not os.path.exists(selected_model_path):
    model_url = MODEL_URLS.get(model_option)
    if model_url:
        # Try to download the model
        if download_model(model_option, model_url, selected_model_path):
            st.sidebar.success(f"✅ Model {model_option} downloaded successfully!")
        else:
            st.sidebar.error(f"⚠️ Could not download {model_option}")
            st.sidebar.info("Please ensure the model file is available or update MODEL_URLS in app.py")
            st.stop()
    else:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(selected_model_path), exist_ok=True)
        
        st.sidebar.warning(f"⚠️ Model not found: {selected_model_path}")
        
        # Check if secrets are configured
        try:
            secrets = st.secrets.get("model_urls", {})
            has_secrets = bool(secrets.get("shha_url") or secrets.get("shhb_url"))
        except:
            has_secrets = False
        
        if has_secrets:
            st.sidebar.info("""
            **Model URLs are configured in Secrets, but download failed.**
            
            Please check:
            - URLs are correct and accessible
            - Files are publicly accessible (no authentication required)
            - Network connection is stable
            """)
        else:
            st.sidebar.info("""
            **Model Setup Required:**
            
            **Option 1: Configure in Streamlit Cloud Secrets (Recommended)**
            1. Go to app settings → Secrets
            2. Add model URLs:
            ```toml
            [model_urls]
            shha_url = "https://your-url.com/SHHA.pth"
            shhb_url = "https://your-url.com/SHHB.pth"
            ```
            
            **Option 2: Manual Installation**
            - Place model files in `./models/` directory:
              - `SHHA.pth` for ShanghaiTech Part A
              - `SHHB.pth` for ShanghaiTech Part B
            - Download from: https://drive.google.com/drive/folders/17WobgYjekLTq3QIRW3wPyNByq9NJTmZ9
            """)
        st.stop()

# Load device and model
device = get_device()
try:
    model = load_model(selected_model_path, device)
    st.sidebar.success(f"✅ Model loaded! Device: {device}")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Main area
st.header("📤 Image Upload")

# Image file upload
uploaded_file = st.file_uploader(
    "Upload an image for counting",
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, JPEG, PNG"
)

# Optional ground truth upload (annotation)
st.sidebar.markdown("---")
st.sidebar.header("📝 Ground Truth (Optional)")
upload_gt = st.sidebar.checkbox(
    "Provide actual count for accuracy calculation",
    help="Check this option if you know the actual number of people in the image"
)

gt_count = None
if upload_gt:
    gt_count = st.sidebar.number_input(
        "Actual number of people:",
        min_value=0,
        value=0,
        step=1,
        help="Enter the actual number of people in the image to calculate accuracy"
    )

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    
    # Show original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, width='stretch')
        st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
    
    # Process image
    with st.spinner("🔄 Processing image..."):
        try:
            count, density_map, was_resized, original_size = process_image(
                image, model, device, log_para, max_image_size
            )
            
            # Warn if image was resized
            if was_resized:
                st.warning(
                    f"⚠️ Image resized from {original_size[0]}x{original_size[1]} "
                    f"to {image.size[0]}x{image.size[1]} pixels to avoid memory issues. "
                    f"Count was adjusted proportionally."
                )
            
            with col2:
                st.subheader("🗺️ Density Map")
                
                # Create density map visualization
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(density_map, cmap='jet', interpolation='nearest')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            # Calculate metrics if ground truth is available
            error = None
            mae = None
            mse = None
            accuracy = None
            error_percent = None
            
            if upload_gt and gt_count is not None and gt_count > 0:
                error = abs(gt_count - count)
                mae = error
                mse = (gt_count - count) ** 2
                error_percent = (error / gt_count) * 100
                accuracy = max(0, 100 - error_percent)
            
            # Show result
            st.markdown("---")
            st.header("📊 Counting Result")
            
            # Create metrics in columns
            if upload_gt and gt_count is not None:
                # If ground truth is available, show more metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        label="👥 People Detected",
                        value=f"{count:.0f}",
                        help="Estimated number of people in the image"
                    )
                
                with metric_col2:
                    st.metric(
                        label="✅ Actual Count",
                        value=f"{gt_count:.0f}",
                        help="Actual number of people (ground truth)"
                    )
                
                with metric_col3:
                    delta = count - gt_count if gt_count is not None else None
                    st.metric(
                        label="📈 Difference",
                        value=f"{delta:.0f}" if delta is not None else "N/A",
                        delta=f"{delta:.0f}" if delta is not None else None,
                        help="Difference between prediction and actual value"
                    )
                
                with metric_col4:
                    st.metric(
                        label="🎯 Accuracy",
                        value=f"{accuracy:.2f}%" if accuracy is not None else "N/A",
                        help="Percentage of accuracy"
                    )
            else:
                # Without ground truth, show only count
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        label="👥 People Detected",
                        value=f"{count:.0f}",
                        help="Estimated number of people in the image"
                    )
                
                with metric_col2:
                    st.metric(
                        label="📈 Precise Count",
                        value=f"{count:.2f}",
                        help="Count with 2 decimal places"
                    )
                
                with metric_col3:
                    # Calculate density (people per pixel)
                    total_pixels = density_map.size
                    density_per_pixel = count / total_pixels * 1000000  # per million pixels
                    st.metric(
                        label="📊 Density",
                        value=f"{density_per_pixel:.2f}",
                        help="People per million pixels"
                    )
            
            # Detailed metrics section (if ground truth is available)
            if upload_gt and gt_count is not None and gt_count > 0:
                st.markdown("---")
                st.header("📈 Evaluation Metrics")
                
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric(
                        label="MAE (Mean Absolute Error)",
                        value=f"{mae:.2f}",
                        help="Mean absolute error"
                    )
                
                with metrics_col2:
                    st.metric(
                        label="MSE (Mean Squared Error)",
                        value=f"{mse:.2f}",
                        help="Mean squared error"
                    )
                
                with metrics_col3:
                    st.metric(
                        label="Relative Error",
                        value=f"{error_percent:.2f}%",
                        help="Percentage of error relative to actual value"
                    )
                
                with metrics_col4:
                    st.metric(
                        label="RMSE (Root Mean Squared Error)",
                        value=f"{np.sqrt(mse):.2f}",
                        help="Root mean squared error"
                    )
                
                # Comparison chart
                st.markdown("### 📊 Visual Comparison")
                fig_comparison, ax_comparison = plt.subplots(figsize=(8, 5))
                categories = ['Actual', 'Predicted']
                values = [gt_count, count]
                colors = ['#2ecc71', '#3498db']
                bars = ax_comparison.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
                ax_comparison.set_ylabel('Number of People', fontsize=12)
                ax_comparison.set_title('Comparison: Actual vs Predicted Count', fontsize=14, fontweight='bold')
                ax_comparison.grid(axis='y', alpha=0.3)
                
                # Add values on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax_comparison.text(bar.get_x() + bar.get_width()/2., height,
                                     f'{value:.1f}',
                                     ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig_comparison)
                plt.close(fig_comparison)
                
                # Quality indicator
                st.markdown("### 🎯 Quality Indicator")
                if accuracy >= 95:
                    quality_status = "🟢 Excellent"
                    quality_color = "green"
                elif accuracy >= 90:
                    quality_status = "🟡 Very Good"
                    quality_color = "orange"
                elif accuracy >= 80:
                    quality_status = "🟠 Good"
                    quality_color = "darkorange"
                elif accuracy >= 70:
                    quality_status = "🔴 Fair"
                    quality_color = "red"
                else:
                    quality_status = "⚫ Needs Improvement"
                    quality_color = "darkred"
                
                st.markdown(f"""
                <div style="background-color: {quality_color}; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: white; margin: 0;">{quality_status}</h3>
                    <p style="color: white; margin: 5px 0 0 0;">Accuracy: {accuracy:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional information
            with st.expander("ℹ️ Technical Information"):
                st.write(f"**Device used:** {device}")
                st.write(f"**Model:** {model_option}")
                st.write(f"**log_para parameter:** {log_para}")
                st.write(f"**Density map format:** {density_map.shape}")
                st.write(f"**Maximum value in map:** {np.max(density_map):.4f}")
                st.write(f"**Mean value in map:** {np.mean(density_map):.4f}")
                
                if upload_gt and gt_count is not None:
                    st.write("---")
                    st.write("**Evaluation Metrics:**")
                    if mae is not None:
                        st.write(f"- **MAE:** {mae:.2f}")
                    if mse is not None:
                        st.write(f"- **MSE:** {mse:.2f}")
                        st.write(f"- **RMSE:** {np.sqrt(mse):.2f}")
                    if error_percent is not None:
                        st.write(f"- **Relative Error:** {error_percent:.2f}%")
                    if accuracy is not None:
                        st.write(f"- **Accuracy:** {accuracy:.2f}%")
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.exception(e)

else:
    # Instructions when there is no image
    st.info("👆 Upload an image above to start crowd counting.")
    
    # Usage example
    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Select the model** in the sidebar:
           - **Part A**: Better for very dense crowds
           - **Part B**: Better for sparse crowds
        
        2. **Upload an image** using the button above
        
        3. **Wait for processing** - the system will:
           - Load and process the image
           - Generate a density map
           - Calculate the number of people
        
        4. **View the results**:
           - People count
           - Density map (warmer colors = more people)
           - Additional statistics
        """)
    
    # Information about the model
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About SASNet")
    st.sidebar.info("""
    SASNet (Scale-Adaptive Selection Network) is a deep learning model 
    for crowd counting in images, presented at AAAI 2021.
    
    The model uses adaptive scale selection to handle different 
    crowd densities.
    """)
