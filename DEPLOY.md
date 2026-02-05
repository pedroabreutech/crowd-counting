# Deployment Guide - Streamlit Cloud

This guide explains how to deploy the Crowd Counting System to Streamlit Cloud.

## Prerequisites

1. GitHub repository with your code
2. Streamlit Cloud account (free tier available)
3. Model files (SHHA.pth and SHHB.pth)

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository has:
- `app.py` (main Streamlit application)
- `requirements.txt` (with updated dependencies)
- `model.py` (SASNet model definition)
- `datasets/` directory with required files
- `.streamlit/config.toml` (configuration file)

### 2. Model Files Setup

**Option A: Using Git LFS (Recommended for large files)**

If your models are larger than 100MB:

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"

# Add and commit
git add .gitattributes
git add models/*.pth
git commit -m "Add models with Git LFS"
git push origin main
```

**Option B: Host Models Externally**

1. Upload models to cloud storage (Google Drive, Dropbox, AWS S3, etc.)
2. Get direct download links
3. Update `MODEL_URLS` in `app.py` with the download links
4. Models will be downloaded automatically on first use

**Option C: Manual Upload (Streamlit Cloud Secrets)**

1. Upload models to a cloud storage service
2. Add model URLs to Streamlit Cloud secrets (Settings → Secrets)
3. Update `app.py` to read from secrets

### 3. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Configure:
   - **Repository**: `pedroabreutech/crowd-counting`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy"

### 4. Configure Secrets (Optional)

If using external model hosting, add secrets in Streamlit Cloud:

1. Go to app settings
2. Click "Secrets"
3. Add model URLs:

```toml
[model_urls]
shha_url = "https://your-url.com/SHHA.pth"
shhb_url = "https://your-url.com/SHHB.pth"
```

## Troubleshooting

### Error: "installer returned a non-zero exit code"

- Check `requirements.txt` for version conflicts
- Ensure all dependencies are compatible with Python 3.10
- Try pinning specific versions

### Error: "Model not found"

- Ensure models are in `./models/` directory
- Or configure `MODEL_URLS` in `app.py`
- Check file permissions

### Memory Issues

- Reduce `max_image_size` in the app
- Use smaller model files if possible
- Consider using CPU-only PyTorch build

### Slow Performance

- Streamlit Cloud free tier has limited resources
- Consider upgrading or using alternative hosting
- Optimize image processing (reduce max size)

## Environment Variables

You can set these in Streamlit Cloud settings:

- `PYTHON_VERSION`: `3.10.12` (set in `runtime.txt`)
- Model URLs (if using secrets)

## Notes

- Free tier has 1GB RAM limit
- Models are cached after first load
- Large models may take time to download on first use
- Consider using lighter PyTorch builds for faster deployment

## Support

For issues, check:
- Streamlit Cloud logs
- GitHub Issues
- Streamlit Community Forum
