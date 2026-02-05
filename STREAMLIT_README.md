# Streamlit Interface - Crowd Counting

Web interface for crowd counting in images using the SASNet model.

## How to Run

### Option 1: Using the script (recommended)

```bash
./run_streamlit.sh
```

### Option 2: Manually

```bash
source venv/bin/activate
streamlit run app.py
```

## Features

- ✅ Image upload (JPG, JPEG, PNG)
- ✅ Model selection (Part A or Part B)
- ✅ Density map visualization
- ✅ Accurate crowd counting
- ✅ Detailed statistics

## Requirements

- Activated virtual environment (`venv`)
- Pre-trained models in `./models/`:
  - `SHHA.pth` (ShanghaiTech Part A)
  - `SHHB.pth` (ShanghaiTech Part B)

## Usage

1. Run the command above
2. Open the browser at the displayed URL (usually `http://localhost:8501`)
3. Select the model in the sidebar
4. Upload an image
5. View the results!

## Notes

- **Part A**: Better for very dense crowds
- **Part B**: Better for sparse crowds
- Processing may take a few seconds depending on image size and device used
