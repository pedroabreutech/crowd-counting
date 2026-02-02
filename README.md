# SASNet (AAAI2021)

 Official implementation in PyTorch of **SASNet** as described in "To Choose or to Fuse? Scale Selection for Crowd Counting" by Qingyu Song *, Changan Wang *, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Jian Wu, Jiayi Ma.

<p align="center"> <img src="imgs/title.png" width="80%" />

The codes is tested with PyTorch 1.5.0. It may not run with other versions.
 
## Visualizations for the scale-adaptive selection
The proposed adaptive selection strategy automatically learns the internal relations and the following visualizations demonstrate its effectiveness.

<p align="center"><img src="imgs/fig1.png" width="80%"/>

## Installation
* Clone this repo into a directory named SASNet_ROOT
```bash
git clone https://github.com/pedroabreutech/crowd-counting.git
cd crowd-counting
```

* Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

* Install Python dependencies. We use python 3.6.8+ and pytorch >= 1.5.0
```bash
pip install -r requirements.txt
```

* Download ShanghaiTech dataset and models from [GoogleDrive](https://drive.google.com/drive/folders/17WobgYjekLTq3QIRW3wPyNByq9NJTmZ9?usp=sharing)

**Note:** The `datas/` and `models/` directories are excluded from git due to their large size. You need to download them separately and place them in the project root.

## Preparation
Organizing the datas and models as following:
```
SASNet_ROOT/
        |->datas/
        |    |->part_A_final/
        |    |->part_B_final/
        |    |->...
        |->models/
        |    |->SHHA.pth
        |    |->SHHB.pth
        |    |->...
        |->main.py
```
Generating the density maps for the data:
```
python prepare_dataset.py --data_path ./datas/part_A_final
python prepare_dataset.py --data_path ./datas/part_B_final
```

## Running

### Command Line Interface

Run the following commands to launch inference:

```bash
# ShanghaiTech Part A
python3 main.py --data_path ./datas/part_A_final --model_path ./models/SHHA.pth 

# ShanghaiTech Part B
python3 main.py --data_path ./datas/part_B_final --model_path ./models/SHHB.pth 

# Roboflow Dataset
python3 main.py --data_path ./datas/roboflow_dataset --model_path ./models/SHHA.pth --dataset_type roboflow --split train
```

### Streamlit Web Interface

Launch the interactive web interface:

```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run Streamlit
streamlit run app.py
# Or use the provided script
./run_streamlit.sh
```

See [STREAMLIT_README.md](STREAMLIT_README.md) for detailed instructions on using the Streamlit interface.

### Features

- **Interactive Web Interface**: Upload images and get real-time crowd counting results
- **Roboflow Dataset Support**: Process datasets exported from Roboflow in COCO format
- **Multiple Image Processing**: Process multiple images and get total count
- **Accuracy Metrics**: Calculate MAE, MSE, RMSE, and accuracy percentage
- **Density Map Visualization**: Visualize predicted density maps
- **Memory Management**: Automatic image resizing for large images
- **MPS/CUDA/CPU Support**: Automatic device detection for optimal performance

## The network
The overall architecture of the proposed SASNet mainly consists of three components: U-shape backbone, confidence branch and density branch.

<img src="imgs/main.png"/>

## Comparison with state-of-the-art methods
The SASNet achieved state-of-the-art performance on several challenging datasets with various densities.

<img src="imgs/results.png"/>

## Roboflow Dataset Integration

This implementation includes support for Roboflow datasets:

1. **Organize your Roboflow dataset:**
```bash
python3 organize_roboflow_dataset.py --source /path/to/roboflow/dataset --target ./datas/roboflow_dataset
```

2. **Prepare the dataset (convert COCO to density maps):**
```bash
python3 prepare_roboflow_dataset.py --data_path ./datas/roboflow_dataset --split train
python3 prepare_roboflow_dataset.py --data_path ./datas/roboflow_dataset --split valid
python3 prepare_roboflow_dataset.py --data_path ./datas/roboflow_dataset --split test
```

3. **Use in Streamlit:** Select "Roboflow (COCO)" option and upload your COCO JSON file.

## Qualitative results
The following qualitative results show impressive counting accuracy under various crowd densities.

<img src="imgs/vis.png"/>


## Citing SASNet

If you think SASNet is useful in your project, please consider citing us.

```BibTeX
@article{sasnet,
  title={To Choose or to Fuse? Scale Selection for Crowd Counting},
  author={Qingyu Song and Changan Wang and Yabiao Wang and Ying Tai and Chengjie Wang and Jilin Li and Jian Wu and Jiayi Ma},
  journal={The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)},
  year={2021}
}
```