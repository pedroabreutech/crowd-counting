# Copyright 2021 Tencent
# Modified to support Roboflow dataset loading

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import glob
from time import sleep

class RoboflowDataset(Dataset):
    """
    Dataset loader para datasets do Roboflow.
    Espera estrutura: data_path/split/images/ e data_path/split/ground_truth/
    """
    def __init__(self, root_path, split='train', transform=None):
        """
        Args:
            root_path: Caminho raiz do dataset Roboflow
            split: Split a usar (train, valid, test)
            transform: Transformações a aplicar nas imagens
        """
        # Construir caminho das imagens
        images_dir = os.path.join(root_path, split, 'images')
        
        # Buscar todas as imagens
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        root = []
        for ext in img_extensions:
            root.extend(glob.glob(os.path.join(images_dir, ext)))
        
        self.nSamples = len(root)
        self.lines = sorted(root)  # Ordenar para consistência
        self.transform = transform
        self.split = split
        self.root_path = root_path
       
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # get the image path
        img_path = self.lines[index]

        img, target = load_roboflow_data(img_path, self.root_path, self.split)
        # perform data augmentation
        if self.transform is not None:
            img = self.transform(img)

        img = torch.Tensor(img)
        target = torch.Tensor(target)

        return img, target

def load_roboflow_data(img_path, root_path, split):
    """
    Carrega imagem e ground truth do dataset Roboflow.
    
    Args:
        img_path: Caminho completo da imagem
        root_path: Caminho raiz do dataset
        split: Split sendo usado (train, valid, test)
    
    Returns:
        tuple: (imagem PIL, mapa de densidade numpy array)
    """
    # Construir caminho do ground truth
    # Formato esperado: root_path/split/ground_truth/imagename_sigma4.h5
    img_filename = os.path.basename(img_path)
    base_name = os.path.splitext(img_filename)[0]
    gt_filename = f"{base_name}_sigma4.h5"
    gt_dir = os.path.join(root_path, split, 'ground_truth')
    gt_path = os.path.join(gt_dir, gt_filename)
    
    # Abrir imagem
    img = Image.open(img_path).convert('RGB')
    
    # Carregar ground truth (mapa de densidade)
    while True:
        try:
            if not os.path.exists(gt_path):
                # Se não existe, criar mapa vazio
                print(f"Aviso: Ground truth não encontrado para {img_filename}, usando mapa vazio")
                # Obter dimensões da imagem
                img_array = np.array(img)
                h, w = img_array.shape[:2]
                target = np.zeros((h, w))
                break
            else:
                gt_file = h5py.File(gt_path, 'r')
                target = np.asarray(gt_file['density'])
                gt_file.close()
                break
        except Exception as e:
            print(f"Erro ao carregar ground truth {gt_path}: {e}")
            sleep(2)

    return img, target
