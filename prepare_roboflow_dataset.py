# Copyright 2021 Tencent
# Modified to support Roboflow COCO dataset preparation

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
import glob
import argparse
import cv2
import h5py
import numpy as np
from prepare_dataset import generate_density_map
from datasets.roboflow_converter import coco_to_center_points, validate_coco_file

# define the argparser
def get_args_parser():
    parser = argparse.ArgumentParser('Roboflow Data Preprocess', add_help=False)
    parser.add_argument('--data_path', type=str, help='root path of the Roboflow dataset')
    parser.add_argument('--split', type=str, default='train', help='split to process: train, valid, or test')
    parser.add_argument('--coco_file', type=str, default=None, 
                       help='path to COCO JSON file (if not in standard location)')
    parser.add_argument('--class_name', type=str, default='person',
                       help='class name to filter (default: person)')
    
    return parser

def find_coco_file(data_path, split):
    """
    Encontra o arquivo COCO JSON no dataset Roboflow.
    Procura por padrões comuns: _annotations.coco.json, annotations.json, etc.
    """
    possible_names = [
        '_annotations.coco.json',
        'annotations.json',
        f'{split}_annotations.coco.json',
        f'{split}_annotations.json',
        'coco.json'
    ]
    
    # Procurar no diretório do split
    split_dir = os.path.join(data_path, split)
    if os.path.exists(split_dir):
        for name in possible_names:
            coco_path = os.path.join(split_dir, name)
            if os.path.exists(coco_path):
                return coco_path
    
    # Procurar no diretório raiz
    for name in possible_names:
        coco_path = os.path.join(data_path, name)
        if os.path.exists(coco_path):
            return coco_path
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Roboflow Data Preprocess', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Determinar caminho do arquivo COCO
    if args.coco_file:
        coco_json_path = args.coco_file
    else:
        coco_json_path = find_coco_file(args.data_path, args.split)
    
    if not coco_json_path or not os.path.exists(coco_json_path):
        raise FileNotFoundError(
            f"Arquivo COCO não encontrado. Procurou em: {args.data_path}\n"
            f"Use --coco_file para especificar o caminho do arquivo JSON."
        )
    
    print(f"Arquivo COCO encontrado: {coco_json_path}")
    
    # Validar arquivo COCO
    if not validate_coco_file(coco_json_path):
        raise ValueError("Arquivo COCO inválido. Verifique o formato do arquivo.")
    
    # Diretórios
    split_dir = os.path.join(args.data_path, args.split)
    images_dir = os.path.join(split_dir, 'images')
    annotations_dir = os.path.join(split_dir, 'annotations')
    ground_truth_dir = os.path.join(split_dir, 'ground_truth')
    
    # Criar diretórios necessários
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {images_dir}")
    
    print(f"Processando split: {args.split}")
    print(f"Diretório de imagens: {images_dir}")
    print(f"Diretório de anotações: {annotations_dir}")
    print(f"Diretório de ground truth: {ground_truth_dir}")
    
    # Converter COCO para pontos centrais
    print("\nConvertendo anotações COCO para pontos centrais...")
    image_points_map = coco_to_center_points(
        coco_json_path, 
        annotations_dir,
        class_name=args.class_name
    )
    
    # Obter todas as imagens
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    
    print(f"\nEncontradas {len(img_paths)} imagens")
    print("Gerando mapas de densidade...")
    
    # Processar cada imagem
    processed_count = 0
    skipped_count = 0
    
    for img_path in img_paths:
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        
        # Caminho do arquivo de anotação (pontos centrais)
        txt_path = os.path.join(annotations_dir, f"{base_name}.txt")
        
        # Verificar se existe anotação para esta imagem
        if not os.path.exists(txt_path):
            print(f"Aviso: Anotação não encontrada para {img_filename}, pulando...")
            skipped_count += 1
            continue
        
        # Ler pontos centrais
        gt = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            x = float(parts[0])
                            y = float(parts[1])
                            gt.append([x, y])
        except Exception as e:
            print(f"Erro ao ler {txt_path}: {e}")
            skipped_count += 1
            continue
        
        # Carregar imagem para obter dimensões
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Erro ao carregar imagem: {img_path}")
                skipped_count += 1
                continue
        except Exception as e:
            print(f"Erro ao carregar imagem {img_path}: {e}")
            skipped_count += 1
            continue
        
        # Gerar mapa de densidade
        try:
            positions = generate_density_map(
                shape=image.shape, 
                points=np.array(gt) if len(gt) > 0 else np.array([]), 
                f_sz=15, 
                sigma=4
            )
            
            # Salvar mapa de densidade
            h5_filename = f"{base_name}_sigma4.h5"
            h5_path = os.path.join(ground_truth_dir, h5_filename)
            
            with h5py.File(h5_path, 'w') as hf:
                hf['density'] = positions
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processadas {processed_count} imagens...")
                
        except Exception as e:
            print(f"Erro ao processar {img_filename}: {e}")
            skipped_count += 1
            continue
    
    print(f"\n{'='*50}")
    print(f"Processamento concluído!")
    print(f"Imagens processadas: {processed_count}")
    print(f"Imagens puladas: {skipped_count}")
    print(f"{'='*50}")
    
    # Verificar estrutura final
    print(f"\nEstrutura criada:")
    print(f"  {images_dir}/")
    print(f"  {annotations_dir}/")
    print(f"  {ground_truth_dir}/")
    print(f"\nPronto para usar com o sistema SASNet!")
