# Copyright 2021 Tencent
# Modified to support Roboflow COCO format conversion

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
import json
import os
import cv2
from PIL import Image

def coco_to_center_points(coco_json_path, output_dir, class_name="person"):
    """
    Converte anotações COCO (bounding boxes) para pontos centrais.
    
    Args:
        coco_json_path: Caminho para o arquivo JSON do COCO
        output_dir: Diretório onde salvar os arquivos TXT com pontos centrais
        class_name: Nome da classe a filtrar (padrão: "person")
    
    Returns:
        dict: Mapeamento de nome da imagem para lista de pontos centrais
    """
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Ler arquivo COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Criar mapeamentos
    images_dict = {img['id']: img for img in coco_data.get('images', [])}
    categories_dict = {cat['id']: cat for cat in coco_data.get('categories', [])}
    
    # Encontrar ID da classe "person"
    person_category_id = None
    for cat_id, cat in categories_dict.items():
        if cat['name'].lower() == class_name.lower():
            person_category_id = cat_id
            break
    
    if person_category_id is None:
        # Se não encontrar "person", usar a primeira categoria disponível
        if categories_dict:
            person_category_id = list(categories_dict.keys())[0]
            print(f"Aviso: Classe '{class_name}' não encontrada. Usando '{categories_dict[person_category_id]['name']}'")
        else:
            raise ValueError("Nenhuma categoria encontrada no arquivo COCO")
    
    # Agrupar anotações por imagem
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        if ann['category_id'] == person_category_id:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
    
    # Converter bounding boxes para pontos centrais
    image_points_map = {}
    
    for image_id, annotations in annotations_by_image.items():
        if image_id not in images_dict:
            continue
        
        image_info = images_dict[image_id]
        image_filename = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        
        points = []
        for ann in annotations:
            # COCO format: [x_min, y_min, width, height] (coordenadas absolutas)
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            
            # Converter para ponto central
            x_center = x_min + width / 2.0
            y_center = y_min + height / 2.0
            
            # Garantir que está dentro dos limites da imagem
            x_center = max(0, min(img_width - 1, x_center))
            y_center = max(0, min(img_height - 1, y_center))
            
            points.append([x_center, y_center])
        
        image_points_map[image_filename] = points
    
    # Salvar pontos em arquivos TXT (formato esperado pelo sistema)
    for image_filename, points in image_points_map.items():
        # Nome do arquivo TXT (mesmo nome da imagem, mas com extensão .txt)
        base_name = os.path.splitext(image_filename)[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # Salvar pontos no formato: x y (um ponto por linha)
        with open(txt_path, 'w') as f:
            for x, y in points:
                f.write(f"{x} {y}\n")
    
    print(f"Convertidas {len(image_points_map)} imagens com {sum(len(p) for p in image_points_map.values())} anotações")
    
    return image_points_map


def get_image_dimensions(image_path):
    """
    Obtém as dimensões de uma imagem.
    
    Args:
        image_path: Caminho para a imagem
    
    Returns:
        tuple: (width, height)
    """
    try:
        img = Image.open(image_path)
        return img.size  # (width, height)
    except Exception as e:
        print(f"Erro ao ler imagem {image_path}: {e}")
        return None, None


def validate_coco_file(coco_json_path):
    """
    Valida se o arquivo COCO JSON é válido.
    
    Args:
        coco_json_path: Caminho para o arquivo JSON
    
    Returns:
        bool: True se válido, False caso contrário
    """
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Verificar estrutura básica
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in coco_data:
                print(f"Arquivo COCO inválido: falta chave '{key}'")
                return False
        
        if len(coco_data['images']) == 0:
            print("Arquivo COCO inválido: nenhuma imagem encontrada")
            return False
        
        print(f"Arquivo COCO válido: {len(coco_data['images'])} imagens, {len(coco_data['annotations'])} anotações")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Erro ao parsear JSON: {e}")
        return False
    except Exception as e:
        print(f"Erro ao validar arquivo COCO: {e}")
        return False
