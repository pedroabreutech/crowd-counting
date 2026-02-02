#!/usr/bin/env python3
# Script para organizar dataset Roboflow no formato esperado pelo sistema

import os
import shutil
import json
import argparse
from pathlib import Path

def find_coco_json(directory):
    """Encontra arquivo COCO JSON no diretório"""
    possible_names = [
        '_annotations.coco.json',
        'annotations.json',
        'train/_annotations.coco.json',
        'valid/_annotations.coco.json',
        'test/_annotations.coco.json',
    ]
    
    for name in possible_names:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    
    # Procurar recursivamente
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json') and ('coco' in file.lower() or 'annotation' in file.lower()):
                return os.path.join(root, file)
    
    return None

def find_roboflow_folder():
    """Procura a pasta do dataset Roboflow em vários locais possíveis"""
    possible_names = [
        'Visao Computacional.v20i.coco',
        'visao computacional.v20i.coco',
        'VisaoComputacional.v20i.coco',
    ]
    
    search_paths = [
        '.',
        './datas',
        '../',
        '../../',
    ]
    
    for search_path in search_paths:
        for name in possible_names:
            path = Path(search_path) / name
            if path.exists():
                return str(path.resolve())
    
    return None

def organize_roboflow_dataset(source_dir=None, target_dir=None):
    """
    Organiza dataset Roboflow da pasta 'Visao Computacional.v20i.coco' 
    para a estrutura esperada pelo sistema.
    
    Args:
        source_dir: Diretório fonte (pasta Visao Computacional.v20i.coco). Se None, procura automaticamente.
        target_dir: Diretório destino (padrão: ./datas/roboflow_dataset)
    """
    if target_dir is None:
        target_dir = os.path.join('datas', 'roboflow_dataset')
    
    # Se source_dir não fornecido, procurar automaticamente
    if source_dir is None:
        source_dir = find_roboflow_folder()
        if source_dir:
            print(f"✓ Pasta encontrada: {source_dir}")
        else:
            print("❌ Pasta 'Visao Computacional.v20i.coco' não encontrada.")
            print("   Por favor, forneça o caminho com --source")
            return False
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"❌ Diretório fonte não encontrado: {source_dir}")
        print(f"   Procurando em: {os.path.abspath('.')}")
        return False
    
    print(f"📁 Organizando dataset de: {source_dir}")
    print(f"📁 Para: {target_dir}")
    
    # Criar estrutura de diretórios
    splits = ['train', 'valid', 'test']
    for split in splits:
        (target_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_path / split / 'ground_truth').mkdir(parents=True, exist_ok=True)
        (target_path / split / 'annotations').mkdir(parents=True, exist_ok=True)
    
    # Organizar cada split (train, valid, test)
    print("\n📋 Organizando splits...")
    processed_splits = []
    
    for split in splits:
        split_source = source_path / split
        if split_source.exists():
            print(f"\n📁 Processando split: {split}")
            
            # Copiar imagens - Roboflow pode ter imagens diretamente no split ou em subpasta images/
            images_source = split_source / 'images'
            if not images_source.exists():
                # Se não existe subpasta images/, as imagens estão diretamente no split
                images_source = split_source
            
            img_count = 0
            images_target = target_path / split / 'images'
            print(f"  📋 Copiando imagens de {images_source}...")
            
            # Procurar imagens
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            for ext in image_extensions:
                for img_file in images_source.glob(f'*{ext}'):
                    if img_file.is_file():  # Garantir que é arquivo, não diretório
                        shutil.copy2(img_file, images_target / img_file.name)
                        img_count += 1
            
            if img_count > 0:
                print(f"  ✓ {img_count} imagens copiadas")
            else:
                print(f"  ⚠️ Nenhuma imagem encontrada")
            
            # Copiar arquivo COCO deste split
            coco_in_split = split_source / '_annotations.coco.json'
            if coco_in_split.exists():
                coco_target = target_path / split / '_annotations.coco.json'
                shutil.copy2(coco_in_split, coco_target)
                print(f"  ✓ Arquivo COCO copiado: {coco_in_split.name}")
                
                # Processar este split
                try:
                    with open(coco_target, 'r') as f:
                        coco_data = json.load(f)
                    num_images = len(coco_data.get('images', []))
                    num_annotations = len(coco_data.get('annotations', []))
                    print(f"  📊 {num_images} imagens, {num_annotations} anotações")
                    processed_splits.append(split)
                except Exception as e:
                    print(f"  ⚠️ Erro ao ler COCO: {e}")
            else:
                print(f"  ⚠️ Arquivo COCO não encontrado para {split}")
    
    # Se não encontrou estrutura padrão, tentar copiar tudo
    if not processed_splits:
        print("\n📋 Estrutura não padrão detectada. Copiando arquivos...")
        
        # Procurar todas as imagens
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        all_images = []
        for ext in image_extensions:
            all_images.extend(source_path.rglob(f'*{ext}'))
        
        if all_images:
            # Copiar para train (padrão)
            images_target = target_path / 'train' / 'images'
            for img_file in all_images:
                shutil.copy2(img_file, images_target / img_file.name)
            print(f"✓ {len(all_images)} imagens copiadas para train/images/")
        
        # Procurar JSON novamente
        json_files = list(source_path.rglob('*.json'))
        if json_files:
            # Copiar primeiro JSON encontrado como COCO
            coco_target = target_path / 'train' / '_annotations.coco.json'
            shutil.copy2(json_files[0], coco_target)
            print(f"✓ Arquivo JSON copiado: {json_files[0].name}")
    
    print(f"\n✅ Dataset organizado em: {target_dir}")
    print(f"\n📝 Próximos passos:")
    for i, split in enumerate(processed_splits, 1):
        print(f"{i}. Execute: python prepare_roboflow_dataset.py --data_path {target_dir} --split {split}")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organizar dataset Roboflow')
    parser.add_argument('--source', type=str, 
                       default=None,
                       help='Diretório fonte (pasta do Roboflow). Se não fornecido, procura automaticamente.')
    parser.add_argument('--target', type=str,
                       default='./datas/roboflow_dataset',
                       help='Diretório destino (padrão: ./datas/roboflow_dataset)')
    
    args = parser.parse_args()
    
    organize_roboflow_dataset(args.source, args.target)
