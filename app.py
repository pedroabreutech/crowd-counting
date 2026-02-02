import streamlit as st
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import SASNet
import argparse
import json
import os
import tempfile
from datasets.roboflow_converter import coco_to_center_points, validate_coco_file
from prepare_dataset import generate_density_map

# Configuração da página
st.set_page_config(
    page_title="SASNet - Contagem de Pessoas",
    page_icon="👥",
    layout="wide"
)

# Título da aplicação
st.title("👥 SASNet - Sistema de Contagem de Pessoas")
st.markdown("### Sistema de contagem de pessoas usando deep learning (SASNet - AAAI 2021)")

# Função para limpar memória
def clear_memory(device):
    """Limpa a memória do dispositivo"""
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    import gc
    gc.collect()

# Função para detectar dispositivo
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Função para carregar o modelo
@st.cache_resource
def load_model(model_path, device):
    """Carrega o modelo SASNet"""
    # Criar args simples para o modelo
    class Args:
        def __init__(self):
            self.block_size = 32
    
    args = Args()
    model = SASNet(args=args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Função para redimensionar imagem mantendo proporção
def resize_image_if_needed(image, max_size=2048):
    """Redimensiona imagem se for muito grande, mantendo proporção"""
    width, height = image.size
    original_size = (width, height)
    
    # Se a imagem for maior que max_size em qualquer dimensão, redimensionar
    if width > max_size or height > max_size:
        # Calcular novo tamanho mantendo proporção
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        return image, original_size, True
    
    return image, original_size, False

# Função para processar imagem com COCO (Roboflow)
def process_image_with_coco(image, coco_data, model, device, log_para=1000, max_image_size=2048):
    """Processa imagem usando anotações COCO do Roboflow"""
    # Redimensionar imagem se necessário
    image_resized, original_size, was_resized = resize_image_if_needed(image, max_image_size)
    
    img_width, img_height = image_resized.size
    
    # Procurar imagem no COCO pelo tamanho (match aproximado)
    matched_image_id = None
    for img_info in coco_data.get('images', []):
        if abs(img_info['width'] - img_width) < 50 and abs(img_info['height'] - img_height) < 50:
            matched_image_id = img_info['id']
            break
    
    # Se não encontrar, usar primeira imagem ou criar pontos vazios
    if matched_image_id is None and coco_data.get('images'):
        matched_image_id = coco_data['images'][0]['id']
    
    # Encontrar categoria "person"
    person_category_id = None
    for cat in coco_data.get('categories', []):
        if cat['name'].lower() == 'person':
            person_category_id = cat['id']
            break
    
    # Converter bounding boxes para pontos centrais
    points = []
    if matched_image_id and person_category_id:
        for ann in coco_data.get('annotations', []):
            if ann['image_id'] == matched_image_id and ann['category_id'] == person_category_id:
                bbox = ann['bbox']  # [x_min, y_min, width, height]
                # Ajustar coordenadas se a imagem foi redimensionada
                if was_resized:
                    scale_w = img_width / original_size[0]
                    scale_h = img_height / original_size[1]
                    bbox = [bbox[0] * scale_w, bbox[1] * scale_h, bbox[2] * scale_w, bbox[3] * scale_h]
                
                x_center = bbox[0] + bbox[2] / 2.0
                y_center = bbox[1] + bbox[3] / 2.0
                points.append([x_center, y_center])
    
    # Gerar mapa de densidade a partir dos pontos
    image_array = np.array(image_resized)
    if points:
        gt_density_map = generate_density_map(
            shape=image_array.shape,
            points=np.array(points),
            f_sz=15,
            sigma=4
        )
        gt_count = len(points)
    else:
        # Se não houver pontos, criar mapa vazio
        h, w = image_array.shape[:2]
        gt_density_map = np.zeros((h, w))
        gt_count = 0
    
    # Processar com o modelo (usar função base)
    count, pred_density_map, was_resized_model, original_size_model = process_image_base(
        image_resized, model, device, log_para
    )
    
    # Ajustar contagem se a imagem foi redimensionada
    if was_resized:
        scale_factor = (original_size[0] * original_size[1]) / (img_width * img_height)
        count = count * scale_factor
    
    return count, pred_density_map, was_resized, original_size, gt_count, gt_density_map

# Função base para processar imagem (sem redimensionamento)
def process_image_base(image, model, device, log_para=1000):
    """Processa uma imagem e retorna a contagem e o mapa de densidade (sem redimensionamento)"""
    # Transformações (mesmas do código original)
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    ])
    
    # Converter PIL para RGB se necessário
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Aplicar transformações
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Fazer predição
    with torch.no_grad():
        pred_map = model(img_tensor)
        pred_map = pred_map.data.cpu().numpy()
    
    # Limpar memória do tensor
    del img_tensor
    clear_memory(device)
    
    # Calcular contagem
    count = np.sum(pred_map) / log_para
    
    # Remover dimensões extras do mapa de densidade
    density_map = pred_map[0, 0]  # [H, W]
    
    # Limpar pred_map da memória
    del pred_map
    
    return count, density_map, False, image.size

# Função para processar imagem
def process_image(image, model, device, log_para=1000, max_image_size=2048):
    """Processa uma imagem e retorna a contagem e o mapa de densidade"""
    # Redimensionar se necessário para evitar problemas de memória
    image_resized, original_size, was_resized = resize_image_if_needed(image, max_image_size)
    
    # Usar função base
    count, density_map, _, _ = process_image_base(image_resized, model, device, log_para)
    
    # Ajustar contagem se a imagem foi redimensionada
    if was_resized:
        scale_factor = (original_size[0] * original_size[1]) / (image_resized.size[0] * image_resized.size[1])
        count = count * scale_factor
    
    return count, density_map, was_resized, original_size

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção do tipo de dataset
dataset_type = st.sidebar.radio(
    "Tipo de Dataset:",
    ["Original (ShanghaiTech)", "Roboflow (COCO)"],
    help="Escolha entre dataset original ou dataset do Roboflow"
)

# Seleção do modelo
model_option = st.sidebar.selectbox(
    "Selecione o modelo:",
    ["ShanghaiTech Part A (SHHA)", "ShanghaiTech Part B (SHHB)"],
    help="Part A é melhor para multidões densas, Part B para multidões esparsas"
)

# Mapear seleção para caminho do modelo
model_paths = {
    "ShanghaiTech Part A (SHHA)": "./models/SHHA.pth",
    "ShanghaiTech Part B (SHHB)": "./models/SHHB.pth"
}

selected_model_path = model_paths[model_option]

# Parâmetro log_para
log_para = st.sidebar.slider(
    "Parâmetro de escala (log_para):",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100,
    help="Fator de amplificação do mapa de densidade"
)

# Tamanho máximo da imagem (para evitar problemas de memória)
max_image_size = st.sidebar.slider(
    "Tamanho máximo da imagem (pixels):",
    min_value=512,
    max_value=4096,
    value=2048,
    step=256,
    help="Imagens maiores serão redimensionadas automaticamente para evitar problemas de memória"
)

# Aviso sobre memória
st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Dica de Memória:**
- Se encontrar erros de memória, reduza o tamanho máximo da imagem
- Imagens muito grandes (>3000px) podem causar problemas
- O sistema redimensiona automaticamente quando necessário
""")

# Verificar se o modelo existe
import os
if not os.path.exists(selected_model_path):
    st.sidebar.error(f"⚠️ Modelo não encontrado: {selected_model_path}")
    st.sidebar.info("Certifique-se de que o modelo está no diretório ./models/")
    st.stop()

# Carregar dispositivo e modelo
device = get_device()
try:
    model = load_model(selected_model_path, device)
    st.sidebar.success(f"✅ Modelo carregado! Dispositivo: {device}")
except Exception as e:
    st.sidebar.error(f"❌ Erro ao carregar modelo: {str(e)}")
    st.stop()

# Área principal
st.header("📤 Upload de Imagem")

# Se Roboflow selecionado, mostrar opção de upload COCO
coco_file = None
if dataset_type == "Roboflow (COCO)":
    st.sidebar.markdown("---")
    st.sidebar.header("📦 Dataset Roboflow")
    st.sidebar.info("""
    Para usar dataset Roboflow:
    1. Exporte seu dataset do Roboflow em formato COCO
    2. Faça upload do arquivo JSON abaixo
    3. Faça upload da imagem correspondente
    """)
    
    coco_file = st.sidebar.file_uploader(
        "Upload arquivo COCO JSON (opcional)",
        type=['json'],
        help="Arquivo de anotações no formato COCO do Roboflow"
    )
    
    if coco_file is not None:
        try:
            # Validar arquivo COCO
            coco_content = coco_file.read()
            coco_data = json.loads(coco_content)
            st.sidebar.success(f"✅ Arquivo COCO válido!")
            st.sidebar.info(f"Imagens: {len(coco_data.get('images', []))}\nAnotações: {len(coco_data.get('annotations', []))}")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao validar COCO: {str(e)}")

# Upload de arquivo de imagem
uploaded_file = st.file_uploader(
    "Faça upload de uma imagem para contagem",
    type=['jpg', 'jpeg', 'png'],
    help="Formatos suportados: JPG, JPEG, PNG"
)

# Upload opcional de ground truth (anotação)
st.sidebar.markdown("---")
st.sidebar.header("📝 Ground Truth (Opcional)")
upload_gt = st.sidebar.checkbox(
    "Fornecer contagem real para cálculo de acurácia",
    help="Marque esta opção se você souber o número real de pessoas na imagem"
)

gt_count = None
if upload_gt:
    gt_count = st.sidebar.number_input(
        "Número real de pessoas:",
        min_value=0,
        value=0,
        step=1,
        help="Digite o número real de pessoas na imagem para calcular a acurácia"
    )

if uploaded_file is not None:
    # Carregar imagem
    image = Image.open(uploaded_file)
    
    # Mostrar imagem original
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Imagem Original")
        st.image(image, use_container_width=True)
        st.caption(f"Tamanho: {image.size[0]} x {image.size[1]} pixels")
    
    # Processar imagem
    with st.spinner("🔄 Processando imagem..."):
        try:
            # Se Roboflow e COCO disponível, usar processamento com COCO
            if dataset_type == "Roboflow (COCO)" and coco_file is not None:
                try:
                    coco_file.seek(0)  # Resetar ponteiro do arquivo
                    coco_content = coco_file.read()
                    coco_data = json.loads(coco_content)
                    
                    count, density_map, was_resized, original_size, gt_count_coco, gt_density_map = process_image_with_coco(
                        image, coco_data, model, device, log_para, max_image_size
                    )
                    
                    # Usar contagem do COCO como ground truth se não foi fornecido manualmente
                    if not upload_gt:
                        gt_count = gt_count_coco
                        upload_gt = True
                    
                    # Mostrar informação sobre COCO
                    st.info(f"📦 Dataset Roboflow: {gt_count_coco} pessoas anotadas no COCO")
                    
                except Exception as e:
                    st.warning(f"⚠️ Erro ao processar com COCO: {str(e)}. Processando sem anotações COCO.")
                    count, density_map, was_resized, original_size = process_image(
                        image, model, device, log_para, max_image_size
                    )
            else:
                # Processamento normal (sem COCO)
                count, density_map, was_resized, original_size = process_image(
                    image, model, device, log_para, max_image_size
                )
            
            # Avisar se a imagem foi redimensionada
            if was_resized:
                st.warning(
                    f"⚠️ Imagem redimensionada de {original_size[0]}x{original_size[1]} "
                    f"para {image.size[0]}x{image.size[1]} pixels para evitar problemas de memória. "
                    f"A contagem foi ajustada proporcionalmente."
                )
            
            with col2:
                st.subheader("🗺️ Mapa de Densidade")
                
                # Criar visualização do mapa de densidade
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(density_map, cmap='jet', interpolation='nearest')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            # Calcular métricas se houver ground truth
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
            
            # Mostrar resultado
            st.markdown("---")
            st.header("📊 Resultado da Contagem")
            
            # Criar métricas em colunas
            if upload_gt and gt_count is not None:
                # Se houver ground truth, mostrar mais métricas
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        label="👥 Pessoas Detectadas",
                        value=f"{count:.0f}",
                        help="Número estimado de pessoas na imagem"
                    )
                
                with metric_col2:
                    st.metric(
                        label="✅ Contagem Real",
                        value=f"{gt_count:.0f}",
                        help="Número real de pessoas (ground truth)"
                    )
                
                with metric_col3:
                    delta = count - gt_count if gt_count is not None else None
                    st.metric(
                        label="📈 Diferença",
                        value=f"{delta:.0f}" if delta is not None else "N/A",
                        delta=f"{delta:.0f}" if delta is not None else None,
                        help="Diferença entre predição e valor real"
                    )
                
                with metric_col4:
                    st.metric(
                        label="🎯 Acurácia",
                        value=f"{accuracy:.2f}%" if accuracy is not None else "N/A",
                        help="Porcentagem de acerto"
                    )
            else:
                # Sem ground truth, mostrar apenas contagem
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        label="👥 Pessoas Detectadas",
                        value=f"{count:.0f}",
                        help="Número estimado de pessoas na imagem"
                    )
                
                with metric_col2:
                    st.metric(
                        label="📈 Contagem Precisa",
                        value=f"{count:.2f}",
                        help="Contagem com 2 casas decimais"
                    )
                
                with metric_col3:
                    # Calcular densidade (pessoas por pixel)
                    total_pixels = density_map.size
                    density_per_pixel = count / total_pixels * 1000000  # por milhão de pixels
                    st.metric(
                        label="📊 Densidade",
                        value=f"{density_per_pixel:.2f}",
                        help="Pessoas por milhão de pixels"
                    )
            
            # Seção de métricas detalhadas (se houver ground truth)
            if upload_gt and gt_count is not None and gt_count > 0:
                st.markdown("---")
                st.header("📈 Métricas de Avaliação")
                
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric(
                        label="MAE (Mean Absolute Error)",
                        value=f"{mae:.2f}",
                        help="Erro médio absoluto"
                    )
                
                with metrics_col2:
                    st.metric(
                        label="MSE (Mean Squared Error)",
                        value=f"{mse:.2f}",
                        help="Erro quadrático médio"
                    )
                
                with metrics_col3:
                    st.metric(
                        label="Erro Relativo",
                        value=f"{error_percent:.2f}%",
                        help="Porcentagem de erro em relação ao valor real"
                    )
                
                with metrics_col4:
                    st.metric(
                        label="RMSE (Root Mean Squared Error)",
                        value=f"{np.sqrt(mse):.2f}",
                        help="Raiz do erro quadrático médio"
                    )
                
                # Gráfico de comparação
                st.markdown("### 📊 Comparação Visual")
                fig_comparison, ax_comparison = plt.subplots(figsize=(8, 5))
                categories = ['Real', 'Predito']
                values = [gt_count, count]
                colors = ['#2ecc71', '#3498db']
                bars = ax_comparison.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
                ax_comparison.set_ylabel('Número de Pessoas', fontsize=12)
                ax_comparison.set_title('Comparação: Contagem Real vs Predita', fontsize=14, fontweight='bold')
                ax_comparison.grid(axis='y', alpha=0.3)
                
                # Adicionar valores nas barras
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax_comparison.text(bar.get_x() + bar.get_width()/2., height,
                                     f'{value:.1f}',
                                     ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig_comparison)
                plt.close(fig_comparison)
                
                # Indicador de qualidade
                st.markdown("### 🎯 Indicador de Qualidade")
                if accuracy >= 95:
                    quality_status = "🟢 Excelente"
                    quality_color = "green"
                elif accuracy >= 90:
                    quality_status = "🟡 Muito Bom"
                    quality_color = "orange"
                elif accuracy >= 80:
                    quality_status = "🟠 Bom"
                    quality_color = "darkorange"
                elif accuracy >= 70:
                    quality_status = "🔴 Regular"
                    quality_color = "red"
                else:
                    quality_status = "⚫ Precisa Melhorar"
                    quality_color = "darkred"
                
                st.markdown(f"""
                <div style="background-color: {quality_color}; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: white; margin: 0;">{quality_status}</h3>
                    <p style="color: white; margin: 5px 0 0 0;">Acurácia: {accuracy:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Informações adicionais
            with st.expander("ℹ️ Informações Técnicas"):
                st.write(f"**Dispositivo usado:** {device}")
                st.write(f"**Modelo:** {model_option}")
                st.write(f"**Parâmetro log_para:** {log_para}")
                st.write(f"**Formato do mapa de densidade:** {density_map.shape}")
                st.write(f"**Valor máximo no mapa:** {np.max(density_map):.4f}")
                st.write(f"**Valor médio no mapa:** {np.mean(density_map):.4f}")
                
                if upload_gt and gt_count is not None:
                    st.write("---")
                    st.write("**Métricas de Avaliação:**")
                    if mae is not None:
                        st.write(f"- **MAE:** {mae:.2f}")
                    if mse is not None:
                        st.write(f"- **MSE:** {mse:.2f}")
                        st.write(f"- **RMSE:** {np.sqrt(mse):.2f}")
                    if error_percent is not None:
                        st.write(f"- **Erro Relativo:** {error_percent:.2f}%")
                    if accuracy is not None:
                        st.write(f"- **Acurácia:** {accuracy:.2f}%")
            
        except Exception as e:
            st.error(f"❌ Erro ao processar imagem: {str(e)}")
            st.exception(e)

else:
    # Instruções quando não há imagem
    st.info("👆 Faça upload de uma imagem acima para começar a contagem de pessoas.")
    
    # Exemplo de uso
    with st.expander("📖 Como usar"):
        st.markdown("""
        1. **Selecione o modelo** na barra lateral:
           - **Part A**: Melhor para multidões muito densas
           - **Part B**: Melhor para multidões esparsas
        
        2. **Faça upload de uma imagem** usando o botão acima
        
        3. **Aguarde o processamento** - o sistema irá:
           - Carregar e processar a imagem
           - Gerar um mapa de densidade
           - Calcular o número de pessoas
        
        4. **Visualize os resultados**:
           - Contagem de pessoas
           - Mapa de densidade (cores mais quentes = mais pessoas)
           - Estatísticas adicionais
        """)
    
    # Informações sobre o modelo
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Sobre o SASNet")
    st.sidebar.info("""
    SASNet (Scale-Adaptive Selection Network) é um modelo de deep learning 
    para contagem de pessoas em imagens, apresentado na AAAI 2021.
    
    O modelo usa seleção adaptativa de escalas para lidar com diferentes 
    densidades de multidões.
    """)
