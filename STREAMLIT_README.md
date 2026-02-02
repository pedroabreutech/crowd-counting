# Interface Streamlit - Contagem de Pessoas

Interface web para contagem de pessoas em imagens usando o modelo SASNet.

## Como Executar

### Opção 1: Usando o script (recomendado)

```bash
./run_streamlit.sh
```

### Opção 2: Manualmente

```bash
source venv/bin/activate
streamlit run app.py
```

## Funcionalidades

- ✅ Upload de imagens (JPG, JPEG, PNG)
- ✅ Seleção de modelo (Part A ou Part B)
- ✅ Visualização do mapa de densidade
- ✅ Contagem precisa de pessoas
- ✅ Estatísticas detalhadas

## Requisitos

- Ambiente virtual ativado (`venv`)
- Modelos pré-treinados em `./models/`:
  - `SHHA.pth` (ShanghaiTech Part A)
  - `SHHB.pth` (ShanghaiTech Part B)

## Uso

1. Execute o comando acima
2. Abra o navegador na URL exibida (geralmente `http://localhost:8501`)
3. Selecione o modelo na barra lateral
4. Faça upload de uma imagem
5. Visualize os resultados!

## Notas

- **Part A**: Melhor para multidões muito densas
- **Part B**: Melhor para multidões esparsas
- O processamento pode levar alguns segundos dependendo do tamanho da imagem e do dispositivo usado
