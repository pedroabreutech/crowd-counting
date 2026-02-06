#!/bin/bash
# Script para instalar Git LFS no macOS

echo "Instalando Git LFS..."

# Verificar se Homebrew está instalado
if command -v brew &> /dev/null; then
    echo "Usando Homebrew para instalar Git LFS..."
    brew install git-lfs
else
    echo "Homebrew não encontrado. Por favor, instale manualmente:"
    echo "1. Acesse: https://git-lfs.github.com/"
    echo "2. Baixe o instalador para macOS"
    echo "3. Execute o instalador"
    exit 1
fi

# Inicializar Git LFS
echo "Inicializando Git LFS..."
git lfs install

echo "✅ Git LFS instalado e configurado!"
