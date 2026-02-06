#!/bin/bash
# Script para corrigir permissões do Homebrew e instalar Git LFS

echo "🔧 Corrigindo permissões do Homebrew..."
echo "⚠️  Este script requer permissões de administrador (sudo)"
echo ""

# Corrigir propriedade dos diretórios
echo "1. Corrigindo propriedade dos diretórios..."
sudo chown -R pedro /usr/local/bin /usr/local/etc/bash_completion.d /usr/local/include /usr/local/lib /usr/local/lib/pkgconfig /usr/local/share /usr/local/share/doc /usr/local/share/man /usr/local/share/man/man1 /usr/local/share/man/man3 /usr/local/share/man/man5 /usr/local/share/man/man7 /usr/local/share/zsh /usr/local/share/zsh/site-functions

if [ $? -eq 0 ]; then
    echo "✅ Propriedade corrigida com sucesso!"
else
    echo "❌ Erro ao corrigir propriedade"
    exit 1
fi

# Adicionar permissões de escrita
echo ""
echo "2. Adicionando permissões de escrita..."
chmod u+w /usr/local/bin /usr/local/etc/bash_completion.d /usr/local/include /usr/local/lib /usr/local/lib/pkgconfig /usr/local/share /usr/local/share/doc /usr/local/share/man /usr/local/share/man/man1 /usr/local/share/man/man3 /usr/local/share/man/man5 /usr/local/share/man/man7 /usr/local/share/zsh /usr/local/share/zsh/site-functions

if [ $? -eq 0 ]; then
    echo "✅ Permissões de escrita adicionadas!"
else
    echo "❌ Erro ao adicionar permissões"
    exit 1
fi

# Instalar Git LFS
echo ""
echo "3. Instalando Git LFS..."
brew install git-lfs

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Git LFS instalado com sucesso!"
    echo ""
    echo "4. Inicializando Git LFS..."
    git lfs install
    
    if [ $? -eq 0 ]; then
        echo "✅ Git LFS inicializado!"
        echo ""
        echo "📋 Próximos passos:"
        echo "   cd /Users/pedro/Documents/SASnet/CrowdCounting-SASNet"
        echo "   git add models/SHHA.pth models/SHHB.pth"
        echo "   git commit -m 'Add model files using Git LFS'"
        echo "   git push origin main"
    else
        echo "❌ Erro ao inicializar Git LFS"
        exit 1
    fi
else
    echo "❌ Erro ao instalar Git LFS"
    echo ""
    echo "💡 Alternativa: Use o instalador manual de https://git-lfs.github.com/"
    exit 1
fi
