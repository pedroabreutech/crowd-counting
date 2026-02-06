# 🔧 Solução para Erro de Instalação do Git LFS

## Problema
O Homebrew não consegue instalar porque os diretórios em `/usr/local` não têm permissões de escrita.

## Solução 1: Corrigir Permissões do Homebrew (Recomendado)

Execute estes comandos no terminal:

```bash
# Corrigir propriedade dos diretórios
sudo chown -R pedro /usr/local/bin /usr/local/etc/bash_completion.d /usr/local/include /usr/local/lib /usr/local/lib/pkgconfig /usr/local/share /usr/local/share/doc /usr/local/share/man /usr/local/share/man/man1 /usr/local/share/man/man3 /usr/local/share/man/man5 /usr/local/share/man/man7 /usr/local/share/zsh /usr/local/share/zsh/site-functions

# Adicionar permissões de escrita
chmod u+w /usr/local/bin /usr/local/etc/bash_completion.d /usr/local/include /usr/local/lib /usr/local/lib/pkgconfig /usr/local/share /usr/local/share/doc /usr/local/share/man /usr/local/share/man/man1 /usr/local/share/man/man3 /usr/local/share/man/man5 /usr/local/share/man/man7 /usr/local/share/zsh /usr/local/share/zsh/site-functions

# Agora tente instalar novamente
brew install git-lfs
```

## Solução 2: Instalar Git LFS Manualmente (Alternativa)

Se a Solução 1 não funcionar, use o instalador oficial:

### Passo 1: Baixar o Instalador
1. Acesse: https://git-lfs.github.com/
2. Clique em "Download" para macOS
3. Baixe o arquivo `.pkg`

### Passo 2: Instalar
1. Abra o arquivo `.pkg` baixado
2. Siga o assistente de instalação
3. Complete a instalação

### Passo 3: Verificar
```bash
git lfs version
```

Deve mostrar: `git-lfs/3.x.x`

## Solução 3: Usar Homebrew com Diretório Alternativo

Se você tem Homebrew instalado em outro local (como `~/homebrew`):

```bash
# Verificar onde está o Homebrew
which brew

# Se estiver em ~/homebrew, use:
~/homebrew/bin/brew install git-lfs
```

## Após Instalar (Qualquer Método)

Depois de instalar o Git LFS com sucesso, execute:

```bash
cd /Users/pedro/Documents/SASnet/CrowdCounting-SASNet

# Inicializar Git LFS
git lfs install

# Adicionar modelos
git add models/SHHA.pth models/SHHB.pth

# Verificar
git lfs ls-files

# Commit e push
git commit -m "Add model files using Git LFS"
git push origin main
```

## Verificação

Para verificar se está funcionando:

```bash
# Verificar instalação
git lfs version

# Verificar arquivos rastreados
git lfs ls-files

# Deve mostrar:
# SHA256-xxx models/SHHA.pth
# SHA256-xxx models/SHHB.pth
```

## Nota Importante

Se você não conseguir instalar o Git LFS agora, os modelos ainda podem ser usados localmente. O Git LFS é necessário apenas para versionar os modelos no GitHub. O Streamlit Cloud baixará automaticamente os arquivos via Git LFS quando você fizer o push.
