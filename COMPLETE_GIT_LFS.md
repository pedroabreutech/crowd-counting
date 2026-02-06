# ✅ Configuração do Git LFS - Próximos Passos

## O que já foi feito:

✅ Arquivo `.gitattributes` criado (rastreia arquivos .pth)  
✅ Arquivo `.gitignore` atualizado (permite modelos em models/)  
✅ Script de instalação criado (`install_git_lfs.sh`)  
✅ Guia de instalação criado (`GIT_LFS_SETUP.md`)  
✅ Modelos encontrados: `models/SHHA.pth` e `models/SHHB.pth` (149MB cada)

## O que você precisa fazer agora:

### 1. Instalar Git LFS

**Opção A: Via Homebrew (Recomendado)**
```bash
brew install git-lfs
```

Se der erro de permissões:
```bash
sudo chown -R $(whoami) /usr/local/bin /usr/local/etc /usr/local/lib /usr/local/share
brew install git-lfs
```

**Opção B: Download Manual**
1. Acesse: https://git-lfs.github.com/
2. Baixe e instale o pacote para macOS

**Opção C: Usar o script**
```bash
cd /Users/pedro/Documents/SASnet/CrowdCounting-SASNet
./install_git_lfs.sh
```

### 2. Verificar Instalação

```bash
git lfs version
```

Deve mostrar: `git-lfs/3.x.x`

### 3. Inicializar Git LFS no Repositório

```bash
cd /Users/pedro/Documents/SASnet/CrowdCounting-SASNet
git lfs install
```

### 4. Adicionar os Modelos

```bash
# Adicionar arquivos de configuração (já estão staged)
git add .gitattributes .gitignore

# Adicionar modelos ao Git LFS
git add models/SHHA.pth
git add models/SHHB.pth

# Verificar se estão sendo rastreados
git lfs ls-files
```

Deve mostrar:
```
SHA256-sha256-hash models/SHHA.pth
SHA256-sha256-hash models/SHHB.pth
```

### 5. Commit e Push

```bash
git commit -m "Add model files using Git LFS"
git push origin main
```

## Verificação Final

Após o push, verifique no GitHub:
- Os arquivos devem aparecer como "Stored with Git LFS" 
- O tamanho mostrado será pequeno (apenas ponteiros)
- O Streamlit Cloud baixará automaticamente via Git LFS

## Limites do GitHub

- ✅ **Armazenamento**: 1 GB (você tem ~298 MB)
- ✅ **Largura de banda**: 1 GB/mês
- ✅ **Status**: Dentro dos limites!

## Troubleshooting

Se encontrar problemas, consulte `GIT_LFS_SETUP.md` para mais detalhes.
