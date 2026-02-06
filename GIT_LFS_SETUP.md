# Guia de Instalação do Git LFS

Este guia explica como instalar e configurar o Git LFS para versionar os arquivos de modelo.

## Passo 1: Instalar Git LFS

### Opção A: Usando Homebrew (Recomendado para macOS)

```bash
brew install git-lfs
```

Se você receber erro de permissões, execute:

```bash
sudo chown -R $(whoami) /usr/local/bin /usr/local/etc /usr/local/lib /usr/local/share
brew install git-lfs
```

### Opção B: Download Manual

1. Acesse: https://git-lfs.github.com/
2. Baixe o instalador para macOS
3. Execute o instalador
4. Siga as instruções na tela

### Opção C: Usando o Script Fornecido

```bash
./install_git_lfs.sh
```

## Passo 2: Verificar Instalação

```bash
git lfs version
```

Deve mostrar algo como: `git-lfs/3.x.x`

## Passo 3: Inicializar Git LFS

```bash
cd /Users/pedro/Documents/SASnet/CrowdCounting-SASNet
git lfs install
```

## Passo 4: Adicionar Modelos ao Git LFS

Os arquivos `.gitattributes` e `.gitignore` já foram configurados. Agora adicione os modelos:

```bash
# Adicionar arquivo de configuração
git add .gitattributes

# Adicionar modelos (já rastreados pelo Git LFS)
git add models/SHHA.pth
git add models/SHHB.pth

# Verificar se estão sendo rastreados
git lfs ls-files
```

## Passo 5: Commit e Push

```bash
git commit -m "Add model files using Git LFS"
git push origin main
```

## Verificação

Após o push, verifique no GitHub:
- Os arquivos devem aparecer como "Stored with Git LFS"
- O tamanho mostrado será pequeno (ponteiros)

## Troubleshooting

### Erro: "git: 'lfs' is not a git command"
- Git LFS não está instalado. Siga o Passo 1.

### Erro: "Permission denied"
- Execute com sudo ou corrija as permissões do Homebrew

### Modelos não aparecem como LFS
- Verifique se `.gitattributes` está commitado
- Execute: `git lfs track "*.pth"` novamente
- Re-adicione os arquivos: `git add models/*.pth`

## Limites do GitHub

- **Armazenamento gratuito**: 1 GB
- **Largura de banda**: 1 GB/mês
- **Modelos atuais**: ~298 MB (SHHA.pth + SHHB.pth)
- ✅ Dentro dos limites!
