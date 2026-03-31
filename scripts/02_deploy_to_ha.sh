#!/usr/bin/env bash
# ==============================================================================
# Script 02: Deploy da wake word "Letícia" no Home Assistant Yellow
# ==============================================================================
#
# Uso: ./02_deploy_to_ha.sh <caminho_do_modelo.tflite> [endereço_do_HA]
#
# Exemplo:
#   ./02_deploy_to_ha.sh ./leticia.tflite homeassistant.local
#   ./02_deploy_to_ha.sh ./leticia.tflite 192.168.1.100
#

set -euo pipefail

# --- Configuração ---
MODEL_FILE="${1:?Erro: Informe o caminho do modelo .tflite como primeiro argumento}"
HA_HOST="${2:-homeassistant.local}"
HA_SHARE_PATH="/share/openwakeword"
MODEL_NAME="leticia.tflite"

# --- Cores ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[AVISO]${NC} $1"; }
log_err()  { echo -e "${RED}[ERRO]${NC} $1"; }

echo "=============================================="
echo "  Deploy Wake Word 'Letícia' no HA Yellow"
echo "=============================================="

# --- Verificações ---
if [[ ! -f "$MODEL_FILE" ]]; then
    log_err "Arquivo não encontrado: $MODEL_FILE"
    exit 1
fi

if [[ ! "$MODEL_FILE" == *.tflite ]]; then
    log_warn "O arquivo não tem extensão .tflite. Tem certeza que é o modelo correto?"
fi

file_size=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat --printf="%s" "$MODEL_FILE" 2>/dev/null)
log_ok "Modelo encontrado: $MODEL_FILE ($(( file_size / 1024 )) KB)"

# --- Teste de conectividade ---
echo ""
echo "[1/3] Testando conexão com $HA_HOST..."
if ping -c 1 -W 3 "$HA_HOST" &>/dev/null; then
    log_ok "Home Assistant acessível em $HA_HOST"
else
    log_err "Não foi possível alcançar $HA_HOST"
    echo "    Verifique:"
    echo "    - O HA Yellow está ligado e na mesma rede?"
    echo "    - O hostname está correto? Tente o IP diretamente."
    exit 1
fi

# --- Transferência via SCP ---
echo ""
echo "[2/3] Transferindo modelo via SCP..."
echo "    Destino: root@${HA_HOST}:${HA_SHARE_PATH}/${MODEL_NAME}"
echo ""

# Tentar criar o diretório primeiro
ssh "root@${HA_HOST}" "mkdir -p ${HA_SHARE_PATH}" 2>/dev/null || true

# Copiar o arquivo
if scp "$MODEL_FILE" "root@${HA_HOST}:${HA_SHARE_PATH}/${MODEL_NAME}"; then
    log_ok "Modelo transferido com sucesso!"
else
    log_err "Falha na transferência SCP."
    echo ""
    echo "    Alternativa via Samba:"
    echo "    1. Abra o Finder"
    echo "    2. Cmd+K -> smb://${HA_HOST}/share"
    echo "    3. Crie a pasta 'openwakeword' se não existir"
    echo "    4. Copie ${MODEL_FILE} para lá renomeando para ${MODEL_NAME}"
    exit 1
fi

# --- Instruções pós-deploy ---
echo ""
echo "[3/3] Próximos passos no Home Assistant:"
echo ""
echo "    1. Reinicie o add-on openWakeWord:"
echo "       Settings > Add-ons > openWakeWord > Restart"
echo ""
echo "    2. Configure o Voice Assistant:"
echo "       Settings > Voice assistants > [Seu assistente]"
echo "       Wake word engine: openWakeWord"
echo "       Wake word: leticia"
echo ""
echo "    3. Associe ao dispositivo de voz:"
echo "       Settings > Devices & Services > [Seu dispositivo]"
echo "       Selecione o pipeline com wake word Letícia"
echo ""
echo "    4. Teste falando: 'Letícia, que horas são?'"
echo ""
echo "=============================================="
log_ok "Deploy concluído!"
echo "=============================================="
