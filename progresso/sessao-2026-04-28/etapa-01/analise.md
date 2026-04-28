# Etapa 1a — Instalação de Dependências

## Horário
- Execução: ~23:33–23:40 (28/04/2026)
- Duração estimada: ~7 min

## Resultado por fase

| Fase | Status | Observação |
|------|--------|------------|
| tree (apt) | ✅ | instalado 2.0.2 |
| piper-sample-generator | ✅ | clonado |
| openWakeWord | ✅ | clonado |
| PyTorch cu128 removido | ✅ | torch 2.10.0+cu128 → removido |
| PyTorch cu121 instalado | ✅ | torch 2.4.0+cu121, torchaudio 2.4.0+cu121 |
| torchmetrics, speechbrain, etc | ✅ | instalados |
| pronouncing | ✅ | instalado (Fix 2) |
| openwakeword --no-deps | ✅ | instalado |
| onnxruntime==1.16.3 | ❌ | versão removida do PyPI — mín. disponível: 1.17.0 |
| tensorflow-cpu==2.15.0 | ❌ | sem suporte Python 3.12 — mín. disponível: 2.16.0 |
| speexdsp-ns | ⚠️ | não instalado — requerido por openwakeword 0.6.0 |

## Warnings não críticos (ignorar)
- timm / fastai precisam de torchvision → não usados pelo pipeline
- protobuf conflicts → não afeta treinamento
- numpy 1.26.4 vs >=2.0 → não afeta treinamento de wake word

## Análise de risco

### ⚡ Alto — onnxruntime e tensorflow-cpu (Etapa 7)
- onnxruntime não instalado na versão correta → pode afetar feature extraction (Etapa 5) e export (Etapa 7)
- tensorflow-cpu não instalado → Etapa 7 (conversão TFLite) vai FALHAR
- **Ação v9:** trocar `onnxruntime==1.16.3` → `onnxruntime>=1.17.0` e `tensorflow-cpu==2.15.0` → `tensorflow-cpu>=2.16.0`

### ⚠️ Médio — speexdsp-ns ausente
- openwakeword 0.6.0 lista como dependência Linux
- Pode causar ImportError ao importar openwakeword
- **Ação v9:** adicionar `!pip install -q speexdsp-ns` na Fase 3

### ⚠️ Médio — huggingface-hub 0.36.2 (antigo)
- transformers 5.0.0 quer >=1.3.0
- Pode afetar downloads de datasets no Etapa 3
- **Ação v9:** adicionar `!pip install -q huggingface-hub>=1.3.0` na Fase 3

## Aprendizado para estimativas futuras
- Etapa 1a demora ~7-10 min com Colab gratuito
- Colab Python 3.12 não suporta tensorflow-cpu<2.16.0 (breaking change)
- onnxruntime 1.16.3 foi dropado do PyPI — fixar em >=1.17.0
- Colab pré-instala torch 2.10.0+cu128 (não cu130 como antes — ecossistema evoluiu)

## Próxima ação
- ✅ Reiniciar sessão (conforme solicitado pelo notebook)
- Executar Etapa 1b e verificar se torch cu121 + todas as deps estão OK
