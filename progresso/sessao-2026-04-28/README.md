# Sessão de Treinamento — 28/04/2026

## Contexto
- Notebook: train_leticia_v8_gpu.ipynb
- Objetivo: treinar wake word "Letícia" para Home Assistant Yellow
- GPU: Tesla T4 (15.0 GB VRAM)
- Início: ~23:30

## Estrutura
Cada subpasta corresponde a uma etapa do notebook:
- `etapa-00/` — Diagnóstico CUDA
- `etapa-01/` — Instalação de dependências (v8 — 8 fixes)
- `etapa-02/` — Teste de pronúncia
- `etapa-03/` — Download de dados (RIRs, FMA, ACAV100M 17GB)
- `etapa-04/` — Geração de clips pt_BR com Piper
- `etapa-05/` — Augmentação + extração de features
- `etapa-06/` — Treinamento do modelo
- `etapa-07/` — Conversão TFLite + download

## Arquivos por etapa
- `output.txt` — saída completa do terminal
- `analise.md` — análise de recursos, tempo, riscos e aprendizados
- `screenshot-HHhMM.png` — prints enviados pelo usuário (quando disponível)
