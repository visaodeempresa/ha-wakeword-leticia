# Progresso do Treinamento — Wake Word "Letícia"

> Sessão: 30/03/2026 (noite)  
> Notebook: `train_leticia_v5.ipynb` (auditado)  
> Ambiente: Google Colab, GPU T4, 112.6 GB disco

---

## Status Geral

| Etapa | Descrição | Status | Horário | Observação |
|-------|-----------|--------|---------|------------|
| 1a | Instalação de dependências | ✅ Concluída | 23:09-23:18 | Avisos de deps são normais |
| — | Reiniciar sessão | ✅ Feito | 23:19 | — |
| 1b | Download de modelos | ✅ Concluída | 23:20-23:21 | Embedding v0.5.1, LibriTTS v2, vozes pt_BR |
| 2 | Testar pronúncia | ✅ Concluída | 23:22 | Faber e Edresson geraram áudio |
| 3 | Download de dados auxiliares | ✅ Concluída | 23:36-23:41 | MIT RIRs 270, FMA 120, ACAV100M 16.1GB, Val 176MB |
| 4 | Gerar clips (Piper pt_BR) | ⏳ Pendente | — | ~20-40 min |
| 5 | Augmentação + features | ⏳ Pendente | — | ~15-30 min |
| 6 | Treinar modelo | ⏳ Pendente | — | ~30-60 min |
| 7 | Verificar + baixar | ⏳ Pendente | — | ~1 min |

---

## Detalhes por Etapa

### Etapa 1a — Instalação (23:09 → 23:18)
- Clonados: `dscripka/piper-sample-generator`, `dscripka/openWakeWord`
- Pacotes pip instalados com avisos de deps (normais — conflitos com jax, opencv etc. não afetam nosso pipeline)
- Tempo: ~9 min

### Etapa 1b — Downloads de modelos (23:20 → 23:21)
- `embedding_model.onnx` ✅
- `embedding_model.tflite` ✅
- `melspectrogram.onnx` ✅
- `melspectrogram.tflite` ✅
- LibriTTS v2 (`en_US-libritts_r-medium.pt`) ✅
- Voz Faber (`pt_BR-faber-medium.onnx`) ✅
- Voz Edresson (`pt_BR-edresson-low.onnx`) ✅
- numpy 1.26.4: compatível ✅

### Etapa 2 — Teste de Pronúncia (23:22)
- Voz Faber: ▶ áudio gerado ✅
- Voz Edresson: ▶ áudio gerado ✅

### Etapa 3 — Download de Dados (23:36 → ...)

**Problema encontrado:** `ReadTimeout` ao tentar `load_dataset` sem autenticação.  
**Solução:** Criado HF_TOKEN (tipo Read) e adicionado como secret no Colab.  
**Resultado:** Após token, downloads funcionando.

| Sub-etapa | Dataset | Método | Status |
|-----------|---------|--------|--------|
| 3a | MIT RIRs (270 WAV) | `load_dataset` | ✅ 270 arquivos, 1m21s |
| 3b | AudioSet (tar) | `wget` | ⚠️ 0 clips (tar path issue, não bloqueante) |
| 3c | FMA (1 hora) | `load_dataset` | ✅ 120 clips, 39s |
| 3d | ACAV100M (16.1 GB) | `wget` | ✅ 16.1 GB em 2m0s a 68 MB/s |
| 3e | Validation (176.3 MB) | `wget` | ✅ 176.3 MB em 0.5s |

### Etapa 4 — Geração de Clips
*Pendente*

### Etapa 5 — Augmentação
*Pendente*

### Etapa 6 — Treinamento
*Pendente*

### Etapa 7 — Modelo Final
*Pendente*

---

## Problemas Encontrados e Resoluções

| # | Problema | Causa | Solução | Resolvido? |
|---|----------|-------|---------|-----------|
| 1 | `DataFilesNotFoundError` nos notebooks v1-v4 | Splits inventados para `openwakeword_features` | Usar `wget` direto + datasets separados | ✅ |
| 2 | `ReadTimeout` na Etapa 3 | Sem autenticação HuggingFace | Criar HF_TOKEN e adicionar no Colab | ✅ |

---

## Recursos do Ambiente

| Recurso | Disponível | Usado |
|---------|-----------|-------|
| RAM Sistema | 12.7 GB | ~1.9 GB |
| RAM GPU | 15.0 GB | ~0 GB |
| Disco | 112.6 GB | ~52.6 GB |
| Disco livre | ~60 GB | Suficiente para ACAV100M (17.3 GB) |
