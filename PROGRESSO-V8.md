# Progresso de Treinamento — v8

## Sessão atual
- **Notebook:** train_leticia_v8_gpu.ipynb
- **Branch:** develop
- **GPU:** T4 (15.0 GB VRAM)

## Histórico de estados

---

### [23:30] 🟢 Notebook aberto — sessão iniciada
- RAM sistema: 1.3 / 12.7 GB
- RAM GPU: 0.0 / 15.0 GB
- Disco: 42.9 / 112.6 GB (69.74 GB livres)
- Status: Aguardando execução da Etapa 0

**Próxima ação:** Executar Etapa 0 (Diagnóstico do Ambiente)

---

### [23:32] 🔐 Aviso de segurança do Colab
- Dialog: "Este notebook não é de autoria do Google"
- Ação: clicou **"Executar assim mesmo"**
- RAM sistema: 1.3 / 12.7 GB | GPU: 0.0 / 15.0 GB
- Etapa 0 iniciando execução


### [23:32] ✅ Etapa 0 — Diagnóstico concluído
```
Driver:  580.82.07  |  CUDA Version (driver): 13.0
nvcc:    12.8, V12.8.93  (toolkit real instalado)
Libs:    libcudart.so.12  ✅  |  libcudart.so.13  ❌ (não existe)
GPU:     Tesla T4 — 15360 MiB VRAM — 43°C — 0% util
```
**Diagnóstico:** ambiente perfeito para o Fix 1 — driver anuncia CUDA 13 mas
toolkit real é 12.8 e só existe `libcudart.so.12`.
PyTorch cu121 vai linkar em `.so.12` → sem OSError.

**Próxima ação:** Executar Etapa 1a (instalação de dependências)

