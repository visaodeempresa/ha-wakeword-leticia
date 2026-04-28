# Etapa 0 — Diagnóstico do Ambiente

## Horário
- Início: 23:32 (28/04/2026)
- Duração: ~1s

## Recursos no início
| Recurso | Usado | Total |
|---------|-------|-------|
| RAM sistema | 1.3 GB | 12.7 GB |
| RAM GPU | 0.0 GB | 15.0 GB |
| Disco | 42.9 GB | 112.6 GB |

## Resultado
| Item | Valor | Status |
|------|-------|--------|
| Driver NVIDIA | 580.82.07 | ✅ |
| CUDA (driver) | 13.0 | ℹ️ (apenas versão suportada pelo driver) |
| CUDA toolkit (nvcc) | 12.8 | ✅ (toolkit real instalado) |
| libcudart.so.12 | presente | ✅ Fix 1 ativado |
| libcudart.so.13 | ausente | ✅ confirma necessidade do cu121 |
| GPU | Tesla T4 15360 MiB | ✅ |
| Temperatura | 43°C | ✅ normal |

## Análise de risco
- **ZERO risco**: ambiente idêntico ao previsto pelo Fix 1.
  Driver anuncia CUDA 13 mas toolkit é 12.8 e só existe libcudart.so.12.
  PyTorch cu121 vai linkar em .so.12 sem OSError.

## Aprendizado para estimativas futuras
- Etapa 0 demora ~1-2 segundos
- Padrão observado: driver CUDA sempre 1 versão acima do toolkit no Colab T4
