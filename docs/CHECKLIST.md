# Checklist de Implantação - Wake Word "Letícia"

## Fase 1: Preparação do Ambiente

- [ ] Verificar versão do Home Assistant OS no Yellow (Settings > About)
- [ ] Instalar add-on **Samba Share** (Settings > Add-ons > Add-on Store)
- [ ] Instalar add-on **openWakeWord** (Settings > Add-ons > Add-on Store)
- [ ] Verificar que o add-on openWakeWord está rodando (status: Started)
- [ ] Ter conta Google para acessar o Colab
- [ ] Dispositivo de voz (ATOM Echo / ESP32-S3-BOX / Voice PE) já pareado

## Fase 2: Treinamento do Modelo (Google Colab)

- [ ] Abrir notebook `notebooks/train_leticia_ptbr.ipynb` no Google Colab
- [ ] Verificar GPU ativada (Ambiente de execução > T4 GPU)
- [ ] **Etapa 1:** Instalação de dependências (fork dscripka/piper-sample-generator)
- [ ] **Etapa 2:** Ouvir pronúncia e ajustar target_word se necessário
- [ ] **Etapa 3:** Download de dados auxiliares (RIR, AudioSet, FMA, ACAV100M)
- [ ] **Etapa 4:** Geração de clips com vozes pt_BR (~10.000 positivos + ~10.000 negativos)
- [ ] **Etapa 5:** Augmentação e extração de features (~15-30 min)
- [ ] **Etapa 6:** Treinamento do modelo (~30-60 min)
- [ ] **Etapa 7:** Download do arquivo `leticia.tflite`

## Fase 3: Teste Local (Opcional mas Recomendado)

- [ ] Instalar dependências: `pip install openwakeword pyaudio numpy`
- [ ] Executar `python scripts/03_test_model.py --model models/leticia.tflite`
- [ ] Verificar detecção com pelo menos 3 falantes diferentes
- [ ] Verificar que palavras similares NÃO ativam (Patricia, Notícia, Delícia)
- [ ] Se resultado insatisfatório, voltar à Fase 2 e ajustar

## Fase 4: Deploy no Home Assistant Yellow

- [ ] Executar `./scripts/02_deploy_to_ha.sh models/leticia.tflite [IP_DO_HA]`
  - Ou copiar manualmente via Samba para `/share/openwakeword/leticia.tflite`
- [ ] Reiniciar add-on openWakeWord
- [ ] Verificar nos logs do add-on que o modelo foi carregado

## Fase 5: Configuração do Pipeline

- [ ] Settings > Voice assistants > Criar/Editar assistente
- [ ] Definir Wake word engine: **openWakeWord**
- [ ] Selecionar wake word: **leticia**
- [ ] Configurar STT (Whisper ou Cloud)
- [ ] Configurar TTS (Piper pt_BR ou Cloud)
- [ ] Associar pipeline ao dispositivo de voz

## Fase 6: Testes no Ambiente Real

- [ ] Teste básico: "Letícia" a 1 metro → LED pisca azul
- [ ] Teste de distância: 3 metros, 5 metros
- [ ] Teste pipeline completo: "Letícia, que horas são?"
- [ ] Teste com ruído ambiente (TV, música)
- [ ] Monitorar falsos positivos por 24 horas
- [ ] Ajustar threshold se necessário (add-on openWakeWord config)

## Fase 7: Refinamento

- [ ] Se falsos negativos > 30%: diminuir threshold ou retreinar
- [ ] Se falsos positivos > 2/hora: aumentar threshold ou retreinar com mais negativos
- [ ] Documentar threshold final escolhido
- [ ] Fazer backup do modelo final em `models/`

---

**Status:** [ ] Não iniciado | [~] Em andamento | [x] Concluído
