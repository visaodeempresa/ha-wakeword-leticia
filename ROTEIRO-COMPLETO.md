# Wake Word "Letícia" para Home Assistant Yellow

## Visao Geral do Projeto

**Objetivo:** Criar uma wake word personalizada "Letícia" para o Home Assistant instalado no Home Assistant Yellow, usando o framework openWakeWord.

**Desafio principal:** openWakeWord foi projetado primariamente para inglês. "Letícia" é uma palavra em português, o que requer uma abordagem adaptada.

---

## Arquitetura do Sistema

```
[Dispositivo de Voz] --stream audio--> [Home Assistant Yellow]
                                            |
                                    [openWakeWord Add-on]
                                            |
                                    Detecta "Letícia"
                                            |
                                    [Assist Pipeline]
                                            |
                                    [STT -> Intent -> TTS]
```

O Home Assistant processa wake words no servidor (Yellow), não no dispositivo de voz. Qualquer dispositivo que faça streaming de áudio pode funcionar como satélite de voz.

---

## Estratégias de Treinamento (da mais simples à mais robusta)

### Estratégia A: Google Colab Oficial (Mais Simples - Recomendada para Começar)

**Tempo estimado:** ~1 hora de treinamento
**Requisitos:** Conta Google, navegador

#### Passo 1: Abrir o Colab Notebook

Acesse: https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb

#### Passo 2: Configurar a Wake Word

No campo `target_word`, use uma grafia fonética que o Piper (em inglês) consiga pronunciar de forma similar a "Letícia":

```
# Opções de grafia fonética para testar:
leh_tee_see_ah
leh_tee_sea_ah
le_tee_see_a
let_ee_see_ah
```

**Dica importante:** Execute a célula de pronúncia e ouça o resultado. Ajuste a grafia até que soe o mais próximo possível de "Letícia" com sotaque brasileiro. O acento tônico deve cair no "tee" (tí).

#### Passo 3: Treinar o Modelo

1. Selecione **Runtime > Run all** no menu do Colab
2. Aguarde ~1 hora (não feche a aba)
3. Dois arquivos serão baixados: `.tflite` e `.onnx`
4. Apenas o `.tflite` é necessário para o Home Assistant

#### Passo 4: Teste Inicial no Colab

Antes de implantar, teste o modelo no próprio Colab:
- Use a célula de teste para verificar se reconhece variações de pronúncia
- Verifique a taxa de falsos positivos

---

### Estratégia B: Colab com Vozes em Português (Intermediária - Melhor Qualidade)

Esta abordagem modifica o notebook do Colab para usar vozes Piper em pt_BR.

#### Vozes Piper pt_BR Disponíveis:
| Nome | Repositório |
|------|-------------|
| cadu | `pt/pt_BR/cadu` |
| edresson | `pt/pt_BR/edresson` |
| faber | `pt/pt_BR/faber` |
| jeff | `pt/pt_BR/jeff` |

#### Passo 1: Modificar o Notebook

No primeiro bloco de código do Colab, substituir o download do modelo TTS por:

```python
# Baixar modelo Piper pt_BR em vez do modelo inglês
import subprocess
import os

# Instalar dependências
subprocess.run(["pip", "install", "piper-tts"], check=True)

# Baixar modelo pt_BR (exemplo com faber-medium)
model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"
config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json"

os.makedirs("voices", exist_ok=True)
subprocess.run(["wget", "-O", "voices/pt_BR-faber-medium.onnx", model_url], check=True)
subprocess.run(["wget", "-O", "voices/pt_BR-faber-medium.onnx.json", config_url], check=True)
```

#### Passo 2: Configurar target_word

```python
target_word = "letícia"
```

Como agora estamos usando TTS em português, podemos escrever a palavra diretamente.

#### Passo 3: Gerar Amostras e Treinar

Seguir o mesmo fluxo do notebook, mas certifique-se de que:
- As amostras de áudio estão sendo reamostradas para 16kHz
- O modelo de embedding do Google ainda será usado (funciona razoavelmente para português)

**Limitação:** As vozes pt_BR do Piper são single-speaker (1 voz por modelo). Isso resulta em menor diversidade de vozes comparado ao inglês. Para contornar:
- Gere amostras com todos os 4 modelos pt_BR (cadu, edresson, faber, jeff)
- Varie `length_scales` (0.8 a 1.3) e `noise_scales` (0.5 a 0.8) para diversidade

---

### Estratégia C: Voice Conversion (Mais Robusta - Melhor Resultado)

Baseada na abordagem documentada em [dscripka/openWakeWord Discussion #266](https://github.com/dscripka/openWakeWord/discussions/266), que obteve sucesso com wake words em idiomas não-ingleses.

#### Fluxo de Trabalho:

```
1. Gerar ~1.000 amostras positivas ("Letícia") via Piper pt_BR
2. Gerar ~1.000 amostras negativas (palavras aleatórias em pt_BR)
3. Coletar áudios de falantes nativos (YouTube, podcasts, Common Voice)
4. Aplicar Voice Conversion (OpenVoice/FreeVC) para diversificar vozes
5. Resultado: ~300k+ amostras diversificadas
6. Treinar com pipeline padrão do openWakeWord
```

#### Ferramentas Necessárias:
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) - Para gerar amostras base
- [OpenVoice](https://github.com/myshell-ai/OpenVoice) ou [FreeVC](https://github.com/OlaWod/FreeVC) - Para voice conversion
- [Whisper](https://github.com/openai/whisper) - Para validar amostras geradas
- GPU recomendada (Google Colab Pro ou local com CUDA)

---

## Implantação no Home Assistant Yellow

### Pré-requisitos no HA Yellow

1. **Home Assistant OS** atualizado (versão mais recente)
2. **Add-on openWakeWord** instalado:
   - Settings > Add-ons > Add-on Store > openWakeWord
   - Ou: Settings > Devices & Services > Add Integration > Wyoming > openWakeWord
3. **Add-on Samba** instalado (para transferência de arquivos)
4. **Assist Pipeline** configurado com STT e TTS

### Passo 1: Transferir o Modelo

```bash
# Via Samba (rede local)
# Monte o share do HA Yellow no Finder/Explorer
# Navegue até: \\homeassistant.local\share\

# Criar diretório se não existir
mkdir -p /share/openwakeword/

# Copiar o modelo .tflite
cp leticia.tflite /share/openwakeword/
```

Ou via SSH:
```bash
# SSH no HA Yellow
ssh root@homeassistant.local

# Copiar o arquivo (via scp de outra máquina)
scp leticia.tflite root@homeassistant.local:/share/openwakeword/
```

### Passo 2: Reiniciar o Add-on openWakeWord

1. Settings > Add-ons > openWakeWord
2. Clique em **Restart**
3. Aguarde o add-on carregar o novo modelo

### Passo 3: Configurar o Assist Pipeline

1. Settings > Voice assistants
2. Crie um novo assistente ou edite um existente
3. Na seção **Wake word**:
   - Engine: openWakeWord
   - Wake word: selecione "leticia" na lista
4. Salve

### Passo 4: Associar ao Dispositivo de Voz

1. Settings > Devices & Services > ESPHome (ou o protocolo do seu dispositivo)
2. Selecione o dispositivo de voz (ex: ATOM Echo, ESP32-S3-BOX)
3. Configure o pipeline com a wake word "Letícia"

---

## Roteiro de Testes

### Teste 1: Detecção Básica
- [ ] Fale "Letícia" a 1 metro de distância - deve detectar
- [ ] Fale "Letícia" a 3 metros - deve detectar
- [ ] Fale "Letícia" a 5 metros - testar limite

### Teste 2: Variações de Pronúncia
- [ ] "Letícia" (pronúncia padrão)
- [ ] "Letíícia" (prolongado)
- [ ] "Leticia" (sem acentuação clara)
- [ ] Diferentes falantes (masculino/feminino/crianças)

### Teste 3: Falsos Positivos
- [ ] Fale "Patricia" - NÃO deve detectar
- [ ] Fale "Notícia" - NÃO deve detectar
- [ ] Fale "Delícia" - NÃO deve detectar
- [ ] Fale "Letícia" em contexto de conversa normal (não dirigido ao assistente)
- [ ] Deixe TV/rádio ligados por 1 hora - verificar falsos positivos

### Teste 4: Pipeline Completo
- [ ] "Letícia, que horas são?"
- [ ] "Letícia, acenda a luz da sala"
- [ ] "Letícia, qual a temperatura?"
- [ ] Verificar se o LED do dispositivo pisca azul ao detectar

### Teste 5: Condições Adversas
- [ ] Com música ambiente
- [ ] Com conversa ao fundo
- [ ] Com ruído de eletrodomésticos
- [ ] Em cômodos diferentes da casa

### Registro de Resultados

| Teste | Resultado | Taxa Detecção | Falsos Positivos | Observações |
|-------|-----------|---------------|------------------|-------------|
| Básico 1m | | /10 | | |
| Básico 3m | | /10 | | |
| Variações | | /10 | | |
| Falsos Pos. | | /hora | | |
| Pipeline | | /10 | | |
| Adverso | | /10 | | |

---

## Ajuste Fino (Se Necessário)

### Se muitos falsos negativos (não detecta):
1. Retreinar com mais amostras positivas
2. Ajustar threshold de detecção no add-on openWakeWord (diminuir)
3. Verificar qualidade do microfone do dispositivo

### Se muitos falsos positivos (detecta errado):
1. Aumentar threshold de detecção
2. Adicionar mais amostras negativas com palavras similares (Patrícia, Notícia, Delícia)
3. Retreinar com dataset mais diversificado

### Ajustar Threshold:
No add-on openWakeWord, edite a configuração:
```yaml
# Configuração do add-on openWakeWord
threshold: 0.5  # Padrão. Aumentar para menos falsos positivos, diminuir para mais sensibilidade
```

---

## Referências

- [Home Assistant - About Wake Words](https://www.home-assistant.io/voice_control/about_wake_word/)
- [Home Assistant - Create Wake Word](https://www.home-assistant.io/voice_control/create_wake_word/)
- [openWakeWord GitHub](https://github.com/dscripka/openWakeWord)
- [Colab Training Notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb)
- [openWakeWord - Multilingual Discussion](https://github.com/dscripka/openWakeWord/discussions/52)
- [openWakeWord - Voice Conversion Approach](https://github.com/dscripka/openWakeWord/discussions/266)
- [Piper Sample Generator](https://github.com/rhasspy/piper-sample-generator)
- [Piper Voices pt_BR](https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_BR)
- [French Wake Word Training Guide](https://community.home-assistant.io/t/guide-train-a-custom-french-wake-word-for-home-assistant-with-openwakeword-colab/943111)
- [Community Wake Words Collection](https://github.com/fwartner/home-assistant-wakewords-collection)
