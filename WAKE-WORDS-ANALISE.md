# Análise de Wake Words — Nomes Femininos para Assistente de Casa (pt-BR)

> **Contexto:** Após 11 versões de notebook e múltiplas sessões tentando treinar "Letícia" via openWakeWord + Piper pt-BR no Google Colab, este documento avalia o cenário atual, alternativas e recomendação de nome.

---

## 1. Estado atual do ecossistema de wake words em pt-BR

### openWakeWord (framework atual)
| Item | Status |
|------|--------|
| Suporte oficial pt-BR | ❌ Não existe |
| Modelos comunitários pt-BR | ❌ Nenhum encontrado (coleção fwartner: inglês, alemão, espanhol) |
| Pipeline de treino pt-BR | ⚠️ Possível com Piper TTS, mas sem validação pela comunidade |
| Estabilidade do pipeline | ❌ Altamente instável — 11 versões de notebook, erros em cascata |
| Tempo por tentativa | ⏱️ 2-3 horas por treino completo (Colab gratuito) |
| Documentação pt-BR | ❌ Inexistente |

**Conclusão:** Somos pioneiros em território sem trilha. Ninguém na comunidade validou openWakeWord com pt-BR com sucesso publicado.

### Alternativas disponíveis hoje
| Engine | pt-BR | Grátis | Privacidade | Integração HA | Tempo para funcionar |
|--------|-------|--------|-------------|---------------|----------------------|
| **openWakeWord** | ⚠️ Experimental | ✅ Sim | ✅ 100% local | ✅ Nativo | 2-3h por tentativa (instável) |
| **Porcupine (Picovoice)** | ✅ Suportado | ⚠️ Freemium (3 grátis) | ✅ Roda local | ✅ Add-on comunidade | **~5 minutos** |
| **microWakeWord (ESPHome)** | ❌ Só inglês | ✅ Sim | ✅ 100% local | ✅ Nativo ESP32 | N/A — sem pt-BR |
| **Snowboy** | ❌ Descontinuado | — | — | — | — |
| **Whisper (STT)** | ✅ Sim | ✅ Sim | ✅ Local | ✅ Via add-on | Pesado demais para wake word |

---

## 2. Nomes femininos recomendados como wake word

### Critérios de avaliação
- **Silabas:** 2-3 sílabas é o ideal (não muito curto, não muito longo)
- **Distintividade fonética:** sons que raramente aparecem no português cotidiano
- **Vogal tônica clara:** facilita a detecção pelo modelo
- **Consoante inicial marcante:** diferencia de sons ambientes
- **Homofonia:** evitar nomes que rimam com palavras comuns

### Tabela principal

| # | Nome | Pronúncia | Sílabas | Nota fonética | Risco de falso positivo | Recomendação |
|---|------|-----------|---------|---------------|------------------------|--------------|
| 1 | **Íris** | /ˈi.ɾis/ | 2 | Î inicial forte, terminação em S | 🟢 Muito baixo | ⭐⭐⭐⭐⭐ |
| 2 | **Nádia** | /ˈna.dʒi.ɐ/ | 3 | Cluster NDJ único | 🟢 Muito baixo | ⭐⭐⭐⭐⭐ |
| 3 | **Vera** | /ˈve.ɾɐ/ | 2 | V inicial, ER tônico | 🟡 Baixo | ⭐⭐⭐⭐ |
| 4 | **Lara** | /ˈla.ɾɐ/ | 2 | L+A limpos, fonética clara | 🟡 Baixo | ⭐⭐⭐⭐ |
| 5 | **Nina** | /ˈni.nɐ/ | 2 | Repetição NI-NA, muito distinto | 🟢 Muito baixo | ⭐⭐⭐⭐ |
| 6 | **Zara** | /ˈza.ɾɐ/ | 2 | Z inicial incomum em falas | 🟢 Muito baixo | ⭐⭐⭐⭐ |
| 7 | **Letícia** | /le.ˈtʃi.sjɐ/ | 4 | TCI distintivo, já em treino | 🟡 Baixo | ⭐⭐⭐ |
| 8 | **Luna** | /ˈlu.nɐ/ | 2 | LU inicial suave | 🟡 Baixo | ⭐⭐⭐ |
| 9 | **Clara** | /ˈklɑ.ɾɐ/ | 2 | KL cluster bom | 🟡 Médio ("é claro") | ⭐⭐⭐ |
| 10 | **Sofia** | /so.ˈfi.ɐ/ | 3 | Tônica no FI | 🟡 Médio (nome comum) | ⭐⭐⭐ |
| 11 | **Mira** | /ˈmi.ɾɐ/ | 2 | MI+R distinto | 🟡 Médio ("mira" = verbo) | ⭐⭐ |
| 12 | **Dara** | /ˈda.ɾɐ/ | 2 | D+A claro | 🟠 Alto ("dá" + "ra") | ⭐⭐ |
| 13 | **Ana** | /ˈɐ.nɐ/ | 2 | Muito curto, vogal inicial | 🔴 Muito alto | ❌ |
| 14 | **Maria** | /mɐ.ˈɾi.ɐ/ | 3 | Extremamente comum | 🔴 Muito alto | ❌ |

### Vencedoras por cenário

| Cenário | Nome sugerido | Motivo |
|---------|--------------|--------|
| **Melhor opção geral** | **Íris** | Fonética única, curta, sem homofonia com palavras comuns |
| **Mais agradável ao ouvido** | **Nádia** | Soa natural como nome de assistente |
| **Mais fácil de treinar** | **Lara** | Simples, vogais abertas, menos ambiguidade fonética |
| **Continuidade do projeto atual** | **Letícia** | Já em treino; 4 sílabas ajuda no reconhecimento |

---

## 3. Análise: vale continuar com openWakeWord pt-BR?

### Histórico desta tentativa
| Sessão | Problema | Versão do notebook |
|--------|----------|-------------------|
| 1 | CUDA 13 vs 12 (`libcudart.so.13`) | v6 |
| 2 | torchmetrics não instalado | v7 |
| 3 | pronouncing não instalado | v7 |
| 4 | onnxscript/beartype/torchvision | v7 |
| 5 | FMA sample rate incorreto | v7 |
| 6 | Features parciais bloqueavam regeneração | v7 |
| 7 | Clips deletados junto com features | v7 |
| 8 | onnx==1.14.1 sem wheel py3.12 | v8→v9 |
| 9 | datasets 2.14.6 conflito hub>=1.x | v9→v10 |
| 10 | pip_instalar() usado para uninstall (bug) | v10→v11 |
| 11 | torchvision RuntimeError não capturada | v10→v11 |

### Prós e Contras de continuar

**✅ Prós:**
- Totalmente gratuito e sem limites
- 100% local — privacidade total
- Nenhuma conta/API necessária
- Integração nativa com Home Assistant
- Aprendizado técnico valioso
- Possibilidade de qualquer nome, em pt-BR real

**❌ Contras:**
- Nenhum precedente de sucesso público com pt-BR
- Pipeline altamente instável (11 versões, ainda não chegou ao Etapa 5)
- 2-3h por tentativa no Colab gratuito
- Dependências quebram a cada versão do Colab
- O Colab gratuito pode desconectar durante downloads longos (17 GB ACAV100M)
- Cada nova sessão recomeça do zero (Colab é efêmero)

### Alternativa Porcupine — avaliação rápida

```
1. Criar conta gratuita em console.picovoice.ai
2. Digitar o nome (ex: "Íris" ou "Nádia")
3. Selecionar idioma: Portuguese (Brazil)
4. Baixar o arquivo .ppn em ~30 segundos
5. Instalar add-on PorcupinePipeline no HA
6. Apontar para o arquivo .ppn
Total: ~15 minutos
```

| | openWakeWord pt-BR | Porcupine |
|--|-------------------|-----------|
| Tempo até funcionar | Semanas (se funcionar) | 15 minutos |
| Custo | Grátis | Grátis (até 3 wake words) |
| Privacidade | ✅ Local | ✅ Roda local (modelo fechado) |
| Qualidade do modelo | Desconhecida (nunca chegamos lá) | ✅ 97%+ accuracy comprovado |
| Manutenção futura | Manual (retreino se quebrar) | ✅ Suporte Picovoice |
| pt-BR real | ✅ Treinado com vozes pt-BR | ✅ Suporte nativo |

---

## 4. Recomendação

### Caminho pragmático (recomendado)
> **Usar Porcupine com o nome "Íris" ou "Nádia"**

1. Cria conta em [console.picovoice.ai](https://console.picovoice.ai)
2. Gera o modelo em pt-BR com o nome escolhido (gratuito)
3. Instala o [PorcupinePipeline add-on](https://github.com/slackr31337/PorcupinePipeline) no HA
4. Funcional em menos de 30 minutos

### Caminho técnico (se quiser continuar)
> **openWakeWord com "Letícia" — mas com expectativas realistas**

- Usar v11 do notebook e resolver o Etapa 1b
- Reservar uma sessão contínua de 3-4 horas (não interromper)
- Se falhar no Etapa 5 novamente → migrar para Porcupine

### Se o objetivo é ter a casa funcionando logo
**→ Porcupine agora, openWakeWord depois como projeto de longo prazo.**

---

## 5. Referências e recursos

| Recurso | Link |
|---------|------|
| openWakeWord GitHub | [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord) |
| Coleção comunitária HA wake words | [fwartner/home-assistant-wakewords-collection](https://github.com/fwartner/home-assistant-wakewords-collection) |
| Console Picovoice (gerar wake word) | [console.picovoice.ai](https://console.picovoice.ai) |
| PorcupinePipeline para HA | [slackr31337/PorcupinePipeline](https://github.com/slackr31337/PorcupinePipeline) |
| Guia HA — criar wake word | [home-assistant.io/voice_control/create_wake_word](https://www.home-assistant.io/voice_control/create_wake_word/) |
| Thread compartilhamento HA | [Wake word model sharing thread](https://community.home-assistant.io/t/wake-word-model-sharing-thread/626849) |
| HA Voice Chapter 11 (multilíngue) | [voice-chapter-11](https://www.home-assistant.io/blog/2025/10/22/voice-chapter-11/) |

---

*Documento gerado em 28/04/2026 — Projeto ha-wakeword-leticia*
