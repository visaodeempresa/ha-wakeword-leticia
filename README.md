# Wake Word Letícia - Home Assistant

Projeto para treinamento e implantação de uma wake word personalizada ("Letícia") para o Home Assistant Yellow.

## Estrutura do Projeto

- `docs/`: Documentação, roteiros, checklists e screenshots.
- `notebooks/`: Notebooks Jupyter/Colab para treinamento.
- `scripts/`: Scripts Python e Bash para automação (treinamento local/remoto e deploy).
- `models/`: Pasta para armazenar os modelos `.tflite` e `.onnx` finais.
- `.github/workflows/`: Automação de treinamento via GitHub Actions.

## Como Usar

### Treinamento
Para treinar o modelo, você pode usar o script principal:
```bash
python scripts/train_leticia.py --all
```

### Teste Local
Para testar o modelo com seu microfone:
```bash
python scripts/03_test_model.py --model models/leticia.tflite
```

### Deploy
Para enviar o modelo para o seu Home Assistant Yellow:
```bash
./scripts/02_deploy_to_ha.sh models/leticia.tflite [IP_DO_HA]
```

Para mais detalhes, veja o [Roteiro Completo](docs/ROTEIRO-COMPLETO.md).
