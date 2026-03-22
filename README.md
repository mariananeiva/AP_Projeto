# Detecção de Texto Gerado por Inteligência Artificial

## 📌 Sobre o Projeto
Este repositório contém o código desenvolvido para o Trabalho Prático da Unidade Curricular de **Aprendizagem Profunda**. 

O objetivo principal deste projeto consiste no desenvolvimento de modelos de Machine Learning e Deep Learning capazes de distinguir (problema multi-classe) entre texto gerado por modelos de Inteligência Artificial (Google, Anthropic, Meta, OpenAI) e texto escrito de forma genuína por seres humanos, em Inglês.

### 🎯 Foco Atual: 1ª e 2ª Submissão
O projeto engloba atualmente duas fases fundamentais de entrega:
- **1ª Submissão** (17 de março): Apresentação dos **dois melhores modelos** iniciais (um com **implementação própria em NumPy** e outro desenvolvido nativamente em **PyTorch** linear).
- **2ª Submissão**: Foco no processamento de texto avançado (extração de métricas Handcrafted misturadas com TF-IDF) e treino de modelos sequenciais complexos (ex: **LSTMClassifier**). Foram adicionados pipelines preparados para importação de novos datasets externos.

## 📁 Estrutura do Repositório

O nosso projeto está estruturado da seguinte forma de forma a garantir uma navegação simples e intuitiva pelos diferentes componentes do trabalho desenvolvido:

- `src/`: Contém toda a implementação de código construída de raiz. Trata-se do "motor" principal do projeto e engloba:
  - `neuralnetwork.py`: O núcleo e classe base da rede neuronal.
  - `layers.py`: Definições das camadas disponíveis.
  - `activation.py`: Funções de ativação.
  - `losses.py`: Cálculo e funções de perda/loss.
  - `optimizer.py`: Contém a nossa classe Optimizer principal e variantes como o SGD.

- `models/`: Pasta destinada ao armazenamento interno dos modelos treinados:
  - Ex: `modelo_pytorch.pth`
  - Ex: `pesos_numpy.pkl`

- `notebooks/`: Zona de protótipos e fluxo prático focado aos modelos:
  - `Checkpoint1.ipynb` - Pipeline para treinar, validar e visualizar os resultados dos melhores modelos apurados.

- `data/`: Armazenamento de datasets:
  - `dataset_final.csv` - Conjunto de dados e recursos, preparados para alimentar os modelos.


