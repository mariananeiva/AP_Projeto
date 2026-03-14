# Detecção de Texto Gerado por Inteligência Artificial

## 📌 Sobre o Projeto
Este repositório contém o código desenvolvido para o Trabalho Prático da Unidade Curricular de **Aprendizagem Profunda**. 

O objetivo principal deste projeto consiste no desenvolvimento de modelos de Machine Learning e Deep Learning capazes de distinguir (problema multi-classe) entre texto gerado por modelos de Inteligência Artificial (Google, Anthropic, Meta, OpenAI) e texto escrito de forma genuína por seres humanos, em Inglês.

### 🎯 Foco Atual: 1ª Submissão
No momento atual, o código e os modelos desenvolvidos encontram-se direcionados para a **1ª Submissão** do projeto (17 de março), que engloba:
- A apresentação dos **dois melhores modelos** desenvolvidos por nós (um com **implementação própria em NumPy** e outro desenvolvido com recurso a **PyTorch**).

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


