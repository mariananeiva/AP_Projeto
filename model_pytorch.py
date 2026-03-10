import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TextClassifierDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=5):
        super(TextClassifierDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU() # Ativação ReLU
        # Dropout para evitar overfitting
        self.dropout = nn.Dropout(0.2) 
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) 
        return out

# Função de Treino Multi-classe
def train_multi_class_model(X_train, y_train, input_dim):
    # Converter dados NumPy para Tensors
    # X_train deve ser a  matriz TF-IDF 
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    # y_train deve conter labels de 0 a 4 
    y_tensor = torch.tensor(y_train, dtype=torch.long).squeeze()

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Inicializar modelo (64 neurónios na camada oculta)
    model = TextClassifierDNN(input_size=input_dim, hidden_size=64, num_classes=5)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    print("A iniciar o treino...")
    for epoch in range(50):
        running_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step() # Atualização dos pesos
            running_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epochs: [{epoch+1}/50], Perda: {running_loss/len(loader):.4f}")
            
    return model

# 3. Função para guardar o modelo
def save_model(model, filepath='modelo_final_pytorch.pth'):
    torch.save(model.state_dict(), filepath)
    print(f"Modelo guardado em {filepath}")

# 4. Função para Previsão
def predict_pytorch(model, X_test):
    model.eval() # Desativa o Dropout para a fase de teste
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_tensor)
        # Pega na classe com maior probabilidade (Softmax)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.numpy()