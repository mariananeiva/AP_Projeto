import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TextClassifierDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextClassifierDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.2) 
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) 
        return out

def train_pytorch_model(X_train, y_train, input_dim, n_classes, epochs=50):
    """
    Função flexível para treinar o modelo binário (n_classes=1) 
    ou o modelo multi-classe (n_classes=4).
    """
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    if n_classes == 1:
        # Para binário, usamos float e BCEWithLogitsLoss
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        criterion = nn.BCEWithLogitsLoss() 
    else:
        # Para multi-classe, usamos long e CrossEntropyLoss
        y_tensor = torch.tensor(y_train, dtype=torch.long).squeeze()
        criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TextClassifierDNN(input_size=input_dim, hidden_size=64, num_classes=n_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    print(f"A iniciar o treino ({n_classes} classes)...")
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epochs: [{epoch+1}/{epochs}], Perda: {running_loss/len(loader):.4f}")
            
    return model

def save_pytorch_model(model, filepath):
    """Guarda os pesos do modelo PyTorch."""
    torch.save(model.state_dict(), filepath)
    print(f"Modelo PyTorch guardado em {filepath}")

def load_pytorch_model(input_size, hidden_size, n_classes, filepath):
    """Carrega os pesos do modelo PyTorch."""
    model = TextClassifierDNN(input_size, hidden_size, n_classes)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Modelo PyTorch carregado de {filepath}")
    return model

def predict_pytorch(model, X_test, n_classes):
    """Realiza previsões considerando se o modelo é binário ou multi-classe."""
    model.eval() 
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_tensor)
        
        if n_classes == 1:
            # Para binário, aplicamos sigmoid e threshold de 0.5
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).int()
        else:
            # Para multi-classe, pegamos na classe com maior score
            _, predicted = torch.max(outputs.data, 1)
            
    return predicted.numpy()