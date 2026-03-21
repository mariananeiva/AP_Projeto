import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# MODELO DNN (Baseline da Fase 1)
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

# LSTM BIDIRECIONAL (Fase 2 )
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        out = self.dropout(hidden_cat)
        return self.fc(out)


def train_pytorch_model(X_train, y_train, input_dim, n_classes, epochs=50):
    """Treino original para modelos DNN (Datasets Tabulares/TF-IDF)."""
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    if n_classes == 1:
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        criterion = nn.BCEWithLogitsLoss() 
    else:
        y_tensor = torch.tensor(y_train, dtype=torch.long).squeeze()
        criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TextClassifierDNN(input_size=input_dim, hidden_size=64, num_classes=n_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    print(f"A iniciar o treino DNN ({n_classes} classes)...")
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

def train_rnn_model(X_train, y_train, vocab_size, n_classes=5, epochs=30):
    """Novo treino para modelo sequencial RNN/LSTM (5 Classes)."""
    X_tensor = torch.tensor(X_train, dtype=torch.long)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = RNNClassifier(vocab_size=vocab_size, embedding_dim=100, hidden_size=64, num_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    model.train()
    print(f"A iniciar o treino RNN/LSTM ({n_classes} classes)...")
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(loader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Perda {avg_loss:.4f}")
            
    return model

def save_pytorch_model(model, filepath):
    """Guarda os pesos do modelo PyTorch."""
    torch.save(model.state_dict(), filepath)
    print(f"Modelo PyTorch guardado em {filepath}")

def load_pytorch_model(model_type, params, filepath):
    """Carrega o modelo especificado (DNN ou RNN)."""
    if model_type == 'dnn':
        model = TextClassifierDNN(params['input_size'], params['hidden_size'], params['n_classes'])
    else:
        model = RNNClassifier(params['vocab_size'], params['embedding_dim'], params['hidden_size'], params['n_classes'])
        
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    model.eval()
    print(f"Modelo {model_type.upper()} carregado de {filepath}")
    return model

def predict_pytorch(model, X_test, n_classes, is_rnn=False):
    """Previsões para modelos tabulares ou sequenciais."""
    model.eval() 
    with torch.no_grad():
        dtype = torch.long if is_rnn else torch.float32
        X_tensor = torch.tensor(X_test, dtype=dtype)
        outputs = model(X_tensor)
        
        if n_classes == 1:
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).int()
        else:
            _, predicted = torch.max(outputs.data, 1)
            
    return predicted.numpy()