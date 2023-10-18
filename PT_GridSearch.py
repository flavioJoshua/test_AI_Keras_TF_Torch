import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Verifica se la GPU è disponibile
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device("cuda:0")  # Sceglie la prima GPU
else:
    device = torch.device("cpu")  # Usa la CPU se la GPU non è disponibile

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

from torch.utils.data import random_split

# Supponiamo che il 10% dei dati venga utilizzato per la validazione
n_train = int(len(train_data) * 0.9)
n_val = len(train_data) - n_train

train_subset, val_subset = random_split(train_data, [n_train, n_val])

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)



# train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Modifica la funzione train_model per includere la validazione
def train_model(model, optimizer, train_loader, val_loader):
    model.to(device)  # Sposta il modello sulla GPU

    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        # Training
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcola le metriche di validazione
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.4f}, Validation F1 Score: {f1:.4f}")





# Using SGD
print("Training model with SGD optimizer")
model_sgd = SimpleNN()
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
# train_model(model_sgd, optimizer_sgd)

# Using Adam
print("Training model with Adam optimizer")
model_adam = SimpleNN()
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
# train_model(model_adam, optimizer_adam)

# Assume che val_loader sia un DataLoader per il tuo set di validazione
train_model(model_sgd, optimizer_sgd, train_loader, val_loader)