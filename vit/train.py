import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm, trange
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from vit.model import Vit
np.random.seed(0)
torch.manual_seed(0)

def main (): 
    transform = ToTensor()
    train_set = MNIST(root='./../datasets', train=True, download = True, transform = transform)
    test_set = MNIST(root='./../datasets', train=False, download = True, transform = transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Vit((1, 28, 28), n_patches=7, n_heads=2, n_blocks=2, hidden_d=2, out_d=10)
    N_EPOCH = 5
    LR = 0.005

    # Traning loop 
    optimizer = Adam(model.parameters(), lr=LR)
    criterion =CrossEntropyLoss()
    for epoch in trange(N_EPOCH, desc='Training'):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} in training'):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().items() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch +1}/ {N_EPOCH} loss: {train_loss:.2f}')
    
    # Test
    with torch.no_grad:
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc='Testing'):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().items() / len(test_loader)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()

            total += len(x)

        print(f'test loss: {test_loss:.2f}')







    

    