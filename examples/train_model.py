import torch 
from torch.utils.data import DataLoader
from belief_state_superposition.model import init_model
from belief_state_superposition.data import get_dataset
from belief_state_superposition.train import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataset = get_dataset(1000)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = init_model().to(device)
train_model(model, train_data_loader, n_epochs=10, show_progress_bar=True, device = device)