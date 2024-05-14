# %%

import torch 
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from torch.utils.data import DataLoader
from belief_state_superposition.model import init_model
from belief_state_superposition.data import get_dataset
from belief_state_superposition.train import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"
n_samples = 1024
batch_size = 32
seq_len = 16
train_dataset = get_dataset(n_samples, seq_len=seq_len)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# %%


model_sizes = [16, 64, 256]
results = collections.defaultdict(dict)
for model_size in model_sizes:
    model = init_model(model_size).to(device)
    loss_hist = train_model(model, train_data_loader, n_epochs=10, device = device)
    results[model_size]['loss'] = loss_hist
    results[model_size]['model'] = model

# %%
# Plot results
sns.set_theme()
fig, ax = plt.subplots()
for model_size, result in results.items():
    ax.plot(result['loss'], label=f"Model size: {model_size}")
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
plt.show()