
# %%
import torch 
import numpy as np
import itertools
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
# Sweep over different model sizes

d_models = [2, 4, 7, 8, 16, 32, 64]
results = collections.defaultdict(dict)
for d_model in d_models:
    model = init_model(d_model = d_model).to(device)
    loss_hist = train_model(model, train_data_loader, n_epochs=10, show_progress_bar=True, device = device)
    results[d_model]['loss'] = loss_hist
    del model
    # results[model_size]['model'] = model

for model_size, result in results.items():
    print(f"Model size: {model_size}, final loss: {result['loss'][-1]}")

sns.set_theme()
fig, ax = plt.subplots()
num_tokens = np.arange(320) * batch_size * seq_len
for model_size, result in results.items():
    sns.lineplot(x = num_tokens, y = result['loss'], label=f"D_model: {model_size}", ax = ax)
ax.set_xlabel('Tokens')
ax.set_ylabel('Loss')
ax.legend()
plt.show()
# %%
# Compare layer 7 and layer 8 over multiple seeds

seeds = [0, 1, 2, 3, 4]
d_models = [7, 8]
results = collections.defaultdict(dict)
for seed, d_model in itertools.product(seeds, d_models):
    model = init_model(
        d_model = d_model,
        seed = seed
    ).to(device)
    loss_hist = train_model(model, train_data_loader, n_epochs=10, show_progress_bar=True, device = device)
    results[f'd_model={d_model}, seed={seed}']['loss'] = loss_hist
    del model
    # results[model_size]['model'] = model

for exp_name, result in results.items():
    print(f"{exp_name}, final loss: {result['loss'][-1]}")

sns.set_theme()
fig, ax = plt.subplots()
num_tokens = np.arange(320) * batch_size * seq_len
for exp_name, result in results.items():
    sns.lineplot(x = num_tokens, y = result['loss'], label=exp_name, ax = ax)
ax.set_xlabel('Tokens')
ax.set_ylabel('Loss')
ax.legend()
plt.show()

# %%
