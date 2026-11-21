
import sys

import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("..")
from functions import load

FILEPATH = "../data/All_spikes.pt"
MODEL_FILE = "../models/snn_example.pt"


kwargs = dict(solution=1, condition=1, start_time=2.0, end_time=4.0)
train = load.spikes(FILEPATH, "train",solution=1, condition=1, start_time=2.0, end_time=4.0)
val = load.spikes(FILEPATH, "val", solution=1, condition=1, start_time=2.0, end_time=4.0)
test = load.spikes(FILEPATH, "test", solution=1, condition=1, start_time=2.0, end_time=4.0)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(val, batch_size=64)
test_loader = DataLoader(test, batch_size=64)


class SNN(nn.Module):
    def __init__(self, n_in=256, n_hidden=128, n_out=4, beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(self, x):                  
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_mem = []
        for t in range(x.shape[1]):      
            spk1, mem1 = self.lif1(self.fc1(x[:, t]), mem1)
            _, mem2 = self.lif2(self.fc2(spk1), mem2)
            out_mem.append(mem2)
        return torch.stack(out_mem).mean(0)


model = SNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total



best_val, best_state = 0, None
for epoch in range(10):
    model.train()
    for x, y in train_loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val_acc = evaluate(val_loader)
    print(f"epoch {epoch + 1:2d}  loss {loss.item():.3f}  val acc {val_acc:.1%}")
    if val_acc > best_val:
        best_val, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}


model.load_state_dict(best_state)
torch.save({"state_dict": best_state, "data_kwargs": dict(solution=1, condition=1, start_time=2.0, end_time=4.0), "val_acc": best_val}, MODEL_FILE)
print(f"\nSaved best model (val acc {best_val:.1%}) to {MODEL_FILE}")
print(f"Test accuracy: {evaluate(test_loader):.1%}  (chance = 25%)")
