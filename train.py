import torch
from torch import nn, optim
from tokenizer import BPETokenizer
from model import TinyTransformer

# Eğitim verisi
texts = [
    "merhaba dünya",
    "nasılsın bugün",
    "yapay zeka güzel",
    "bugün hava çok güzel",
    "benim adım onur",
    "ne zaman yola çıkıyoruz",
    "dil modeli eğitiyorum",
    "transformer mimarisiyle çalışıyorum",
    "bilgisayar çok yavaşladı",
    "yeni bir şeyler öğrenmek güzel"
]

# Tokenizer'ı eğit
tok = BPETokenizer(vocab_size=100)
tok.train(texts)
vocab_size = len(tok.get_vocab())

# Modeli oluştur
model = TinyTransformer(vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizasyon
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Verileri encode et
data = [torch.tensor(tok.encode(t), dtype=torch.long) for t in texts]

# Eğitim
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for seq in data:
        if len(seq) < 2:
            continue
        inp = seq[:-1].unsqueeze(0).to(device)
        target = seq[1:].unsqueeze(0).to(device)
        out = model(inp)
        loss = loss_fn(out.reshape(-1, vocab_size), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# Modeli kaydet
torch.save(model.state_dict(), "model.pt")
print("Model saved as model.pt")
