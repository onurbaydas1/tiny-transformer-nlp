import torch
from model import TinyTransformer
from tokenizer import BPETokenizer

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

# Tokenizer
tok = BPETokenizer(vocab_size=100)
tok.train(texts)
vocab_size = len(tok.get_vocab())

# Model
model = TinyTransformer(vocab_size)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prompt
prompt = "merhaba"
ids = tok.encode(prompt)
if not ids:
    print("Uyarı: Prompt boş token listesi döndürdü.")
    exit()

context = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

# Üretim
for _ in range(10):
    with torch.no_grad():
        out = model(context)  # (1, seq_len, vocab)

    logits = out[:, -1, :] / 1.0  # (1, vocab), temperature
    top_k = 5
    values, indices = torch.topk(logits, top_k)  # (1, top_k)
    probs = torch.softmax(values, dim=-1)        # (1, top_k)
    sampled = torch.multinomial(probs, num_samples=1)  # (1, 1)
    next_token = indices.gather(-1, sampled)  # (1, 1)

    # next_token: shape (1, 1) → cat ile uyumlu
    context = torch.cat([context, next_token], dim=1)

# Çıktı
print(tok.decode(context[0].tolist()))
