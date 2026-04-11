import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, max_len=10, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(max_len-1, batch_size, trg_vocab_size, device=self.device)

        hidden = self.encoder(src)

        input = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=self.device)

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t-1] = output
            top1 = output.argmax(1)

            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input = trg[t]
            else:
                input = top1

        return outputs

import re
import random
import os
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from data_loader import load_examples

SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

def build_vocab(examples, min_freq=1):
    counter = Counter()
    for ex in examples:
        counter.update(tokenize(ex["q"]))
        counter.update(tokenize(ex["a"]))

    itos = SPECIALS.copy()
    for tok, freq in counter.items():
        if freq >= min_freq:
            itos.append(tok)

    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos

def encode(text, stoi):
    toks = tokenize(text)
    ids = [SOS_IDX] + [stoi.get(t, UNK_IDX) for t in toks] + [EOS_IDX]
    return torch.tensor(ids, dtype=torch.long)

def create_batches(examples, stoi, batch_size = 16, shuffle=True):
    indices = list(range(len(examples)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        idxs = indices[start:start + batch_size]
        batch = [examples[i] for i in idxs]

        src_list = [encode(ex["q"], stoi) for ex in batch]
        trg_list = [encode(ex["a"], stoi) for ex in batch]

        src = pad_sequence(src_list, padding_value=PAD_IDX).to(device)
        trg = pad_sequence(trg_list, padding_value=PAD_IDX).to(device)
        yield src, trg

def train_epoch(model, examples, stoi, optimizer, criterion, batch_size=16):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for src, trg in create_batches(examples, stoi, batch_size=batch_size, shuffle=True):
        optimizer.zero_grad()

        outputs = model(src, trg, max_len=trg.shape[0], teacher_forcing_ratio=0.7)
        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            trg[1:].reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def generate_code(model, question, stoi, itos, device, max_len=120):
    model.eval()

    src = encode(question, stoi).unsqueeze(1).to(device)
    hidden = model.encoder(src)

    input_tok = torch.tensor([SOS_IDX], dtype=torch.long, device=device)
    out_tokens = []

    for _ in range(max_len):
        logits, hidden = model.decoder(input_tok, hidden)
        pred_id = int(logits.argmax(1).item())

        if pred_id == EOS_IDX:
            break

        out_tokens.append(pred_id)
        input_tok = torch.tensor([pred_id], dtype=torch.long, device=device)

    words = [
        itos[t] for t in out_tokens
        if t < len(itos) and itos[t] not in {"<pad>", "<sos>", "<eos>", "<unk>"}
    ]
    return " ".join(words)

examples = load_examples("data")
print(f"examples: {len(examples)}")

stoi, itos = build_vocab(examples, min_freq=1)
vocab_size = len(itos)
print(f"vocab size: {vocab_size}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMB_DIM = 128
HID_DIM = 256
BATCH_SIZE = 16

enc = Encoder(vocab_size, EMB_DIM, HID_DIM)
dec = Decoder(vocab_size, EMB_DIM, HID_DIM)
model = Seq2Seq(enc, dec, device).to(device)

# src, trg = make_batch(examples, stoi, batch_size=BATCH_SIZE)
# src, trg = src.to(device), trg.to(device)

# print(f"src shape: {src.shape}")
# print(f"trg shape: {trg.shape}")

# outputs = model(src, trg, max_len=trg.shape[0], teacher_forcing_ratio=0.7)
# print(f"pred shape: {outputs.shape}")

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 3
for epoch in range(1, EPOCHS+1):
    avg_loss = train_epoch(model, examples, stoi, optimizer, criterion, batch_size=BATCH_SIZE)
    print(f"epoch {epoch}/{EPOCHS} - train loss: {avg_loss:.4f}")

os.makedirs("models", exist_ok=True)
ckpt_path = "models/seq2seq_prototype.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "emb_dim": EMB_DIM,
        "hid_dim": HID_DIM,
        "pad_idx": PAD_IDX,
        "sos_idx": SOS_IDX,
        "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX,
    },
    ckpt_path,
)
print(f"saved checkpoint: {ckpt_path}")

test_q = "how do i read a csv file in pandas"
pred = generate_code(model, test_q, stoi, itos, device)
print("\nTEST QUESTION:", test_q)
print("GENERATED CODE:\n", pred)