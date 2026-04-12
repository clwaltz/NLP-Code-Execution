import re
import random
import os

from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from data_loader import load_examples
from sklearn.model_selection import KFold
import numpy as np
import json
from pathlib import Path

import torch
import torch.nn as nn

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

def create_batches(examples, stoi, device, batch_size = 16, shuffle=True):
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

def train_epoch(model, examples, stoi, optimizer, criterion, device, batch_size=16):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for src, trg in create_batches(examples, stoi, device, batch_size=batch_size, shuffle=True):
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
    return "".join(words)

def build_model(vocab_size, emb_dim, hid_dim, device):
    enc = Encoder(vocab_size, emb_dim, hid_dim)
    dec = Decoder(vocab_size, emb_dim, hid_dim)
    model = Seq2Seq(enc, dec, device).to(device)
    return model

@torch.no_grad()
def eval_epoch(model, examples, stoi, criterion, device, batch_size=16):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for src, trg in create_batches(examples, stoi, device, batch_size=batch_size, shuffle=False):
        outputs = model(src, trg, max_len=trg.shape[0], teacher_forcing_ratio=0.0)
        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            trg[1:].reshape(-1),
        )
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

def run_cv(examples, n_splits, config, device, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(examples))
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(indices), start=1):
        train_examples = [examples[i] for i in tr_idx]
        val_examples = [examples[i] for i in va_idx]

        stoi, itos = build_vocab(train_examples, min_freq=config["min_freq"])
        vocab_size = len(itos)

        model = build_model(vocab_size, config["emb_dim"], config["hid_dim"], device)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        for epoch in range(1, config["epochs"] + 1):
            train_loss = train_epoch(model, train_examples, stoi, optimizer, criterion, device, batch_size=config["batch_size"])
            val_loss = eval_epoch(model, val_examples, stoi, criterion, device, batch_size=config["batch_size"])
            print(f"[Fold {fold}] epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}", flush=True)

        fold_results.append({"fold": fold, "n_val": len(val_examples), "val_loss": float(val_loss)})

    summary = {
        "n_splits": n_splits,
        "n_examples": len(examples),
        "folds": fold_results,
        "mean_val_loss": float(np.mean([f["val_loss"] for f in fold_results])),
        "std_val_loss": float(np.std([f["val_loss"] for f in fold_results]))
    }
    return summary

def main():
    examples = load_examples("data")
    print(f"examples: {len(examples)}", flush=True)

    config = {
        "emb_dim": 128,
        "hid_dim": 256,
        "batch_size": 16,
        "epochs": 10,
        "lr": 1e-3,
        "min_freq": 1
    }

    stoi, itos = build_vocab(examples, min_freq=config["min_freq"])
    vocab_size = len(itos)
    print(f"vocab size: {vocab_size}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # summary_5 = run_cv(examples, n_splits=5, config=config, device=device, seed=42)
    # (reports_dir / "seq2seq_cv_5fold.json").write_text(
    #     json.dumps(summary_5, indent=2),
    #     encoding="utf-8",
    # )
    # print("saved report: reports/seq2seq_cv_5fold.json", flush=True)

    summary_10 = run_cv(examples, n_splits=10, config=config, device=device, seed=42)
    (reports_dir / "seq2seq_cv_10fold.json").write_text(
        json.dumps(summary_10, indent=2),
        encoding="utf-8",
    )
    print("saved report: reports/seq2seq_cv_10fold.json", flush=True)

    model = build_model(vocab_size, config["emb_dim"], config["hid_dim"], device)
    print("model created", flush=True)

# src, trg = make_batch(examples, stoi, batch_size=BATCH_SIZE)
# src, trg = src.to(device), trg.to(device)

# print(f"src shape: {src.shape}")
# print(f"trg shape: {trg.shape}")

# outputs = model(src, trg, max_len=trg.shape[0], teacher_forcing_ratio=0.7)
# print(f"pred shape: {outputs.shape}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(1, config["epochs"]+1):
        avg_loss = train_epoch(model, examples, stoi, optimizer, criterion, device, batch_size=config["batch_size"])
        print(f"epoch {epoch}/{config['epochs']} - train loss: {avg_loss:.4f}", flush=True)

    os.makedirs("models", exist_ok=True)
    ckpt_path = "models/seq2seq_prototype.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "emb_dim": config["emb_dim"],
            "hid_dim": config["hid_dim"],
            "pad_idx": PAD_IDX,
            "sos_idx": SOS_IDX,
            "eos_idx": EOS_IDX,
            "unk_idx": UNK_IDX,
        },
        ckpt_path,
    )
    print(f"saved checkpoint: {ckpt_path}", flush=True)

    test_q = "sort the data by population column"
    pred = generate_code(model, test_q, stoi, itos, device)
    print("\nTEST QUESTION:", test_q)
    print("GENERATED CODE:\n", pred)

if __name__ == "__main__":
    main()
    