# Minimal Seq2Seq Transformer for toy translation with PyTorch
# - Builds a small vocab from toy parallel data
# - Trains with teacher forcing (cross-entropy)
# - Greedy decodes to translate new inputs

import math
import random
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) Toy parallel corpus
# -----------------------------
pairs = [
    ("i am a student",           "ich bin ein student"),
    ("you are a student",        "du bist ein student"),
    ("he is a teacher",          "er ist ein lehrer"),
    ("she is a teacher",         "sie ist eine lehrerin"),
    ("we are friends",           "wir sind freunde"),
    ("they are friends",         "sie sind freunde"),
    ("this is a book",           "das ist ein buch"),
    ("that is a cat",            "das ist eine katze"),
    ("i love apples",            "ich liebe äpfel"),
    ("i love you",               "ich liebe dich"),
    ("do you speak english",     "sprichst du englisch"),
    ("good morning",             "guten morgen"),
    ("good night",               "gute nacht"),
    ("thank you",                "danke"),
    ("how are you",              "wie geht es dir"),
    ("i am fine",                "mir geht es gut"),
]


# -----------------------------
# 2) Tokenization & vocab
# -----------------------------
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

def tokenize(s: str) -> List[str]:
    # ultra-simple whitespace tokenizer
    return s.lower().strip().split()

def build_vocab(sentences: List[List[str]], min_freq: int = 1):
    from collections import Counter
    c = Counter()
    for toks in sentences:
        c.update(toks)
    itos = [PAD, BOS, EOS, UNK]
    for tok, freq in c.items():
        if freq >= min_freq and tok not in itos:
            itos.append(tok)
    stoi = {t:i for i,t in enumerate(itos)}
    return stoi, itos

src_sents = [tokenize(src) for src, _ in pairs]
tgt_sents = [tokenize(tgt) for _, tgt in pairs]

src_stoi, src_itos = build_vocab(src_sents)
tgt_stoi, tgt_itos = build_vocab(tgt_sents)
SRC_V, TGT_V = len(src_itos), len(tgt_itos)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = src_stoi[PAD], tgt_stoi[BOS], tgt_stoi[EOS], tgt_stoi[UNK]

def encode(tokens: List[str], stoi: dict, add_bos_eos=False) -> List[int]:
    ids = [stoi.get(t, stoi[UNK]) for t in tokens]
    if add_bos_eos:
        return [tgt_stoi[BOS]] + [tgt_stoi.get(t, tgt_stoi[UNK]) for t in tokens] + [tgt_stoi[EOS]]
    return [stoi.get(t, stoi[UNK]) for t in tokens]

def encode_src(tokens: List[str]) -> List[int]:
    return [src_stoi.get(t, src_stoi[UNK]) for t in tokens]

def encode_tgt(tokens: List[str]) -> Tuple[List[int], List[int]]:
    # returns (input sequence with BOS, output sequence with EOS)
    inp = [tgt_stoi[BOS]] + [tgt_stoi.get(t, tgt_stoi[UNK]) for t in tokens]
    out = [tgt_stoi.get(t, tgt_stoi[UNK]) for t in tokens] + [tgt_stoi[EOS]]
    return inp, out

def decode(ids: List[int], itos: List[str]) -> str:
    toks = []
    for i in ids:
        if i == tgt_stoi[EOS]:
            break
        if i not in (tgt_stoi[BOS], tgt_stoi[PAD]):
            toks.append(itos[i])
    return " ".join(toks)

# -----------------------------
# 3) Dataset & collator
# -----------------------------
class ParallelToy(Dataset):
    def __init__(self, pairs):
        # small random split
        random.seed(0)
        shuffled = pairs[:]
        random.shuffle(shuffled)
        self.data = shuffled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = encode_src(tokenize(src))
        tinp, tout = encode_tgt(tokenize(tgt))
        return torch.tensor(src_ids), torch.tensor(tinp), torch.tensor(tout)

def pad_seq(seqs: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    maxlen = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :s.size(0)] = s
    return out

def collate(batch):
    srcs, tinps, touts = zip(*batch)
    src_batch  = pad_seq(list(srcs),  src_stoi[PAD])
    tinp_batch = pad_seq(list(tinps), tgt_stoi[PAD])
    tout_batch = pad_seq(list(touts), tgt_stoi[PAD])
    # padding masks (True where PAD so Transformer can ignore)
    src_key_padding_mask = (src_batch == src_stoi[PAD])  # (B, S)
    tgt_key_padding_mask = (tinp_batch == tgt_stoi[PAD]) # (B, T)
    return src_batch, tinp_batch, tout_batch, src_key_padding_mask, tgt_key_padding_mask

dataset = ParallelToy(pairs)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

# -----------------------------
# 4) Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# -----------------------------
# 5) Transformer Seq2Seq model
# -----------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=src_stoi[PAD])
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=tgt_stoi[PAD])
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, L, D)
        )
        self.generator = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt_inp, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src, tgt_inp: (B, S/T)
        src_emb = self.pos(self.src_embed(src))   # (B, S, D)
        tgt_emb = self.pos(self.tgt_embed(tgt_inp))  # (B, T, D)

        # Subsequent mask for autoregressive decoding
        T = tgt_inp.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(src.device)  # (T, T)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,  # (B, S)
            tgt_key_padding_mask=tgt_key_padding_mask,  # (B, T)
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        logits = self.generator(out)  # (B, T, Vtgt)
        return logits

# -----------------------------
# 6) Training setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqTransformer(SRC_V, TGT_V).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_stoi[PAD])
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# -----------------------------
# 7) Train loop
# -----------------------------
EPOCHS = 40  # small dataset -> short epochs
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for src, tinp, tout, src_padmask, tgt_padmask in loader:
        src, tinp, tout = src.to(device), tinp.to(device), tout.to(device)
        src_padmask, tgt_padmask = src_padmask.to(device), tgt_padmask.to(device)

        optimizer.zero_grad()
        logits = model(src, tinp, src_padmask, tgt_padmask)  # (B, T, V)
        # shift already handled by tinp/tout construction
        loss = criterion(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | loss={total_loss/len(loader):.4f}")

# -----------------------------
# 8) Greedy decoding
# -----------------------------
@torch.no_grad()
def translate(model, src_sentence: str, max_len: int = 30) -> str:
    model.eval()
    src = torch.tensor([encode_src(tokenize(src_sentence))], dtype=torch.long, device=device)
    src_padmask = (src == src_stoi[PAD])

    # encode once
    memory = model.transformer.encoder(
        model.pos(model.src_embed(src)), src_key_padding_mask=src_padmask
    )

    ys = torch.tensor([[tgt_stoi[BOS]]], dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_padmask = (ys == tgt_stoi[PAD])
        T = ys.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        out = model.transformer.decoder(
            model.pos(model.tgt_embed(ys)),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padmask,
            memory_key_padding_mask=src_padmask
        )
        logits = model.generator(out[:, -1, :])  # last step
        next_token = logits.argmax(-1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == tgt_stoi[EOS]:
            break

    return decode(ys[0].tolist(), tgt_itos)

# -----------------------------
# 9) Quick tests
# -----------------------------
tests = [
    "i am a student",
    "you are a student",
    "this is a book",
    "i love you",
    "how are you",
]

for s in tests:
    print(f"SRC: {s}")
    print(f"TGT: {translate(model, s)}")
    print("-" * 40)
