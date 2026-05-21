"""
Шаг 5 — генерация бот-корпуса словесной LSTM языковой моделью.

По методичке Spot the bot бот-тексты создаёт нейросетевой генератор,
обученный на человеческом корпусе. Берём словесную (token-level) LSTM —
она оперирует теми же лемма-токенами, что и SVD-словарь, поэтому
сгенерированные тексты сразу годятся для build_dataset.py (тот же словарь).

Пайплайн:
  1. Читаем человеческий корпус corpus_clean_L2/all_clean.txt (строка = текст).
  2. Словарь: спец-токены + слова с частотой >= min_count (кап vocab_size).
  3. Поток обучения: для каждого текста <bos> ... <eos>, всё сконкатенировано
     в один id-массив; обучение next-token с truncated BPTT.
  4. LSTM-LM: Embedding -> LSTM -> Linear.
  5. Генерация n_texts текстов; длина каждого берётся из эмпирического
     распределения длин человеческих текстов — бот-корпус сопоставим по объёму
     (README шаг 5: «сгенерировать столько же текстов, сколько у человека»).
  6. Сохраняем чекпойнт модели и corpus_bot/all_clean.txt.

Запуск:
  python scripts/train_lstm.py --corpus corpus_clean_L2 --epochs 8 --device cuda:0

Выход:
  models/lstm/mt_lstm.pt      — чекпойнт (веса + словарь + конфиг)
  corpus_bot/all_clean.txt    — бот-корпус (строка = текст)
  corpus_bot/_meta.json       — мета-инфа
"""
import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


def load_human_docs(corpus_dir: Path):
    """Список текстов; каждый текст — список токенов."""
    single = corpus_dir / "all_clean.txt"
    files = [single] if single.exists() else sorted(corpus_dir.glob("all_clean.shard*.txt"))
    if not files:
        raise FileNotFoundError(f"Нет all_clean.txt(.shard*) в {corpus_dir}")
    docs = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                toks = line.split()
                if toks:
                    docs.append(toks)
    return docs


def build_vocab(docs, min_count: int, vocab_size: int | None):
    cnt = Counter()
    for d in docs:
        cnt.update(d)
    words = [w for w, c in cnt.most_common() if c >= min_count]
    if vocab_size:
        words = words[: max(0, vocab_size - len(SPECIALS))]
    itos = SPECIALS + words
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def encode_stream(docs, stoi):
    """Все тексты -> один id-поток: <bos> текст <eos> <bos> текст <eos> ..."""
    bos, eos, unk = stoi[BOS], stoi[EOS], stoi[UNK]
    stream = []
    for d in docs:
        stream.append(bos)
        stream.extend(stoi.get(w, unk) for w in d)
        stream.append(eos)
    return np.asarray(stream, dtype=np.int64)


class LSTMLM(nn.Module):
    def __init__(self, vocab, emb, hidden, layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, layers,
                            dropout=dropout if layers > 1 else 0.0,
                            batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, state=None):
        e = self.drop(self.embed(x))
        out, state = self.lstm(e, state)
        logits = self.fc(self.drop(out))
        return logits, state


def batchify(ids: np.ndarray, batch_size: int, device):
    """(N,) -> (batch_size, N // batch_size) — параллельные потоки для BPTT."""
    n = len(ids) // batch_size
    data = torch.from_numpy(ids[: n * batch_size]).long()
    return data.view(batch_size, n).contiguous().to(device)


def detach_state(state):
    return tuple(s.detach() for s in state)


def run_epoch(model, data, seq_len, optimizer, criterion, clip, scaler, train: bool):
    model.train(train)
    total_loss, total_tok = 0.0, 0
    state = None
    n = data.size(1)
    for i in range(0, n - 1, seq_len):
        x = data[:, i:i + seq_len]
        y = data[:, i + 1:i + 1 + seq_len]
        if x.size(1) != y.size(1):
            x = x[:, :y.size(1)]
        if x.size(1) == 0:
            break
        if state is not None:
            state = detach_state(state)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits, state = model(x, state)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
        ntok = y.numel()
        total_loss += loss.item() * ntok
        total_tok += ntok
    return total_loss / max(1, total_tok)


@torch.no_grad()
def generate(model, stoi, itos, n_texts, lengths, device, temperature, batch):
    """Сгенерировать n_texts текстов; длина каждого — сэмпл из распределения lengths."""
    model.eval()
    bos = stoi[BOS]
    block = torch.tensor([stoi[t] for t in SPECIALS], device=device)  # спец-токены не генерим
    targets = [int(lengths[random.randrange(len(lengths))]) for _ in range(n_texts)]
    results = []
    done = 0
    while done < n_texts:
        bs = min(batch, n_texts - done)
        tgt = targets[done:done + bs]
        maxlen = max(tgt)
        cur = torch.full((bs, 1), bos, dtype=torch.long, device=device)
        state = None
        seqs = [[] for _ in range(bs)]
        for step in range(maxlen):
            logits, state = model(cur, state)
            logits = logits[:, -1, :].float() / temperature
            logits[:, block] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            for b in range(bs):
                if step < tgt[b]:
                    seqs[b].append(int(nxt[b, 0]))
            cur = nxt
        for b in range(bs):
            results.append([itos[t] for t in seqs[b]])
        done += bs
        print(f"  generated {done}/{n_texts}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus_clean_L2", help="каталог с all_clean*")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=70)
    ap.add_argument("--emb", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--vocab-size", type=int, default=50000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-texts", type=int, default=0, help="0 = столько же, сколько у человека")
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"=== LSTM bot-corpus | device={device} ===", flush=True)

    corpus_dir = ROOT / args.corpus
    docs = load_human_docs(corpus_dir)
    doc_lengths = np.array([len(d) for d in docs])
    print(f"Human docs: {len(docs)} | tokens: {doc_lengths.sum()} | "
          f"len mean/median: {doc_lengths.mean():.0f}/{np.median(doc_lengths):.0f}", flush=True)

    itos, stoi = build_vocab(docs, args.min_count, args.vocab_size)
    print(f"Vocab: {len(itos)} (specials + words, min_count={args.min_count})", flush=True)

    stream = encode_stream(docs, stoi)
    n_val = int(len(stream) * 0.05)
    train_ids, val_ids = stream[:-n_val], stream[-n_val:]
    train_data = batchify(train_ids, args.batch_size, device)
    val_data = batchify(val_ids, args.batch_size, device)
    print(f"Stream: {len(stream)} ids | train {train_data.shape} | val {val_data.shape}", flush=True)

    model = LSTMLM(len(itos), args.emb, args.hidden, args.layers, args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_par/1e6:.1f}M params", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if device.startswith("cuda") else None

    print("\n[train]", flush=True)
    best_val = math.inf
    mdl_dir = ROOT / "models" / "lstm"
    mdl_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = mdl_dir / "mt_lstm.pt"
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_data, args.seq_len, optimizer, criterion, args.clip, scaler, True)
        vl = run_epoch(model, val_data, args.seq_len, optimizer, criterion, args.clip, None, False)
        print(f"  epoch {ep:2d}/{args.epochs} | train loss {tr:.3f} ppl {math.exp(tr):.1f} "
              f"| val loss {vl:.3f} ppl {math.exp(vl):.1f} | {time.time()-t0:.0f}s", flush=True)
        if vl < best_val:
            best_val = vl
            torch.save({"state_dict": model.state_dict(), "itos": itos,
                        "config": vars(args)}, ckpt_path)
    print(f"Best val ppl: {math.exp(best_val):.1f} | checkpoint: {ckpt_path}", flush=True)

    # лучшая модель -> генерация
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    n_texts = args.n_texts or len(docs)
    print(f"\n[generate] {n_texts} bot texts (temperature={args.temperature})", flush=True)
    t0 = time.time()
    bot_docs = generate(model, stoi, itos, n_texts, doc_lengths, device,
                        args.temperature, args.gen_batch)
    print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)

    out_dir = ROOT / "corpus_bot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "all_clean.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for d in bot_docs:
            f.write(" ".join(d) + "\n")
    bot_lengths = np.array([len(d) for d in bot_docs])
    meta = {
        "n_texts": int(n_texts),
        "tokens_total": int(bot_lengths.sum()),
        "len_mean": float(bot_lengths.mean()),
        "len_median": float(np.median(bot_lengths)),
        "human_len_mean": float(doc_lengths.mean()),
        "temperature": args.temperature,
        "vocab": len(itos),
        "best_val_ppl": math.exp(best_val),
        "model": "word-LSTM",
        "config": vars(args),
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\nSaved bot corpus: {out_file} ({n_texts} texts)", flush=True)
    print(f"Bot len mean/median: {bot_lengths.mean():.0f}/{np.median(bot_lengths):.0f} "
          f"(human {doc_lengths.mean():.0f}/{np.median(doc_lengths):.0f})", flush=True)


if __name__ == "__main__":
    main()
