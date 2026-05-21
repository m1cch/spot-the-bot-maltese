"""
Шаг 5b — генерация бот-корпуса из обученной LSTM (отдельно от обучения).

train_lstm.py делает обучение И генерацию, но генерация долгая; если процесс
прервётся (на сервере был рестарт контейнера) — она теряется целиком.
Этот скрипт грузит готовый чекпойнт models/lstm/mt_lstm.pt и генерирует
бот-корпус надёжно и быстро:
  - длины текстов сэмплятся из распределения человеческих, капятся --max-len;
  - тексты сортируются по длине -> в батче близкие длины -> нет холостых
    шагов на короткие тексты (генерация в разы быстрее);
  - запись в файл инкрементальная, с flush после каждого батча
    (краш теряет максимум один батч).

Запуск:
  python scripts/generate_bot.py --device cuda:0

Выход:
  corpus_bot/all_clean.txt   — бот-корпус (строка = текст)
  corpus_bot/_meta.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_lstm import LSTMLM, load_human_docs, SPECIALS, BOS, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/lstm/mt_lstm.pt")
    ap.add_argument("--corpus", default="corpus_clean_L2", help="для распределения длин")
    ap.add_argument("--n-texts", type=int, default=0, help="0 = как у человека")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=3000, help="кап длины текста")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="corpus_bot/all_clean.txt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"=== generate bot corpus | device={device} ===", flush=True)

    ck = torch.load(ROOT / args.ckpt, map_location=device)
    itos = ck["itos"]
    cfg = ck["config"]
    stoi = {w: i for i, w in enumerate(itos)}
    model = LSTMLM(len(itos), cfg["emb"], cfg["hidden"], cfg["layers"], cfg["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    print(f"checkpoint {args.ckpt}: vocab={len(itos)} | best_val_ppl≈"
          f"{cfg.get('epochs', '?')}ep", flush=True)

    docs = load_human_docs(ROOT / args.corpus)
    doc_lengths = np.array([len(d) for d in docs])
    n_texts = args.n_texts or len(docs)

    # целевые длины: сэмпл из человеческих длин, кап max_len, сортировка
    rng = np.random.default_rng(args.seed)
    targets = np.minimum(rng.choice(doc_lengths, n_texts, replace=True), args.max_len)
    targets.sort()
    print(f"texts={n_texts} | target len mean/median/max="
          f"{targets.mean():.0f}/{np.median(targets):.0f}/{targets.max()}", flush=True)

    bos = stoi[BOS]
    block = torch.tensor([stoi[t] for t in SPECIALS], device=device)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    written, tok_total = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in range(0, n_texts, args.gen_batch):
            tgt = targets[s:s + args.gen_batch]
            bs = len(tgt)
            maxlen = int(tgt.max())
            cur = torch.full((bs, 1), bos, dtype=torch.long, device=device)
            state = None
            seqs = [[] for _ in range(bs)]
            with torch.no_grad():
                for step in range(maxlen):
                    logits, state = model(cur, state)
                    logits = logits[:, -1, :].float() / args.temperature
                    logits[:, block] = float("-inf")
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                    ids = nxt.squeeze(1).tolist()
                    for b in range(bs):
                        if step < tgt[b]:
                            seqs[b].append(itos[ids[b]])
                    cur = nxt
            for d in seqs:
                f.write(" ".join(d) + "\n")
                tok_total += len(d)
            f.flush()
            written += bs
            print(f"  {written}/{n_texts}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    meta = {
        "n_texts": int(n_texts),
        "tokens_total": int(tok_total),
        "len_mean": float(tok_total / n_texts),
        "human_len_mean": float(doc_lengths.mean()),
        "temperature": args.temperature,
        "max_len": args.max_len,
        "vocab": len(itos),
        "model": "word-LSTM (generate_bot.py)",
    }
    (out_path.parent / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved bot corpus: {out_path} ({n_texts} texts, {tok_total} tokens)", flush=True)
    print(f"Bot len mean {tok_total/n_texts:.0f} (human {doc_lengths.mean():.0f})", flush=True)
    print(f"GENBOT_COMPLETE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
