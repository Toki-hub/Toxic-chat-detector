"""
train.py - latih model deteksi toxic dari train.csv
Contoh:
    python train.py --data-dir "C:\\path\\ke\\iiitb-toxic-comment.zip"     # zip langsung, di folder mana pun
    python train.py --data-dir /path/ke/folder_berisi_csv --epochs 4
Tanpa --data-dir, script mencari train.csv (atau *.zip yang berisi train.csv) di folder kerja,
folder ./data, dan folder script ini.
Catatan: test.csv pada dataset ini TIDAK berlabel (sample_submission.csv hanya berisi 0),
jadi evaluasi memakai validation split dari train.csv. Prediksi test.csv tetap dibuat
sebagai submission.csv.
"""
import argparse
import glob
import json
import os
import random
import time
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

from toxic_core import (DEFAULT_MODEL_DIR, HERE, LABELS, SUBS_PER_WORD, ToxicNet, collate,
                        encode, tokenize)


# ----------------------------------------------------------------------------
# Lokasi data: folder ATAU file .zip, di path mana pun
# ----------------------------------------------------------------------------
def list_names(source):
    """Nama file (huruf kecil, tanpa folder) di dalam folder atau zip."""
    if os.path.isdir(source):
        return {n.lower() for n in os.listdir(source)}
    if os.path.isfile(source) and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            return {os.path.basename(n).lower() for n in z.namelist()}
    return set()


def find_data_source(user_path=None):
    if user_path:
        src = os.path.abspath(os.path.expanduser(user_path))
        if "train.csv" in list_names(src):
            return src
        raise SystemExit(f"train.csv tidak ditemukan di: {src}\n"
                         "--data-dir boleh berupa FOLDER berisi train.csv atau FILE .zip yang memuatnya.")
    dirs = []
    for d in (os.getcwd(), os.path.join(os.getcwd(), "data"), HERE, os.path.join(HERE, "data")):
        if os.path.isdir(d) and d not in dirs:
            dirs.append(d)
    for d in dirs:
        if "train.csv" in list_names(d):
            return d
        for z in sorted(glob.glob(os.path.join(d, "*.zip"))):
            if "train.csv" in list_names(z):
                return z
    raise SystemExit("train.csv tidak ditemukan otomatis (dicari di: " + ", ".join(dirs) + ").\n"
                     'Tunjuk lokasinya:  python train.py --data-dir "D:\\unduhan\\iiitb-toxic-comment.zip"')


def read_csv_any(source, name):
    """Baca CSV bernama `name` dari folder atau zip (boleh di subfolder dalam zip). None bila tidak ada."""
    if os.path.isdir(source):
        for n in os.listdir(source):
            if n.lower() == name.lower():
                return pd.read_csv(os.path.join(source, n))
        return None
    with zipfile.ZipFile(source) as z:
        for n in z.namelist():
            if os.path.basename(n).lower() == name.lower():
                with z.open(n) as f:
                    return pd.read_csv(f)
    return None


def encode_all(texts, max_len):
    rows = []
    for t in texts:
        _, m = encode(t, max_len)
        rows.append(m if len(m) else np.zeros((1, SUBS_PER_WORD), np.int32))
    return rows


def make_batches(rows, batch_size, shuffle, rng):
    """Batch dengan panjang mirip (di-sort per chunk) -> jauh lebih cepat di CPU/GPU."""
    idx = np.arange(len(rows))
    if shuffle:
        rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: rows[i].shape[0])
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict_rows(model, rows, device, bs=256):
    model.eval()
    out = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    for b in make_batches(rows, bs, False, None):
        x = collate([rows[i] for i in b]).to(device)
        out[b] = torch.sigmoid(model(x)).cpu().numpy()
    return out


def label_auc(y, p):
    """ROC-AUC per label; NaN bila label itu hanya punya satu kelas di data validasi."""
    return [roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else float("nan")
            for j in range(y.shape[1])]


def mean_auc(y, p):
    ok = [v for v in label_auc(y, p) if v == v]      # lewati NaN
    return float(np.mean(ok)) if ok else 0.0


def best_thresholds(y, p):
    ths = []
    for j in range(y.shape[1]):
        best, bt = -1, 0.5
        for t in np.arange(0.05, 0.95, 0.025):
            f = f1_score(y[:, j], p[:, j] >= t, zero_division=0)
            if f > best:
                best, bt = f, t
        ths.append(float(bt))
    return ths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="folder berisi train.csv ATAU file .zip data (lokasi bebas); default: cari otomatis")
    ap.add_argument("--out", default=DEFAULT_MODEL_DIR, help="folder tujuan model (default: folder model di samping script)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--extra-csv", nargs="*", default=[],
                    help="CSV tambahan (kolom comment_text + label; label yang tidak ada dianggap 0), "
                         "mis. dataset toxic bahasa Indonesia")
    ap.add_argument("--sample", type=int, default=0, help="pakai N baris saja (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-test-pred", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    a.out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(a.out, exist_ok=True)
    source = find_data_source(a.data_dir)
    print(f"data  : {source}\nmodel : {a.out}")

    df = read_csv_any(source, "train.csv")
    for path in a.extra_csv:
        ex = pd.read_csv(os.path.expanduser(path))
        for l in LABELS:
            if l not in ex.columns:
                ex[l] = 0
        print(f"+ {len(ex)} baris tambahan dari {path}")
        df = pd.concat([df, ex[["comment_text"] + LABELS]], ignore_index=True)
    if a.sample:
        df = df.sample(a.sample, random_state=a.seed).reset_index(drop=True)
    texts = df["comment_text"].fillna("").tolist()
    Y = df[LABELS].values.astype(np.float32)

    perm = rng.permutation(len(df))
    n_val = int(len(df) * a.val_frac)
    va_idx, tr_idx = perm[:n_val], perm[n_val:]

    t0 = time.time()
    rows = encode_all(texts, a.max_len)
    print(f"encode selesai ({time.time() - t0:.0f}s), {len(rows)} teks")
    tr_rows, va_rows = [rows[i] for i in tr_idx], [rows[i] for i in va_idx]
    tr_y, va_y = Y[tr_idx], Y[va_idx]

    model = ToxicNet(dim=a.dim, hidden=a.hidden).to(device)
    pos = tr_y.mean(0).clip(min=1e-4)
    pos_weight = torch.tensor(np.clip(np.sqrt((1 - pos) / pos), 1, 10), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt_s = torch.optim.SparseAdam(list(model.emb.parameters()), lr=a.lr)
    opt_d = torch.optim.AdamW(model.dense_parameters(), lr=a.lr, weight_decay=1e-4)
    steps_total = a.epochs * ((len(tr_rows) + a.batch - 1) // a.batch)
    sched = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: max(0.05, 1 - s / steps_total))
             for o in (opt_s, opt_d)]

    best_auc, best_state, best_p = -1, None, None
    step = 0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        batches = make_batches(tr_rows, a.batch, True, rng)
        for bi, b in enumerate(batches, 1):
            x = collate([tr_rows[i] for i in b]).to(device)
            y = torch.from_numpy(tr_y[b]).to(device)
            loss = loss_fn(model(x), y)
            opt_s.zero_grad(); opt_d.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.dense_parameters(), 1.0)
            opt_s.step(); opt_d.step()
            for s in sched: s.step()
            run += loss.item(); step += 1
            if bi % 100 == 0:
                print(f"  ep{ep} batch {bi}/{len(batches)} loss {run / bi:.4f} ({time.time() - t0:.0f}s)", flush=True)
        p = predict_rows(model, va_rows, device)
        auc = mean_auc(va_y, p)
        per = {l: round(v, 4) for l, v in zip(LABELS, label_auc(va_y, p))}
        print(f"epoch {ep}: train_loss {run / len(batches):.4f} | val mean ROC-AUC {auc:.4f} | {per} | {time.time() - t0:.0f}s", flush=True)
        if auc > best_auc:
            best_auc, best_p = auc, p
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ths = best_thresholds(va_y, best_p)
    f1s = {l: round(f1_score(va_y[:, j], best_p[:, j] >= ths[j], zero_division=0), 4) for j, l in enumerate(LABELS)}
    print("threshold:", dict(zip(LABELS, [round(t, 3) for t in ths])))
    print("F1 val   :", f1s)

    # kosakata "dikenal" (untuk mendeteksi kata baru saat inferensi)
    cnt = Counter()
    for t in texts:
        cnt.update(w.lower() for w in tokenize(t) if w.isalpha())
    known = [w for w, c in cnt.items() if c >= 2]

    torch.save({"cfg": model.cfg, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "thresholds": ths, "known_words": known, "max_len": a.max_len,
                "val_auc": best_auc}, os.path.join(a.out, "model.pt"))

    # replay buffer kecil untuk belajar-online (agar tidak lupa)
    ridx = rng.choice(len(df), size=min(3000, len(df)), replace=False)
    with open(os.path.join(a.out, "replay.jsonl"), "w", encoding="utf-8") as f:
        for i in ridx:
            f.write(json.dumps({"text": texts[i][:500], "y": [int(v) for v in Y[i]]}, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump({"val_mean_auc": best_auc, "f1": f1s, "thresholds": dict(zip(LABELS, ths)),
                   "n_train": len(tr_rows), "n_val": len(va_rows), "epochs": a.epochs}, f, indent=2)

    te = None if a.no_test_pred else read_csv_any(source, "test.csv")
    if te is not None:
        te_rows = encode_all(te["comment_text"].fillna("").tolist(), a.max_len)
        pt = predict_rows(model, te_rows, device)
        sub = pd.DataFrame(pt, columns=LABELS); sub.insert(0, "id", te["id"])
        sub.to_csv(os.path.join(a.out, "submission.csv"), index=False)
        print("submission.csv ditulis:", sub.shape)
    print("Selesai. Model tersimpan di", a.out)


if __name__ == "__main__":
    main()
