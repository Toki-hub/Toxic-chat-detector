"""
toxic_core.py
=============
Inti sistem deteksi komentar toxic:
  * Tokenizer subword berbasis hash (mirip FastText) -> kata baru / kata disamarkan
    ("f*ck", "idi0t", "stuuupid") tetap punya representasi, tanpa perlu vocab tetap.
  * Model BiGRU + attention/max/mean pooling (PyTorch), 6 label sekaligus (multi-label).
  * ToxicDetector: prediksi, penjelasan kata (occlusion), dan belajar dari koreksi
    pengguna (online learning dengan replay agar tidak lupa).
"""
import json
import os
import random
import re
import time
import unicodedata
import zlib
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Lokasi folder script ini -> model dicari relatif ke SINI, bukan ke folder tempat
# perintah dijalankan (cwd). Bisa ditimpa dengan --model-dir atau env TOXIC_MODEL_DIR.
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.environ.get("TOXIC_MODEL_DIR") or os.path.join(HERE, "model")

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_BUCKETS = 200_003      # id 0 = padding
SUBS_PER_WORD = 12         # jumlah sub-id (kata + n-gram karakter) per kata

# ----------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_TOKEN = re.compile(r"[\w@$*'\-]+|[!?]+")
_REPEAT = re.compile(r"(.)\1{2,}")


def _h(s: str) -> int:
    # crc32 bersifat deterministik antar proses (hash() bawaan Python TIDAK).
    return zlib.crc32(s.encode("utf-8")) % (NUM_BUCKETS - 1) + 1


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = _URL.sub(" url ", text)
    text = _IP.sub(" ipaddr ", text)
    return text


def tokenize(text: str):
    return _TOKEN.findall(normalize(text))


_SUB_CACHE = {}


def word_subids(raw: str) -> np.ndarray:
    """Kata mentah -> array sub-id hash (ukuran tetap SUBS_PER_WORD)."""
    hit = _SUB_CACHE.get(raw)
    if hit is not None:
        return hit
    caps = len(raw) >= 3 and raw.isupper()
    w = _REPEAT.sub(r"\1\1", raw.lower())
    ids = [_h("w:" + w)]
    if caps:
        ids.append(_h("<CAPS>"))          # penanda TULISAN KAPITAL (teriak)
    padded = f"<{w}>"
    for n in (3, 4, 5):
        for i in range(len(padded) - n + 1):
            ids.append(_h(padded[i:i + n]))
    ids = ids[:SUBS_PER_WORD]
    arr = np.zeros(SUBS_PER_WORD, dtype=np.int32)
    arr[:len(ids)] = ids
    if len(_SUB_CACHE) < 2_000_000:
        _SUB_CACHE[raw] = arr
    return arr


def encode(text: str, max_len: int = 100):
    """Teks -> (daftar kata, matriks [n_kata, SUBS_PER_WORD]).
    Teks panjang dipotong: 75% bagian awal + 25% bagian akhir."""
    words = tokenize(text)
    if len(words) > max_len:
        head = int(max_len * 0.75)
        words = words[:head] + words[-(max_len - head):]
    if not words:
        return [], np.zeros((0, SUBS_PER_WORD), dtype=np.int32)
    return words, np.stack([word_subids(w) for w in words])


def collate(rows):
    """List matriks [n_i, K] -> tensor long [B, L, K] (zero-padded)."""
    max_len = max(1, max(r.shape[0] for r in rows))
    out = np.zeros((len(rows), max_len, SUBS_PER_WORD), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :r.shape[0]] = r
    return torch.from_numpy(out)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class ToxicNet(nn.Module):
    def __init__(self, num_buckets=NUM_BUCKETS, dim=96, hidden=96, drop=0.3,
                 n_labels=len(LABELS)):
        super().__init__()
        self.cfg = dict(num_buckets=num_buckets, dim=dim, hidden=hidden,
                        drop=drop, n_labels=n_labels)
        self.emb = nn.Embedding(num_buckets, dim, padding_idx=0, sparse=True)
        nn.init.normal_(self.emb.weight, 0.0, 0.1)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.drop = nn.Dropout(drop)
        self.rnn = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(6 * hidden, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, n_labels))

    def dense_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("emb.")]

    def forward(self, x):                       # x: [B, L, K]
        B, L, _ = x.shape
        sub_mask = (x > 0).unsqueeze(-1).float()
        wv = (self.emb(x) * sub_mask).sum(2) / sub_mask.sum(2).clamp(min=1)
        wmask = x[:, :, 0] > 0                  # sub-id pertama selalu id kata
        lengths = wmask.sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(self.drop(wv), lengths, batch_first=True,
                                      enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=L)

        a = self.att(h).squeeze(-1).masked_fill(~wmask, -1e4)
        a = torch.softmax(a, 1)
        att_pool = (h * a.unsqueeze(-1)).sum(1)
        max_pool = h.masked_fill(~wmask.unsqueeze(-1), -1e4).max(1).values
        any_word = wmask.any(1, keepdim=True)
        max_pool = torch.where(any_word, max_pool, torch.zeros_like(max_pool))
        mean_pool = (h * wmask.unsqueeze(-1)).sum(1) / wmask.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([att_pool, max_pool, mean_pool], 1))


# ----------------------------------------------------------------------------
# Detector (dipakai aplikasi / CLI)
# ----------------------------------------------------------------------------
class ToxicDetector:
    def __init__(self, model_dir=None, device=None):
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir or DEFAULT_MODEL_DIR))
        if not os.path.exists(os.path.join(self.model_dir, "model.pt")):
            raise FileNotFoundError(
                f"model.pt tidak ditemukan di: {self.model_dir}\n"
                "Tunjuk folder model dengan --model-dir <folder>, atau set env TOXIC_MODEL_DIR, "
                "atau latih dulu dengan train.py.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(os.path.join(self.model_dir, "model.pt"), map_location="cpu",
                          weights_only=False)
        self.model = ToxicNet(**ckpt["cfg"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.thresholds = np.array(ckpt.get("thresholds", [0.5] * len(LABELS)), dtype=float)
        self.known_words = set(ckpt.get("known_words", []))
        self.max_len = ckpt.get("max_len", 100)
        self.replay = self._read_jsonl("replay.jsonl")
        self.feedback = self._read_jsonl("feedback.jsonl")
        self.new_word_counts = Counter()

    # -- util file ---------------------------------------------------------
    def _path(self, name):
        return os.path.join(self.model_dir, name)

    def _read_jsonl(self, name):
        p = self._path(name)
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _append_jsonl(self, name, obj):
        with open(self._path(name), "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # -- prediksi ------------------------------------------------------------
    @torch.no_grad()
    def _probs(self, rows, bs=128):
        self.model.eval()
        out = []
        for i in range(0, len(rows), bs):
            x = collate(rows[i:i + bs]).to(self.device)
            out.append(torch.sigmoid(self.model(x)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))

    def predict(self, texts, explain=False):
        """texts: str atau list[str]. Mengembalikan list dict per teks."""
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        enc = [encode(t, self.max_len) for t in texts]
        rows = [m if len(w) else np.zeros((1, SUBS_PER_WORD), np.int32) for w, m in enc]
        probs = self._probs(rows)
        results = []
        for (words, mat), p in zip(enc, probs):
            if not words:
                p = np.zeros(len(LABELS))
            flags = [l for l, pi, th in zip(LABELS, p, self.thresholds) if pi >= th]
            res = {
                "probs": {l: float(pi) for l, pi in zip(LABELS, p)},
                "flags": flags,
                "is_toxic": len(flags) > 0,
                "score": float(p.max()) if len(words) else 0.0,
                "new_words": self.unknown_words(words),
            }
            if explain and flags and 1 < len(words) <= 80:
                res["culprits"] = self._occlusion(words, mat, p)
            results.append(res)
        return results[0] if single else results

    def _occlusion(self, words, mat, base_probs, top=5):
        """Seberapa turun skor (skala logit, agar tidak jenuh di ~1.0) jika kata ke-i
        dihapus -> kata pemicu toxic."""
        j = int(np.argmax(base_probs))
        variants = []
        for i in range(len(words)):
            keep = np.arange(len(words)) != i
            variants.append(mat[keep])
        pv = self._probs(variants)[:, j]
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        drop = logit(base_probs[j]) - logit(pv)
        mx = float(drop.max())
        if mx <= 0.5:
            return []
        order = np.argsort(-drop)[:top]
        return [(int(i), words[i], float(drop[i])) for i in order if drop[i] >= max(0.5, 0.35 * mx)]

    # -- kosakata baru ------------------------------------------------------
    def unknown_words(self, words):
        out = []
        for w in words:
            lw = _REPEAT.sub(r"\1\1", w.lower())
            if lw.isalpha() and len(lw) > 2 and lw not in self.known_words:
                out.append(lw)
        return out

    def register_new_words(self, words, persist=True):
        """Catat kata baru yang muncul (disimpan di model/new_words.json) agar bisa
        ditinjau lalu diajarkan lewat `cli.py teach`."""
        words = list(words)
        if not words:
            return
        for w in words:
            self.new_word_counts[w] += 1
        if persist:
            path = self._path("new_words.json")
            old = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    old = json.load(f)
            for w in words:
                old[w] = old.get(w, 0) + 1
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dict(sorted(old.items(), key=lambda kv: -kv[1])), f, ensure_ascii=False, indent=1)

    # -- belajar dari koreksi -----------------------------------------------
    def learn(self, texts, labels, epochs=8, lr_dense=3e-4, lr_sparse=3e-3,
              replay_per_step=48, save=True):
        """
        Fine-tune cepat dari koreksi pengguna.
        texts : str / list[str]
        labels: dict {label: 0/1} / list 6 angka / list of those (sesuai texts)
        Data lama (replay + feedback sebelumnya) dicampur agar model tidak lupa.
        """
        if isinstance(texts, str):
            texts, labels = [texts], [labels]
        labels = [self._vec(l) for l in labels]
        before = self.predict(texts)
        for t, l in zip(texts, labels):
            rec = {"text": t[:1000], "y": l, "ts": time.time()}
            self.feedback.append(rec)
            self._append_jsonl("feedback.jsonl", rec)
            self.register_new_words(self.unknown_words(tokenize(t)))

        new_rows = [encode(t, self.max_len)[1] for t in texts]
        new_rows = [r if len(r) else np.zeros((1, SUBS_PER_WORD), np.int32) for r in new_rows]
        new_y = list(labels)
        old_pool = [(r["text"], r["y"]) for r in (self.replay + self.feedback[:-len(texts)])]

        opt_s = torch.optim.SparseAdam(list(self.model.emb.parameters()), lr=lr_sparse)
        opt_d = torch.optim.Adam(self.model.dense_parameters(), lr=lr_dense)
        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(epochs):
            rows, ys = list(new_rows) * 2, list(new_y) * 2      # data baru dibobot lebih
            k = min(replay_per_step, len(old_pool))
            for t, y in random.sample(old_pool, k) if k else []:
                m = encode(t, self.max_len)[1]
                rows.append(m if len(m) else np.zeros((1, SUBS_PER_WORD), np.int32))
                ys.append(y)
            x = collate(rows).to(self.device)
            y = torch.tensor(ys, dtype=torch.float32, device=self.device)
            loss = loss_fn(self.model(x), y)
            opt_s.zero_grad(); opt_d.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.dense_parameters(), 1.0)
            opt_s.step(); opt_d.step()
        self.model.eval()
        # kata yang baru dipelajari dianggap "dikenal"
        for t in texts:
            self.known_words.update(w.lower() for w in tokenize(t) if w.isalpha())
        if save:
            self.save()
        return {"before": before, "after": self.predict(texts)}

    @staticmethod
    def _vec(l):
        if isinstance(l, dict):
            return [int(l.get(k, 0)) for k in LABELS]
        return [int(v) for v in l]

    def save(self):
        torch.save({
            "cfg": self.model.cfg,
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "thresholds": self.thresholds.tolist(),
            "known_words": sorted(self.known_words),
            "max_len": self.max_len,
        }, self._path("model.pt"))
