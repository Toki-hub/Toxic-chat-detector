"""
cli.py - pakai model lewat terminal.
    python cli.py predict "you are an idiot"
    python cli.py teach "text" --labels toxic insult       # ajari model (belajar dari koreksi)
    python cli.py teach "text" --labels none               # koreksi: ini BUKAN toxic
    python cli.py chat                                     # mode interaktif
    python cli.py newwords                                 # kata baru yang pernah terdeteksi
"""
import argparse
import json
import os

from toxic_core import DEFAULT_MODEL_DIR, LABELS, ToxicDetector


def show(res, text=None):
    if text:
        print(f"\n> {text}")
    bar = lambda p: "█" * int(p * 20) + "░" * (20 - int(p * 20))
    for l in LABELS:
        p = res["probs"][l]
        mark = " <== " if l in res["flags"] else ""
        print(f"  {l:14s} {bar(p)} {p:5.1%}{mark}")
    print("  Hasil:", "TOXIC " + str(res["flags"]) if res["is_toxic"] else "aman")
    if res.get("culprits"):
        print("  Kata pemicu:", ", ".join(w for _, w, d in res["culprits"]))
    if res["new_words"]:
        print("  Kosakata baru (belum pernah dilihat saat training):", ", ".join(sorted(set(res["new_words"]))))


def parse_labels(items):
    if not items or items == ["none"]:
        return {}
    bad = [i for i in items if i not in LABELS]
    if bad:
        raise SystemExit(f"label tidak dikenal: {bad}. Pilihan: {LABELS} atau 'none'")
    return {i: 1 for i in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="folder berisi model.pt (default: folder model di samping script)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict"); p.add_argument("text", nargs="+")
    t = sub.add_parser("teach"); t.add_argument("text"); t.add_argument("--labels", nargs="*", default=["none"])
    tf = sub.add_parser("teachfile", help="ajari dari CSV (kolom comment_text + label 0/1)")
    tf.add_argument("csv"); tf.add_argument("--passes", type=int, default=2)
    sub.add_parser("chat"); sub.add_parser("newwords")
    a = ap.parse_args()
    a.model_dir = os.path.abspath(os.path.expanduser(a.model_dir))

    if a.cmd == "newwords":
        path = os.path.join(a.model_dir, "new_words.json")
        if not os.path.exists(path):
            print("Belum ada kata baru yang tercatat."); return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print("Kata yang belum pernah dilihat saat training (paling sering di atas):")
        for w, n in list(data.items())[:50]:
            print(f"  {w:20s} {n}x")
        print("\nAjari model dengan:  python cli.py teach \"kalimat berisi kata itu\" --labels toxic insult")
        return

    det = ToxicDetector(a.model_dir)
    if a.cmd == "predict":
        for text in a.text:
            show(det.predict(text, explain=True), text)
    elif a.cmd == "teach":
        y = parse_labels(a.labels)
        r = det.learn(a.text, y)
        print("Sebelum:"); show(r["before"][0])
        print("Sesudah:"); show(r["after"][0])
        print("Model diperbarui & disimpan.")
    elif a.cmd == "teachfile":
        import pandas as pd
        df = pd.read_csv(os.path.expanduser(a.csv))
        for l in LABELS:
            if l not in df.columns:
                df[l] = 0
        texts = df["comment_text"].fillna("").astype(str).tolist()
        ys = df[LABELS].values.tolist()
        for p in range(a.passes):
            order = list(range(len(texts))); import random; random.shuffle(order)
            for s in range(0, len(order), 32):
                idx = order[s:s + 32]
                det.learn([texts[i] for i in idx], [ys[i] for i in idx], epochs=2, save=False)
            print(f"pass {p + 1}/{a.passes} selesai")
        det.save()
        print(f"{len(texts)} contoh dipelajari. Model disimpan.")
    elif a.cmd == "chat":
        print("Ketik kalimat untuk dicek. Perintah: /ajari <label,label|none>  (mengajari kalimat terakhir), /keluar")
        last = None
        while True:
            try:
                s = input("\n>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not s:
                continue
            if s == "/keluar":
                break
            if s.startswith("/ajari"):
                if not last:
                    print("belum ada kalimat."); continue
                arg = s[len("/ajari"):].strip().replace(",", " ").split()
                r = det.learn(last, parse_labels(arg))
                show(r["after"][0], "setelah belajar")
                continue
            last = s
            show(det.predict(s, explain=True))


if __name__ == "__main__":
    main()
