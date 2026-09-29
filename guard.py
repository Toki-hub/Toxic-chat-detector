"""
guard.py - "Grammarly untuk toksisitas": memantau teks di layar / clipboard / ketikan
dan memberi peringatan bila terdeteksi toxic.

  python guard.py screen     # baca layar via OCR, beri kotak merah di teks toxic (overlay penuh hanya di Windows)
  python guard.py clipboard  # cek teks yang Anda copy
  python guard.py typing     # cek apa yang Anda ketik (F9 = jeda/lanjut)

Semua diproses LOKAL. Tidak ada teks yang dikirim ke internet atau disimpan
(kecuali koreksi yang Anda kirim lewat tombol di notifikasi -> model/feedback.jsonl).
"""
import argparse
import hashlib
import importlib.util
import os
import queue
import re
import signal
import sys
import threading
import time

from toxic_core import DEFAULT_MODEL_DIR, LABELS, ToxicDetector

LABEL_ID = {"toxic": "Toxic", "severe_toxic": "Sangat toxic", "obscene": "Kata kasar/cabul",
            "threat": "Ancaman", "insult": "Hinaan", "identity_hate": "Kebencian identitas"}


def note_new_words(det, res):
    if res.get("is_toxic") and res.get("new_words"):
        det.register_new_words(res["new_words"])


def describe(res):
    top = sorted(res["flags"], key=lambda l: -res["probs"][l])
    return ", ".join(f"{LABEL_ID[l]} {res['probs'][l]:.0%}" for l in top[:3])


# ----------------------------------------------------------------------------
# Logika OCR (murni, bisa diuji tanpa layar)
# ----------------------------------------------------------------------------
def group_ocr(data, min_conf=50):
    """Hasil pytesseract.image_to_data (dict) -> daftar paragraf
    [{'text':..., 'bbox':(x,y,w,h), 'words':[(teks,(x,y,w,h)),...]}]"""
    paras = {}
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if not txt or conf < min_conf:
            continue
        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i])
        box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        paras.setdefault(key, []).append((txt, box))
    out = []
    for words in paras.values():
        x0 = min(b[0] for _, b in words); y0 = min(b[1] for _, b in words)
        x1 = max(b[0] + b[2] for _, b in words); y1 = max(b[1] + b[3] for _, b in words)
        text = " ".join(w for w, _ in words)
        if len(text) >= 3:
            out.append({"text": text, "bbox": (x0, y0, x1 - x0, y1 - y0), "words": words})
    return out


def _clean(w):
    return re.sub(r"[^\w*@$']", "", w.lower())


def match_culprits(para, culprits):
    """Cocokkan kata pemicu dari model dengan kotak kata hasil OCR."""
    targets = {_clean(w) for _, w, _ in culprits}
    return [box for w, box in para["words"] if _clean(w) in targets and _clean(w)]


# ----------------------------------------------------------------------------
# Persiapan & keamanan: cek Tesseract di awal, jangan membaca jendela sendiri
# ----------------------------------------------------------------------------
TESSERACT_HELP = r"""
Program Tesseract OCR belum terpasang atau tidak ditemukan.
('pytesseract' hanya penghubung Python; program Tesseract-nya harus dipasang terpisah.)

Windows:
  1) Pasang:  winget install --id UB-Mannheim.TesseractOCR -e
     atau unduh installer dari https://github.com/UB-Mannheim/tesseract/wiki
  2) Tutup lalu buka PowerShell BARU, cek:  tesseract --version
  3) Jika tetap tidak ditemukan, beri path lengkap:
       python guard.py screen --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
Linux : sudo apt install tesseract-ocr
macOS : brew install tesseract
"""


def _tesseract_candidates():
    if not sys.platform.startswith("win"):
        return []
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    return [os.path.join(pf, "Tesseract-OCR", "tesseract.exe"),
            os.path.join(pf86, "Tesseract-OCR", "tesseract.exe"),
            os.path.join(local, "Programs", "Tesseract-OCR", "tesseract.exe")]


def setup_tesseract(pytesseract, cmd=None, lang="eng"):
    """Cari & verifikasi Tesseract SEBELUM memulai (gagal cepat, bukan looping).
    Urutan: --tesseract-cmd, PATH, lalu lokasi instalasi umum di Windows."""
    tried, found, ver = [], None, None
    options = ([os.path.expanduser(cmd)] if cmd else []) + [None] + _tesseract_candidates()
    for c in options:
        if c is not None and not os.path.isfile(c):
            tried.append(f"{c} (file tidak ada)")
            continue
        pytesseract.pytesseract.tesseract_cmd = c or "tesseract"
        try:
            ver = str(pytesseract.get_tesseract_version())
            found = c or "tesseract (dari PATH)"
            break
        except Exception:
            tried.append(c or "PATH")
    if not found:
        raise SystemExit("[ERROR]" + TESSERACT_HELP + "Sudah dicoba: " + "; ".join(tried))
    try:
        have = set(pytesseract.get_languages(config=""))
    except Exception:
        have = set()
    missing = [l for l in lang.split("+") if have and l not in have]
    if missing:
        raise SystemExit(f"[ERROR] Data bahasa Tesseract '{'+'.join(missing)}' belum terpasang "
                         f"(terpasang: {', '.join(sorted(have))}).\n"
                         "Pakai --lang eng, atau pasang data bahasa tersebut.")
    return ver, found


def enable_dpi_awareness():
    """Windows: koordinat layar = piksel fisik (penting untuk overlay & penutup jendela)."""
    if not sys.platform.startswith("win"):
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def own_console_rect():
    """Windows: kotak (l,t,r,b) jendela terminal tempat program ini berjalan, agar OCR tidak
    membaca log kita sendiri (yang bisa memicu loop). None bila tidak diketahui."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if not hwnd:
            return None
        hwnd = ctypes.windll.user32.GetAncestor(hwnd, 3) or hwnd   # 3 = GA_ROOTOWNER (Windows Terminal)
        r = wintypes.RECT()
        if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return (r.left, r.top, r.right, r.bottom)
    except Exception:
        pass
    return None


def mask_rect(img, rect):
    """Tutup area (l,t,r,b) pada gambar dengan warna putih agar tidak ikut di-OCR."""
    from PIL import ImageDraw
    l, t, r, b = [int(v) for v in rect]
    if r <= 0 or b <= 0 or l >= img.width or t >= img.height:
        return img
    ImageDraw.Draw(img).rectangle([max(l, 0), max(t, 0), min(r, img.width), min(b, img.height)], fill="white")
    return img


SELF_MARKERS = ("[toxic]", "ocr error", "kemungkinan toxic", "kata pemicu", "memantau layar")


def is_self_text(text):
    """Teks yang berasal dari log/notifikasi program ini sendiri -> jangan dianalisis ulang."""
    t = text.lower()
    return any(m in t for m in SELF_MARKERS)


# ----------------------------------------------------------------------------
# UI (tkinter): notifikasi + overlay
# ----------------------------------------------------------------------------
class UI:
    def __init__(self, det, lock):
        import tkinter as tk
        self.tk = tk
        self.det, self.lock = det, lock
        self.root = tk.Tk()
        self.root.withdraw()
        self.q = queue.Queue()
        self.toast_win = None
        self.overlay = None
        self.stop_requested = False
        self.exit_code = 0
        self.root.after(100, self._poll)

    def post(self, fn, *args):
        """Aman dipanggil dari thread lain: jalankan fn di thread UI."""
        self.q.put((fn, args))

    def request_stop(self, code=0):
        """Aman dipanggil dari thread mana pun / signal handler (Ctrl+C)."""
        self.exit_code = code or self.exit_code
        self.stop_requested = True

    def toast_rect(self):
        """Kotak notifikasi yang sedang tampil (untuk ditutup saat screenshot)."""
        w = self.toast_win
        try:
            if w is not None and w.winfo_exists():
                x, y = w.winfo_rootx(), w.winfo_rooty()
                return (x, y, x + w.winfo_width(), y + w.winfo_height())
        except Exception:
            pass
        return None

    def _poll(self):
        if self.stop_requested:
            self.root.quit()
            return
        try:
            while True:
                fn, args = self.q.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # -- notifikasi -----------------------------------------------------------
    def toast(self, text, res, seconds=10):
        tk = self.tk
        if self.toast_win is not None:
            try:
                self.toast_win.destroy()
            except Exception:
                pass
        w = tk.Toplevel(self.root)
        self.toast_win = w
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        w.configure(bg="#7a1f1f", padx=2, pady=2)
        f = tk.Frame(w, bg="#fff8f6", padx=12, pady=10)
        f.pack()
        tk.Label(f, text="⚠ Kemungkinan toxic: " + describe(res), bg="#fff8f6", fg="#a61b1b",
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x")
        preview = text.replace("\n", " ")
        preview = preview if len(preview) <= 140 else preview[:137] + "..."
        tk.Label(f, text=preview, bg="#fff8f6", fg="#222", wraplength=380, justify="left",
                 font=("Segoe UI", 9)).pack(fill="x", pady=(4, 2))
        if res.get("culprits"):
            tk.Label(f, text="Kata pemicu: " + ", ".join(c[1] for c in res["culprits"]),
                     bg="#fff8f6", fg="#555", font=("Segoe UI", 9, "italic")).pack(fill="x")
        if res.get("new_words"):
            tk.Label(f, text="Kata baru: " + ", ".join(sorted(set(res["new_words"]))[:5]),
                     bg="#fff8f6", fg="#886", font=("Segoe UI", 8)).pack(fill="x")
        row = tk.Frame(f, bg="#fff8f6"); row.pack(fill="x", pady=(8, 0))
        status = tk.Label(row, text="", bg="#fff8f6", fg="#2a6", font=("Segoe UI", 8))

        def teach(y, msg):
            status.config(text="belajar...")
            threading.Thread(target=self._teach, args=(text, y, status, msg), daemon=True).start()

        tk.Button(row, text="Bukan toxic", command=lambda: teach({}, "dikoreksi ✓")).pack(side="left")
        tk.Button(row, text="Benar toxic", command=lambda: teach({l: 1 for l in res["flags"]}, "dipelajari ✓")
                  ).pack(side="left", padx=6)
        tk.Button(row, text="Tutup", command=w.destroy).pack(side="right")
        status.pack(side="left", padx=6)
        w.update_idletasks()
        sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
        w.geometry(f"+{sw - w.winfo_width() - 24}+{sh - w.winfo_height() - 72}")
        w.after(int(seconds * 1000), lambda: w.destroy() if w.winfo_exists() else None)

    def _teach(self, text, y, status, msg):
        with self.lock:
            self.det.learn(text, y)
        self.post(lambda: status.config(text=msg) if status.winfo_exists() else None)

    # -- overlay (Windows) ---------------------------------------------------------
    def make_overlay(self):
        if not sys.platform.startswith("win"):
            return None
        import ctypes
        tk = self.tk
        w = tk.Toplevel(self.root)
        sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        w.geometry(f"{sw}x{sh}+0+0")
        w.configure(bg="#010101")
        w.attributes("-transparentcolor", "#010101")
        cv = tk.Canvas(w, bg="#010101", highlightthickness=0)
        cv.pack(fill="both", expand=True)
        w.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(w.winfo_id())
        GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT = -20, 0x80000, 0x20
        st = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, st | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        self.overlay = (w, cv)
        return self.overlay

    def draw_boxes(self, items):
        if not self.overlay:
            return
        _, cv = self.overlay
        cv.delete("all")
        for it in items:
            for (x, y, bw, bh) in it["boxes"]:
                cv.create_rectangle(x - 2, y - 2, x + bw + 2, y + bh + 2, outline="#ff2d2d", width=3)
            x, y, bw, bh = it["para_bbox"]
            cv.create_text(x, max(0, y - 14), text=it["label"], anchor="w", fill="#ff2d2d",
                           font=("Segoe UI", 9, "bold"))

    def set_overlay_visible(self, visible):
        if self.overlay:
            self.overlay[0].attributes("-alpha", 1.0 if visible else 0.0)


# ----------------------------------------------------------------------------
# Mode: layar (OCR)
# ----------------------------------------------------------------------------
def run_screen(ui, det, lock, a):
    import mss
    import pytesseract
    from PIL import Image

    overlay = None if a.no_overlay else ui.make_overlay()
    if overlay is None and not a.no_overlay:
        print("[info] overlay kotak merah hanya tersedia di Windows; memakai notifikasi saja.")
    work_q, cache, alerted = queue.Queue(maxsize=1), {}, set()
    sct = (getattr(mss, "MSS", None) or mss.mss)()     # dibuat & dipakai di thread UI
    if a.region:
        x, y, w, h = [int(v) for v in a.region.split(",")]
        mon = {"left": x, "top": y, "width": w, "height": h}
    else:
        mon = sct.monitors[1]
    off_x, off_y = mon["left"], mon["top"]
    state = {"hash": None}

    MAX_FAILS = 3

    def worker():
        fails = 0
        while True:
            img = work_q.get()
            try:
                data = pytesseract.image_to_data(img, lang=a.lang, output_type=pytesseract.Output.DICT)
                paras = [p for p in group_ocr(data, a.min_conf) if not is_self_text(p["text"])]
                new = [p for p in paras if p["text"] not in cache]
                if new:
                    with lock:
                        results = det.predict([p["text"] for p in new], explain=True)
                    for p, r in zip(new, results):
                        cache[p["text"]] = r
                        note_new_words(det, r)
                        if len(cache) > 2000:
                            cache.pop(next(iter(cache)))
                items = []
                for p in paras:
                    r = cache[p["text"]]
                    if not r["is_toxic"]:
                        continue
                    boxes = match_culprits(p, r.get("culprits", [])) or [p["bbox"]]
                    shift = lambda b: (b[0] + off_x, b[1] + off_y, b[2], b[3])
                    items.append({"boxes": [shift(b) for b in boxes], "para_bbox": shift(p["bbox"]),
                                  "label": describe(r), "text": p["text"], "res": r})
                ui.post(on_result, items)
                fails = 0
            except Exception as e:
                fails += 1
                print(f"[OCR error {fails}/{MAX_FAILS}] {type(e).__name__}: {e}")
                fatal = isinstance(e, getattr(pytesseract, "TesseractNotFoundError", ()))
                if fatal or fails >= MAX_FAILS:
                    print("Dihentikan karena OCR gagal. Perbaiki penyebabnya, lalu jalankan lagi.")
                    ui.request_stop(1)
                    return

    def on_result(items):
        ui.draw_boxes(items)
        for it in items:
            if it["text"] not in alerted:
                alerted.add(it["text"])
                print(f"[TOXIC] {describe(it['res'])} :: {it['text'][:100]}")
                if not overlay:
                    ui.toast(it["text"], it["res"])

    tick_fails = {"n": 0}

    def tick():
        if ui.stop_requested:
            return
        try:
            ui.set_overlay_visible(False)
            ui.root.update()
            time.sleep(0.06)
            shot = sct.grab(mon)
            ui.set_overlay_visible(True)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            # Tutup jendela kita sendiri (terminal + notifikasi) agar log/peringatan tidak
            # terbaca ulang oleh OCR dan memicu loop.
            for r in (own_console_rect(), ui.toast_rect()):
                if r:
                    mask_rect(img, (r[0] - off_x, r[1] - off_y, r[2] - off_x, r[3] - off_y))
            hsh = hashlib.md5(img.tobytes()).hexdigest()
            if hsh != state["hash"] and not work_q.full():
                state["hash"] = hsh
                work_q.put(img)
            tick_fails["n"] = 0
        except Exception as e:
            try:
                ui.set_overlay_visible(True)
            except Exception:
                pass
            tick_fails["n"] += 1
            print(f"[capture error {tick_fails['n']}/{MAX_FAILS}] {type(e).__name__}: {e}")
            if tick_fails["n"] >= MAX_FAILS:
                print("Dihentikan karena gagal mengambil layar berulang.")
                ui.request_stop(1)
                return
        ui.root.after(int(a.interval * 1000), tick)

    threading.Thread(target=worker, daemon=True).start()
    ui.root.after(500, tick)
    print("Memantau layar... (Ctrl+C untuk berhenti). Saran: pakai --region agar hanya area chat yang dipindai.")


# ----------------------------------------------------------------------------
# Mode: clipboard
# ----------------------------------------------------------------------------
def run_clipboard(ui, det, lock, a):
    import pyperclip

    def loop():
        last = pyperclip.paste()
        while True:
            time.sleep(0.5)
            try:
                cur = pyperclip.paste()
            except Exception:
                continue
            if cur != last and cur and len(cur.strip()) >= 3:
                last = cur
                with lock:
                    r = det.predict(cur[:5000], explain=True)
                note_new_words(det, r)
                print(("[TOXIC] " + describe(r)) if r["is_toxic"] else "[aman]", "::", cur[:80].replace("\n", " "))
                if r["is_toxic"]:
                    ui.post(ui.toast, cur, r)

    threading.Thread(target=loop, daemon=True).start()
    print("Memantau clipboard... (Ctrl+C untuk berhenti)")


# ----------------------------------------------------------------------------
# Mode: mengetik
# ----------------------------------------------------------------------------
def run_typing(ui, det, lock, a):
    from pynput import keyboard
    buf, st = [], {"paused": False, "timer": None}

    def check(clear):
        text = "".join(buf).strip()
        if clear:
            buf.clear()
        if len(text) < 3:
            return
        with lock:
            r = det.predict(text, explain=True)
        note_new_words(det, r)
        if r["is_toxic"]:
            print("[TOXIC]", describe(r), "::", text[:80])
            ui.post(ui.toast, text, r)

    def idle_flush():
        check(clear=True)

    def on_press(key):
        if key == keyboard.Key.f9:
            st["paused"] = not st["paused"]
            buf.clear()
            print("[typing]", "JEDA" if st["paused"] else "AKTIF")
            return
        if st["paused"]:
            return
        if st["timer"]:
            st["timer"].cancel()
        ch = getattr(key, "char", None)
        if ch and ch.isprintable():
            buf.append(ch)
            if ch in ".!?":
                threading.Thread(target=check, args=(False,), daemon=True).start()
        elif key == keyboard.Key.space:
            buf.append(" ")
        elif key == keyboard.Key.backspace and buf:
            buf.pop()
        elif key == keyboard.Key.enter:
            threading.Thread(target=check, args=(True,), daemon=True).start()
            return
        st["timer"] = threading.Timer(a.idle, idle_flush)
        st["timer"].daemon = True
        st["timer"].start()

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    print("Memantau ketikan (F9 = jeda/lanjut; JEDA saat mengetik password!)...")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["screen", "clipboard", "typing"])
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="folder berisi model.pt (default: folder model di samping script)")
    ap.add_argument("--interval", type=float, default=3.0, help="(screen) jeda antar pemindaian, detik")
    ap.add_argument("--region", help="(screen) x,y,lebar,tinggi area yang dipantau; disarankan agar cepat & akurat")
    ap.add_argument("--lang", default="eng", help="(screen) bahasa tesseract, mis. eng / ind / eng+ind")
    ap.add_argument("--min-conf", type=float, default=50, help="(screen) minimal confidence OCR")
    ap.add_argument("--tesseract-cmd", help=r"path tesseract.exe, mis. C:\Program Files\Tesseract-OCR\tesseract.exe")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--idle", type=float, default=2.5, help="(typing) detik diam sebelum teks diperiksa")
    a = ap.parse_args()

    # 1) Library yang dibutuhkan mode ini
    needs = {"screen": ["mss", "pytesseract", "PIL"], "clipboard": ["pyperclip"], "typing": ["pynput"]}[a.mode]
    missing = [m for m in needs if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit("[ERROR] Library belum terpasang: " + ", ".join(missing) +
                         "\nJalankan:  pip install -r requirements.txt")
    # 2) Mode screen: cek Tesseract SEKARANG (sebelum memuat model) -> gagal cepat, tidak looping
    if a.mode == "screen":
        enable_dpi_awareness()
        import pytesseract
        ver, where = setup_tesseract(pytesseract, a.tesseract_cmd, a.lang)
        print(f"Tesseract {ver} siap: {where}")

    det = ToxicDetector(a.model_dir)
    lock = threading.Lock()
    ui = UI(det, lock)
    signal.signal(signal.SIGINT, lambda *_: ui.request_stop(0))       # Ctrl+C -> berhenti rapi
    {"screen": run_screen, "clipboard": run_clipboard, "typing": run_typing}[a.mode](ui, det, lock, a)
    try:
        ui.root.mainloop()
    except KeyboardInterrupt:
        pass
    print("Berhenti.")
    sys.exit(ui.exit_code)


if __name__ == "__main__":
    main()
