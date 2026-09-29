# Toxic Comment Detector (PyTorch) + "Grammarly" untuk toksisitas

Deteksi 6 label: toxic, severe_toxic, obscene, threat, insult, identity_hate.
Model sudah dilatih (folder `model/`). Hasil validasi: ROC-AUC rata-rata 0,983.

## Instalasi
    pip install -r requirements.txt
Untuk `guard.py screen` pasang juga program Tesseract OCR
(Windows: installer UB Mannheim; Ubuntu: `sudo apt install tesseract-ocr`; macOS: `brew install tesseract`).
Di Linux, tkinter: `sudo apt install python3-tk`.

## Pakai model
    python cli.py predict "you are an idiot"
    python cli.py chat                        # interaktif, ketik /ajari toxic,insult untuk koreksi
    python cli.py teach "kalimat" --labels toxic insult
    python cli.py teach "kalimat" --labels none      # koreksi: BUKAN toxic
    python cli.py teachfile data_saya.csv     # ajari dari CSV (comment_text + kolom label 0/1)
    python cli.py newwords                    # kata baru yang terdeteksi

## Aplikasi pemantau
    python guard.py screen --region 0,0,900,800   # baca layar via OCR (Windows: kotak merah overlay)
    python guard.py clipboard                     # cek teks yang Anda copy
    python guard.py typing                        # cek ketikan; F9 = jeda (JEDA saat mengetik password!)
Windows: tambahkan --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" bila perlu.
Semua diproses lokal. Yang tersimpan hanya koreksi Anda (model/feedback.jsonl) dan daftar kata baru.

## Script dan data di folder berbeda
Script tidak harus satu folder dengan zip/data Anda. Aturannya:
- `train.py --data-dir <path>`: `<path>` boleh FOLDER berisi train.csv atau langsung FILE .zip (tidak perlu diekstrak),
  di drive/folder mana pun. Contoh Windows:
      python train.py --data-dir "D:\\Unduhan\\iiitb-toxic-comment.zip"
  Tanpa `--data-dir`, dicari otomatis di folder kerja, `./data`, dan folder script.
- Model default disimpan/dibaca di folder `model/` DI SAMPING script (bukan di folder tempat Anda mengetik perintah),
  jadi `python C:\\proyek\\guard.py screen` bisa dijalankan dari folder mana saja.
  Untuk lokasi lain: `--model-dir <folder>` atau variabel lingkungan `TOXIC_MODEL_DIR`.
- File .py harus tetap berdampingan (terutama `toxic_core.py`, yang di-import oleh yang lain).
  Kalau ingin memindahkan, pindahkan seluruh folder proyek.
- Path yang mengandung spasi harus diapit tanda kutip. `~` (home) didukung.

## Masalah umum: `tesseract is not installed or it's not in your PATH`
`pytesseract` hanya penghubung Python; program **Tesseract OCR** harus dipasang terpisah.
1. Windows: `winget install --id UB-Mannheim.TesseractOCR -e`
   atau unduh installer dari https://github.com/UB-Mannheim/tesseract/wiki
   (pasang di folder yang disarankan installer: `C:\Program Files\Tesseract-OCR`).
2. Tutup lalu buka PowerShell BARU (agar PATH terbaca), cek: `tesseract --version`
3. `guard.py` juga mencari otomatis di `C:\Program Files\Tesseract-OCR`. Jika tetap gagal:
   `python guard.py screen --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"`
Jika Tesseract tidak ada, program berhenti dengan panduan (tidak lagi berulang tanpa henti);
bila OCR gagal 3x berturut-turut di tengah jalan, program juga berhenti sendiri.

## Menghentikan program
- Tekan **Ctrl+C** di jendela PowerShell tempat program berjalan.
- Jika tidak merespons: tutup jendela PowerShell, atau dari PowerShell lain:
      Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*guard.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
- Saran: gunakan `--region x,y,lebar,tinggi` (hanya area chat) dan jangan menaruh jendela terminal
  di area yang dipantau. Versi ini menutup jendela terminal & notifikasinya sendiri saat memindai (Windows).

## Latih ulang
    mkdir data && letakkan train.csv & test.csv di sana
    python train.py --data-dir data --epochs 4 --out model
    python train.py --extra-csv toxic_indonesia.csv ...   # campur dataset bahasa lain
GPU (mis. Google Colab) otomatis dipakai bila tersedia. Di CPU 1 core: ~16 menit.

## Batasan yang perlu diketahui
- Data latih bahasa Inggris: bahasa Indonesia BELUM terdeteksi sampai dilatih dengan data Indonesia.
  Koreksi satu-satu hanya membuat model menghafal kalimat itu; gunakan --extra-csv atau teachfile
  dengan ratusan-ribuan contoh.
- F1 label langka masih rendah (threat 0,44; identity_hate 0,49; severe_toxic 0,54).
- Overlay kotak merah hanya Windows dan belum diuji di mesin nyata; mode typing/clipboard belum diuji.
  Mode screen (OCR + notifikasi) sudah diuji di layar virtual Linux.
- Model bisa keliru pada sarkasme, kutipan, atau diskusi tentang kata kasar; gunakan sebagai peringatan, bukan hukuman otomatis.
