import shutil
from pathlib import Path
import pandas as pd

# === ajusta si tu estructura difiere:
CSV_PATH   = Path("ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
IMAGES_DIR = Path("ISIC2018_Task3_Training_Input")
OUT_DIR    = Path("data_isic_bin")

MALIGNANT = {"MEL","BCC","AKIEC"}
BENIGN    = {"NV","BKL","DF","VASC"}

def main():
    assert CSV_PATH.exists(), f"CSV not found: {CSV_PATH}"
    assert IMAGES_DIR.exists(), f"Images dir not found: {IMAGES_DIR}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR/"benign").mkdir(exist_ok=True)
    (OUT_DIR/"malignant").mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    # columnas esperadas: image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
    assert "image" in df.columns, "CSV must contain 'image' column"
    for c in MALIGNANT | BENIGN:
        assert c in df.columns, f"CSV must contain column '{c}'"

    def to_bin(row):
        for c in MALIGNANT:
            if int(row[c]) == 1: return "malignant"
        for c in BENIGN:
            if int(row[c]) == 1: return "benign"
        return None

    df["label_bin"] = df.apply(to_bin, axis=1)
    df = df[df["label_bin"].notna()]

    copied, missing = 0, 0
    for _, r in df.iterrows():
        img_id = r["image"]
        label  = r["label_bin"]
        # nombres vienen como ISIC_0024306 => busca ISIC_0024306.jpg
        src = IMAGES_DIR / f"{img_id}.jpg"
        if not src.exists():
            # intentar otras extensiones por si acaso
            found = None
            for ext in (".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                p = IMAGES_DIR / f"{img_id}{ext}"
                if p.exists():
                    found = p; break
            if not found:
                missing += 1
                continue
            src = found
        dst = OUT_DIR / label / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    print(f"Copied: {copied} | Missing: {missing}")
    print(f"Benign: {len(list((OUT_DIR/'benign').glob('*')))} | "
          f"Malignant: {len(list((OUT_DIR/'malignant').glob('*')))}")
    print(f"Done. Output: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()