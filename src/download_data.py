"""Baixa e extrai o dataset vangogh2photo (cerca de 300 MB) para data/vangogh2photo/.

Uso: python -m src.download_data
"""
import urllib.request
import zipfile

from tqdm import tqdm

from src.utils import DATA_DIR, DATASET_DIR, DATASET_URL


def main():
    if DATASET_DIR.exists():
        print(f"Já existe: {DATASET_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "vangogh2photo.zip"
    with tqdm(unit="B", unit_scale=True, desc="vangogh2photo.zip") as bar:
        def progress(blocks, block_size, total):
            bar.total = total
            bar.update(blocks * block_size - bar.n)
        urllib.request.urlretrieve(DATASET_URL, zip_path, reporthook=progress)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()

    for folder in ("trainA", "testA", "trainB", "testB"):
        n = len(list((DATASET_DIR / folder).glob("*.jpg")))
        print(f"{folder}: {n} imagens")


if __name__ == "__main__":
    main()
