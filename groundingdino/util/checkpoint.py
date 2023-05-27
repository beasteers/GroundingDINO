import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../..'))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
CKPT_URL = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'

def ensure_checkpoint(path=None):
    path = path or os.path.join(ROOT_DIR, f'./weights/{os.path.basename(CKPT_URL)}')
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print("No checkpoint found. Downloading...")
        def show_progress(i, size, total):
            print(f'downloading checkpoint to {path}: {i * size / total:.2%}', end="\r")
        
        import urllib.request
        urllib.request.urlretrieve(CKPT_URL, path, show_progress)
    return path
