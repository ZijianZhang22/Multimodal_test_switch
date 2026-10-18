"""Download a small reproducible v1.1 cohort; never silently drop failed images."""
import argparse
import hashlib
import json
from pathlib import Path

from core import normalize, partition, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='outputs/smoke')
    p.add_argument('--source-split', choices=['test_small', 'test_large'], default='test_small')
    p.add_argument('--revision', default='main')
    p.add_argument('--n-val', type=int, default=50)
    p.add_argument('--n-test', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-root', type=Path, help='Optional local WorldCuisines v1.0 image archive root')
    a = p.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from PIL import Image
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    root = Path(a.data)
    if (root / 'samples.jsonl').exists():
        raise FileExistsError('Prepared data exists; reuse it or choose a NEW --data directory')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'images').mkdir(exist_ok=True)
    repo = 'worldcuisines/vqa-v1.1'
    revision = HfApi().dataset_info(repo, revision=a.revision).sha
    ds = load_dataset(repo, 'task1', split=a.source_split, revision=revision)
    rows = [r for source in ds if (r := normalize(source)) is not None]
    samples = partition(rows, a.n_val, a.n_test, a.seed)
    session = requests.Session()
    session.headers['User-Agent'] = 'WorldCuisinesFoodSteering/0.1 (academic reproduction)'
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    for i, row in enumerate(samples):
        dest = root / 'images' / (hashlib.sha256(row['image_url'].encode()).hexdigest() + '.jpg')
        if not dest.exists():
            if a.image_root:
                src = (a.image_root / row['source_image_path']).resolve()
                if not src.is_relative_to(a.image_root.resolve()):
                    raise ValueError('Unsafe source image path')
                with Image.open(src) as im:
                    im.convert('RGB').save(dest)
            else:
                import io
                response = session.get(row['image_url'], timeout=60)
                response.raise_for_status()
                with Image.open(io.BytesIO(response.content)) as im:
                    im.convert('RGB').save(dest)
        with Image.open(dest) as im:
            im.verify()
        row['image'] = str(dest.relative_to(root))
        row['image_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(f'Image {i + 1}/{len(samples)}', flush=True)
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in samples)
    write_json(root / 'dataset_manifest.json', dict(dataset=repo, revision=revision,
        source_split=a.source_split, language='en', task='task1', prompt_type=1,
        n_val=a.n_val, n_test=a.n_test, seed=a.seed,
        samples_sha256=hashlib.sha256(text.encode()).hexdigest(),
        note='Local food-disjoint partition of official evaluation data, not official benchmark score.'))
    (root / 'samples.jsonl').write_text(text)


if __name__ == '__main__':
    main()
