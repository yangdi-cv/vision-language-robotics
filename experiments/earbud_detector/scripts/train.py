"""Train on all 100 images; evaluate on a clearly marked overlapping subset."""
from pathlib import Path
import json
import random
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'datasets/earbud_container/session_20261024_01'
OUT = ROOT / 'experiments/earbud_detector'

def main():
    images = sorted((DATA / 'images').glob('*.jpg'))
    assert len(images) == 100
    for image in images:
        label = DATA / 'labels' / (image.stem + '.txt')
        if not label.exists():
            # Images 75--89 were visually checked as target-free backgrounds.
            assert 75 <= int(image.stem.rsplit('_', 1)[1]) <= 89
            label.touch()
    subset = random.Random(20261024).sample(images, 20)
    (OUT / 'train.txt').write_text(''.join(str(p) + '\n' for p in images))
    (OUT / 'fit_check.txt').write_text(''.join(str(p) + '\n' for p in subset))
    config = OUT / 'data.yaml'
    config.write_text(f'path: {DATA}\ntrain: {OUT / "train.txt"}\nval: {OUT / "fit_check.txt"}\nnames:\n  0: earbud_case\n  1: container\n')
    (OUT / 'split.json').write_text(json.dumps({
        'train': [p.name for p in images],
        'fit_check': [p.name for p in subset],
        'independent_validation': False,
        'note': 'All 20 fit-check images are included in the 100 training images.',
        'seed': 20261024,
    }, indent=2))
    model = YOLO(str(ROOT / 'models/ultralytics/yolo26n.pt'))
    model.train(data=str(config), epochs=100, imgsz=640, batch=16,
                device=0, workers=4, project=str(OUT / 'runs'), name='all100_v1',
                seed=20261024, patience=0, amp=True, close_mosaic=10,
                fliplr=0.5, flipud=0.0, degrees=15, cache=False)
    best = OUT / 'runs/all100_v1/weights/best.pt'
    YOLO(str(best)).predict(source=[str(p) for p in subset], conf=0.25,
                           save=True, project=str(OUT / 'runs'), name='fit_check_preview')

if __name__ == '__main__':
    main()
