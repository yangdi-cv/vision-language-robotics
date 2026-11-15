"""Open the two-class wrist-camera dataset in LabelImg (YOLO format)."""
import sys
from pathlib import Path
from labelImg.labelImg import get_main_app, FORMAT_YOLO
root = Path(__file__).resolve().parents[1]
data = root / 'datasets/earbud_container/session_20261024_01'
app, window = get_main_app([sys.argv[0], str(data / 'images'), str(data / 'classes.txt'), str(data / 'labels')])
window.set_format(FORMAT_YOLO)
sys.exit(app.exec_())
