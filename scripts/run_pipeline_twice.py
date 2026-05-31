import sys
import shutil
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import run_pipeline
from src.config import ProjectConfig

cfg=ProjectConfig.from_cli(data_root=project_root / 'data' / 'tmp_smoke')
cfg = ProjectConfig(
    project_root=cfg.project_root,
    data_root=cfg.data_root,
    zip_path=cfg.zip_path,
    output_dir=cfg.output_dir,
    image_size=cfg.image_size,
    gaussian_kernel=cfg.gaussian_kernel,
    clahe_clip_limit=cfg.clahe_clip_limit,
    clahe_tile_grid_size=cfg.clahe_tile_grid_size,
    morph_kernel_size=cfg.morph_kernel_size,
    test_size=0.5,
    random_state=cfg.random_state,
    hog_orientations=cfg.hog_orientations,
    hog_pixels_per_cell=cfg.hog_pixels_per_cell,
    hog_cells_per_block=cfg.hog_cells_per_block,
)
predictions_dir = cfg.output_dir / 'predictions'
prediction_correct_dir = predictions_dir / 'correct'
prediction_incorrect_dir = predictions_dir / 'incorrect'


def clear_prediction_dirs() -> None:
    for legacy_name in ('cases', 'gallery', 'misclassified', 'reports', 'logs', 'models'):
        shutil.rmtree(cfg.output_dir / legacy_name, ignore_errors=True)
    shutil.rmtree(predictions_dir, ignore_errors=True)
    prediction_correct_dir.mkdir(parents=True, exist_ok=True)
    prediction_incorrect_dir.mkdir(parents=True, exist_ok=True)


clear_prediction_dirs()

res1 = run_pipeline(cfg, sample_limit=3)
pred_root = cfg.output_dir / 'predictions'
files1 = sorted([p.relative_to(pred_root).as_posix() for p in pred_root.rglob('*.png')])
print('Run1 files:', files1)
clear_prediction_dirs()
res2 = run_pipeline(cfg, sample_limit=3)
files2 = sorted([p.relative_to(pred_root).as_posix() for p in pred_root.rglob('*.png')])
print('Run2 files:', files2)
