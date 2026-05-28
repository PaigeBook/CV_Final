"""Project-wide configuration and path resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    data_root: Path
    zip_path: Path
    output_dir: Path
    image_size: tuple[int, int] = (256, 256)
    gaussian_kernel: tuple[int, int] = (5, 5)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    morph_kernel_size: int = 5
    test_size: float = 0.2
    random_state: int = 42
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (8, 8)
    hog_cells_per_block: tuple[int, int] = (2, 2)

    @classmethod
    def from_cli(
        cls,
        data_root: Path | None = None,
        zip_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> "ProjectConfig":
        project_root = Path(__file__).resolve().parent.parent
        resolved_zip = zip_path or project_root / "Brain_Tumor_Dataset.zip"
        resolved_data_root = data_root or project_root / "data" / "raw" / "Brain_Tumor_Dataset"
        resolved_output_dir = output_dir or project_root / "outputs"
        return cls(
            project_root=project_root,
            data_root=resolved_data_root,
            zip_path=resolved_zip,
            output_dir=resolved_output_dir,
        )
