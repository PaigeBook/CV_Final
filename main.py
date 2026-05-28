"""Repository root entry point for the brain MRI tumour classification project."""

from __future__ import annotations

from src.main import main as run_classical_pipeline


def main() -> int:
	"""Run the extended classical computer vision pipeline."""

	return run_classical_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
