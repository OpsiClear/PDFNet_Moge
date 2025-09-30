#!/usr/bin/env python3
"""
PDFNet CLI Entry Point

This is a convenience wrapper for running PDFNet without installation.

Usage:
    python pdfnet.py train --config-file config.yaml
    uv run pdfnet.py infer --input image.jpg --output result.png
    uv run pdfnet.py evaluate --pred-dir results/ --gt-dir DATA/
    uv run pdfnet.py benchmark --checkpoint checkpoints/PDFNet_Best.pth

For full functionality, install the package:
    uv pip install -e .
    # Then use: pdfnet train --config-file config.yaml
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add src to path for development mode (when not installed)
    src_path = Path(__file__).parent / "src"
    if src_path not in [Path(p) for p in sys.path]:
        sys.path.insert(0, str(src_path))
    
    from pdfnet.__main__ import main
    main()
