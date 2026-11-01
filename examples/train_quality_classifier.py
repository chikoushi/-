"""产线质量分类示例。

在仓库根目录运行：

    python examples/train_quality_classifier.py
    python examples/train_quality_classifier.py --csv data/raw/quality.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlstudy.train import main

if __name__ == "__main__":
    raise SystemExit(main())
