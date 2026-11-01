"""产线质量数据：合成一份工位传感器表，或读取自己的 CSV。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "cycle_time_s",
    "motor_current_a",
]

TARGET_COLUMN = "defect"

# 打印指标时用的中文名。键的顺序与 FEATURE_COLUMNS 一致。
FEATURE_LABELS = {
    "temperature_c": "温度 (°C)",
    "vibration_mm_s": "振动 (mm/s)",
    "pressure_bar": "压力 (bar)",
    "cycle_time_s": "节拍 (s)",
    "motor_current_a": "电机电流 (A)",
}


def generate_quality_frame(n_samples: int = 800, random_state: int = 42) -> pd.DataFrame:
    """生成一份看起来像工位传感器记录的表格。

    标签带噪声，大致规律是：温度偏高、振动偏大、节拍偏长时更容易不良；
    压力略高略好。电机电流故意不参与标签，用来对照「无关特征」。
    """
    if n_samples < 40:
        raise ValueError("样本数至少 40，否则划分训练/测试后某一类可能不够")

    rng = np.random.default_rng(random_state)
    temperature = rng.normal(70.0, 8.0, n_samples)
    vibration = rng.normal(2.5, 0.6, n_samples)
    pressure = rng.normal(5.0, 0.8, n_samples)
    cycle_time = rng.normal(12.0, 1.5, n_samples)
    motor_current = rng.normal(8.0, 1.2, n_samples)

    logit = (
        -0.8
        + 0.14 * (temperature - 70.0)
        + 2.24 * (vibration - 2.5)
        - 0.63 * (pressure - 5.0)
        + 0.77 * (cycle_time - 12.0)
        + rng.normal(0.0, 0.50, n_samples)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    defect = (rng.random(n_samples) < probability).astype(int)

    return pd.DataFrame(
        {
            "temperature_c": temperature,
            "vibration_mm_s": vibration,
            "pressure_bar": pressure,
            "cycle_time_s": cycle_time,
            "motor_current_a": motor_current,
            "defect": defect,
        }
    )


def load_quality_frame(csv_path: Path) -> pd.DataFrame:
    """读取 CSV。必须包含特征列和 defect（0/1）。"""
    path = Path(csv_path)
    if not path.is_file():
        raise ValueError(f"找不到文件: {path}")

    frame = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "CSV 缺少列: "
            + ", ".join(missing)
            + "。需要: "
            + ", ".join(required)
        )

    out = frame.loc[:, required].copy()
    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    if out[TARGET_COLUMN].isna().any():
        raise ValueError("defect 列含有无法转成数字的值，只能是 0 或 1")
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)
    if not set(out[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("defect 列只能是 0 或 1")
    if out[TARGET_COLUMN].nunique() < 2:
        raise ValueError("defect 只有一种取值，无法训练分类器")
    if len(out) < 40:
        raise ValueError("至少需要 40 行，否则划分训练/测试后某一类可能不够")
    return out
