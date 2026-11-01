"""用逻辑回归做一次「合格 / 不良」分类，并打印能看懂的指标。"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlstudy.data import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    TARGET_COLUMN,
    generate_quality_frame,
    load_quality_frame,
)


@dataclass
class TrainResult:
    source: str
    n_rows: int
    n_train: int
    n_test: int
    defect_rate: float
    accuracy: float
    precision_defect: float
    recall_defect: float
    f1_defect: float
    confusion_matrix: object
    coefficients: dict[str, float]
    test_frame: pd.DataFrame


def train_quality_classifier(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainResult:
    """标准化特征后训练逻辑回归。系数对应的是标准化之后的特征。"""
    labels = frame[TARGET_COLUMN]
    if labels.nunique() < 2:
        raise ValueError("defect 只有一种取值，无法训练分类器")

    features = frame.loc[:, FEATURE_COLUMNS]
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(
            "某一类样本太少，无法按比例划分训练/测试。请增加行数，或检查 defect 是否极度不平衡。"
        ) from exc

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probability = pipeline.predict_proba(x_test)[:, 1]

    report = classification_report(
        y_test,
        predicted,
        labels=[0, 1],
        target_names=["合格", "不良"],
        digits=3,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    coefficients = pipeline.named_steps["clf"].coef_.ravel()

    test_frame = x_test.copy()
    test_frame[TARGET_COLUMN] = y_test.to_numpy()
    test_frame["predicted"] = predicted
    test_frame["defect_probability"] = probability

    return TrainResult(
        source="",
        n_rows=len(frame),
        n_train=len(x_train),
        n_test=len(x_test),
        defect_rate=float(labels.mean()),
        accuracy=float(report["accuracy"]),
        precision_defect=float(report["不良"]["precision"]),
        recall_defect=float(report["不良"]["recall"]),
        f1_defect=float(report["不良"]["f1-score"]),
        confusion_matrix=matrix,
        coefficients=dict(zip(FEATURE_COLUMNS, map(float, coefficients))),
        test_frame=test_frame,
    )


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def format_report(result: TrainResult) -> str:
    """把一次训练结果整理成可以直接阅读的文本。"""
    lines = [
        f"数据来源: {result.source}",
        f"总行数 {result.n_rows}，训练 {result.n_train}，测试 {result.n_test}",
        f"整体不良率: {result.defect_rate:.1%}",
        "",
        "测试集指标（1 = 不良）",
        f"  准确率 accuracy : {result.accuracy:.3f}",
        f"  精确率 precision: {result.precision_defect:.3f}"
        "    判成不良的里面，有多少真是不良",
        f"  召回率 recall   : {result.recall_defect:.3f}"
        "    真不良里面，有多少被找出来",
        f"  F1             : {result.f1_defect:.3f}",
        "",
        "混淆矩阵（行 = 真实，列 = 预测）",
        "                预测合格    预测不良",
    ]
    ok_ok, ok_ng = result.confusion_matrix[0]
    ng_ok, ng_ng = result.confusion_matrix[1]
    lines.append(f"  真实合格      {ok_ok:8d}    {ok_ng:8d}")
    lines.append(f"  真实不良      {ng_ok:8d}    {ng_ng:8d}")
    lines.append("")
    if result.recall_defect + 0.1 < result.precision_defect:
        lines.append("不良召回率明显低于精确率：模型更惜判「不良」，漏检会偏多。")
        lines.append("")
    elif result.precision_defect + 0.1 < result.recall_defect:
        lines.append("不良精确率明显低于召回率：模型更敢判「不良」，误报会偏多。")
        lines.append("")
    lines.append("标准化后的逻辑回归系数（正值 → 更倾向判为不良）")
    lines.append("系数只能互相比较大小，不能直接读成「振动每增加 1 mm/s」。")
    ranked = sorted(result.coefficients.items(), key=lambda item: abs(item[1]), reverse=True)
    for name, value in ranked:
        lines.append(f"  {_pad(FEATURE_LABELS[name], 16)} {value:+.3f}")
    lines.append("")
    current_coef = abs(result.coefficients["motor_current_a"])
    if current_coef < 0.2:
        lines.append("电机电流在合成数据里不参与标签，这次它的系数也接近 0。")
    else:
        lines.append(
            "电机电流在合成数据里不参与标签，但这次系数并不接近 0。"
            "样本少或噪声大时，无关列也会被带起来，不要据此认为它有用。"
        )
    lines.append("这不是上线用的质检程序，阈值固定为 0.5，只用来把流程走通。")
    return "\n".join(lines)


def build_frame(csv_path: Path | None, n_samples: int, random_state: int) -> tuple[pd.DataFrame, str]:
    if csv_path is None:
        frame = generate_quality_frame(n_samples=n_samples, random_state=random_state)
        return frame, f"合成产线数据（{n_samples} 行，seed={random_state}）"
    frame = load_quality_frame(csv_path)
    return frame, str(csv_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个产线质量分类示例（合格 / 不良）")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="自己的 CSV 路径。不传则使用合成数据",
    )
    parser.add_argument("--samples", type=int, default=800, help="合成数据的行数，默认 800")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="测试集比例，默认 0.25",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="可选。把测试集的预测结果写到这个 CSV",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.1 <= args.test_size <= 0.5:
        print("test-size 需要在 0.1 和 0.5 之间", file=sys.stderr)
        return 1
    try:
        frame, source = build_frame(args.csv, args.samples, args.seed)
        result = train_quality_classifier(
            frame,
            test_size=args.test_size,
            random_state=args.seed,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    result.source = source
    print(format_report(result))
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        result.test_frame.to_csv(args.save, index=False)
        print(f"\n测试集预测已写入 {args.save}")
    return 0
