# 机器学习学习项目

GitHub 仓库名是 `-`。这里把它当作机器学习练习场：面向工厂自动化里常见的传感器表，把「读数据 → 训练一个小模型 → 看懂指标」走通，之后继续把笔记和代码推上来。

当前是骨架加一份可运行示例，不是产线质检系统，也没有训练流水线、实验跟踪或模型仓库。

## 用来做什么

示例假设一台工位记下了这些量，并要判断这件产品是合格（0）还是不良（1）：

| 列名 | 含义 |
| --- | --- |
| `temperature_c` | 温度 (°C) |
| `vibration_mm_s` | 振动 (mm/s) |
| `pressure_bar` | 压力 (bar) |
| `cycle_time_s` | 节拍 (s) |
| `motor_current_a` | 电机电流 (A) |
| `defect` | 0 合格，1 不良 |

不放 CSV 时，程序会生成一份带噪声的合成数据。合成规则大致是：温度偏高、振动偏大、节拍偏长更容易不良，压力略高略好。电机电流故意不影响标签，用来观察模型会不会抓住无关列。

## 环境

需要 Python 3.10 或更高。在仓库根目录：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows 上激活虚拟环境用 `.venv\Scripts\activate`。

依赖只有 `numpy`、`pandas`、`scikit-learn`。要打开 `notebooks/` 时再额外安装 Jupyter：

```bash
pip install jupyter
```

也可以把本目录装成可编辑包，之后就能用 `python -m mlstudy`：

```bash
pip install -e .
```

## 目录

```text
src/mlstudy/     造数、训练、打印指标
examples/        直接运行的脚本
notebooks/       可选的 Jupyter 笔记
data/raw/        原始 CSV（不提交真实数据）
data/processed/  清洗结果或预测输出（不提交）
notes/           学习笔记
```

`data/raw/` 和 `data/processed/` 里除了 `.gitkeep` 以外的文件已被忽略。设备导出、MES 表、客户数据不要推进这个仓库。

## 学习路径

1. 装好依赖，跑通下面的示例，确认终端里有准确率、精确率、召回率和混淆矩阵。
2. 读 `examples/train_quality_classifier.py`，再看 `src/mlstudy/data.py` 和 `src/mlstudy/train.py`。先分清哪一段在造数据，哪一段在训练。
3. 改 `--samples` 和 `--seed`，看指标怎么变。样本少的时候，一次结果说明不了太多。
4. 打开 `notes/学习笔记.md`，用自己的话写下：更在意漏检还是误报，哪个特征和不良走在一起。
5. 把自己的表放到 `data/raw/`（列名与上表一致，`defect` 为 0/1），用 `--csv` 再训练一次。先确认标签不是「事后才知道、等于把答案泄给模型」的字段。
6. 想边改边看表格时，再开 `notebooks/01_quality_classifier.ipynb`。

## 怎么跑示例

在仓库根目录：

```bash
python examples/train_quality_classifier.py
```

终端会打印测试集上的准确率、不良类的精确率 / 召回率 / F1、混淆矩阵，以及标准化之后的逻辑回归系数。正系数表示这个特征偏高时，模型更倾向判为不良。系数只能互相比较，不能直接读成「振动每增加 1 mm/s」。

这几个数通常不会一样高。准确率把合格和不良混在一起算；产线上更要看不良那一行的精确率和召回率。输出里如果两者差得比较多，会多一行说明是漏检偏多还是误报偏多。

常用参数：

```bash
python examples/train_quality_classifier.py --samples 200 --seed 7
python examples/train_quality_classifier.py --csv data/raw/quality.csv
python examples/train_quality_classifier.py --save data/processed/quality_scored.csv
```

`--save` 只写出测试集，并带上 `predicted` 和 `defect_probability`。这个文件在 `data/processed/` 下，不会被提交。

装成可编辑包之后，等价入口是：

```bash
python -m mlstudy
python -m mlstudy --csv data/raw/quality.csv
```

## 接下来可以自己加的东西

- 把逻辑回归换成随机森林，看同一份数据上指标差多少。
- 按设备或班次分开看，不要先把所有行混成一个模型。
- 把「不良」写成车间里真正用的定义：尺寸超差、返工，还是人工复判。
- 笔记继续记在 `notes/`，一次实验一段即可。
