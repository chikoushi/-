# 数据目录

真实的产线数据、导出表、客户数据都不要提交到这个仓库。

- `raw/`：原始 CSV，例如从设备或 MES 导出的表。
- `processed/`：清洗后的表，或示例脚本用 `--save` 写出的预测结果。

这两个目录里的文件已被 `.gitignore` 忽略，只保留 `.gitkeep` 以便空目录能进版本库。

自己的表示例列名：

```text
temperature_c,vibration_mm_s,pressure_bar,cycle_time_s,motor_current_a,defect
```

`defect` 用 `0` 表示合格，`1` 表示不良。
