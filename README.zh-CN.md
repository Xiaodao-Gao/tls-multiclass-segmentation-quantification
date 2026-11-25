# 多尺度融合与单步特征重建的三级淋巴结构分割代码

本仓库按最后专利选定的方法整理，适用于论文投稿时提供代码、配置与复现说明。**多尺度输入全部来自编码器四级特征**，分割结果统一由原生 U-Net 输出头产生。

模型路径为：

```text
224×224 RGB 图块 → EfficientNet-B3 编码器
四级原始特征：32×56×56、48×28×28、136×14×14、384×7×7
通道适配：64、128、256、512（分别保持各级空间大小）
上下文生成、金字塔融合、全局引导 → 32×56×56 融合特征
单步噪声预测与特征重建 → 1×1卷积32→16 → 双线性上采样至224×224
经 tanh 逐通道门控残差加到原生末端16×224×224解码特征
原生四类输出头 → 四类像素概率/标签
```

这是将已有融合和 DiT 组件接成上述路径的新实现。旧代码中“三层解码器特征＋最深编码器特征”的实验成绩，以及“四处编码器跳接注入”的权重，不能作为本实现的验证结果。本包没有真实患者数据、训练权重或论文结果数值。

## 安装与试运行

进入本目录，使用 Python 3.10–3.12。本次在 macOS ARM64、Python 3.12、CPU 上检查。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/make_synthetic_data.py --output outputs/synthetic
tlsseg train --config configs/patent.yaml --manifest outputs/synthetic/manifest.csv --output outputs/smoke --epochs 1 --device cpu --max-train-batches 2 --max-val-batches 1
tlsseg evaluate --checkpoint outputs/smoke/best.pt --manifest outputs/synthetic/manifest.csv --split test --output outputs/smoke/test.json --device cpu
tlsseg predict --checkpoint outputs/smoke/best.pt --manifest outputs/synthetic/manifest.csv --split test --output outputs/predictions --device cpu
tlsseg stitch --layout examples/stitch_layout.json --output outputs/stitched
```

这些图像完全人工生成，用于检查训练、保存、回载、推理和面积计算是否接通，不表示真实病理精度。`outputs/`、`data/`、`weights/`、`runs/` 均不进入 Git 仓库。`requirements-tested.txt` 记录本次实际安装版本，不冒充原历史实验环境。

## 使用自己的数据

准备224×224 RGB图块与同尺寸单通道索引掩膜。类别为0背景、1淋巴细胞聚集型、2初级滤泡样、3次级滤泡样；255忽略。CSV字段为 `patient_id,split,image_path,mask_path,tile_id`，其中同一患者必须只属于train、val、test中的一个集合。相对路径以清单所在目录为起点。预测可以不提供mask。详见 [数据规范](docs/DATA_FORMAT.md)。

```bash
tlsseg train --config configs/patent.yaml --manifest /数据目录/manifest.csv --output runs/patent_seed42 --device cuda
tlsseg evaluate --checkpoint runs/patent_seed42/best.pt --manifest /数据目录/manifest.csv --split test --output runs/patent_seed42/final_test.json --device cuda
tlsseg predict --checkpoint runs/patent_seed42/best.pt --manifest /数据目录/manifest.csv --split test --output outputs/test_predictions --device cuda
```

训练只读取训练和验证图块，根据验证集全局前景宏平均Dice选权重。测试集由单独命令评估。指标先累计全部有效像素混淆矩阵，再计算各类及前景宏平均；这与“逐图块Dice再平均”是不同定义。没有分母的指标写为null并从对应宏平均中排除。默认参数、恢复训练与可选本地原生U-Net初始化见 [英文README](README.md)。

训练时每个样本独立随机抽取索引 `t∈{0,…,49}`，重新生成与当前融合特征同形状的标准高斯噪声，以该t的系数组合出带噪特征，预测目标就是本次生成的这份噪声。推理时不抽噪声，固定t=0进行一次特征重建。第0项仍有0.01噪声系数，因此训练第0项与无噪声推理存在轻微分布差异；本实现保留这一约定，尚需真实数据消融验证。它并不执行从纯噪声到图像的多步生成。公式和尺寸见 [方法说明](docs/METHOD.md)。

整片工具接受已预测的四类概率图、输出网格坐标和可选组织掩膜，重叠区域先融合概率再argmax。面积在唯一网格上统计；只有基础层横纵微米/像素与输出网格到基础层横纵倍率四项齐全时才报告平方毫米。它不包含整片文件读取、标注栅格化及扫描仪校准；NumPy实现适用于能放入内存的网格，超大整片需外部按块处理。

投稿前请按 [复现说明](docs/REPRODUCIBILITY.md) 用实际数据重训该架构，补齐实验结果、数据获取说明及稿件引用信息。现有源码未附完整上游许可，本包保留来源，不擅自赋予MIT等开放授权，见 [来源说明](docs/PROVENANCE.md) 和 [许可状态](LICENSE)。
