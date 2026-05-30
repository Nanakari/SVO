# paper_reproduce

`paper_reproduce` 是《实验设计方案.docx》中 SVO（Selective Verification of Objects）方法的实验复现项目。项目只实现代码、配置、运行脚本、README 和结果导出工具；不撰写论文正文，不手工填入或伪造实验结果。

当前状态：七个阶段均已实现：项目结构、Base Caption/POPE 推理、对象抽取、风险评分、GroundingDINO 验证、保守修正、评估和结果导出。后续运行真实实验时仍需准备数据、模型权重和检测器环境。

## 云服务器快速开始

目标环境为 Ubuntu 22.04/24.04 + CUDA 12.x + Python 3.10。仓库提供 Bash + Python CLI，不强制使用 Docker。

```bash
git clone <your-repo-url> paper_reproduce
cd paper_reproduce
bash scripts/setup_cloud.sh --env-name SVO --python 3.10
conda activate SVO
bash scripts/smoke_test.sh
bash scripts/run_all.sh --dry-run
```

真实实验前再准备数据和模型：

```bash
bash scripts/prepare_data.sh --download-coco-required --confirm
bash scripts/prepare_data.sh --prepare-train2017-subset --subset-size 5000 --confirm
bash scripts/download_models.sh --confirm --install-groundingdino
python scripts/check_assets.py --strict
```

验证集阈值调参只使用 COCO train2017_val5000：

```bash
bash scripts/tune_svo_threshold.sh --thresholds "0.5 1.0 1.5 2.0" --gpu 0
```

主实验运行示例：

```bash
bash scripts/run_all.sh \
  --datasets coco_chair,pope \
  --methods base,svo,verify_all,random_verify \
  --risk-threshold <VAL_THRESHOLD> \
  --gpu 0
```

`<VAL_THRESHOLD>` 必须来自验证集调参，不能用测试集结果反推。详见 `docs/reproduction.md`。

扩展消融和敏感性分析：

```bash
# 单风险信号分析：Uncertainty Only / Position Only / Prior Only
bash scripts/run_all.sh --datasets coco_chair --methods components --risk-threshold <VAL_THRESHOLD>

# SVO 风险阈值扫描（只使用 train2017_val5000 验证集）
bash scripts/tune_svo_threshold.sh --thresholds "0.5 1.0 1.5 2.0" --gpu 0

# GroundingDINO 阈值敏感性分析
python scripts/sweep_detector_thresholds.py \
  --objects outputs/validation/objects/coco_train2017_val5000_svo_objects.jsonl \
  --base-predictions outputs/validation/predictions/coco_train2017_val5000_base_captions.jsonl \
  --risk-threshold <VAL_THRESHOLD> \
  --box-thresholds 0.25 0.35 0.45 \
  --text-thresholds 0.20 0.25 0.30 \
  --coco-annotations data/coco/annotations/instances_train2017.json \
  --output-dir outputs/validation/sweeps/detector_thresholds
```

## 方法范围

SVO 是训练免费的选择性后处理式视觉验证方法：

1. 使用 LLaVA-1.5-7B 为图像生成描述。
2. 从描述文本中抽取可见对象短语。
3. 根据生成不确定性、对象位置风险、静态对象幻觉先验计算风险分数。
4. 只对高风险对象调用 GroundingDINO 进行开放词汇视觉验证。
5. 对缺乏视觉证据的对象执行保守删除或弱化修正。

本项目只关注对象存在性幻觉，不处理属性、数量、空间关系或 OCR 幻觉。

## 目录结构

```text
paper_reproduce/
  configs/                 # 全局、数据集、方法和词表配置
  data/                    # 本地数据软链接或解压目录，不纳入版本管理
  docs/                    # 阶段计划、结果 schema 等说明
  models/                  # 本地模型权重或软链接，不纳入版本管理
  outputs/                 # 推理、评估和导出结果，不纳入版本管理
  scripts/                 # 后续阶段补充命令行入口
  src/paper_reproduce/     # 模块化源码包
  examples/smoke/          # 无模型最小烟测样例
  tests/                   # 单元测试
```

源码包按可替换模块划分：`models`、`datasets`、`extraction`、`scoring`、`verification`、`revision`、`methods`、`evaluation` 和 `utils`。

## 环境配置

推荐 Python 3.10+。当前本机已创建专用 conda 环境 `SVO`，路径为 `D:\anaconda3\envs\SVO`。GPU 环境建议先按本机 CUDA 版本安装 PyTorch，再安装项目依赖。

```bash
cd paper_reproduce
conda activate SVO
pip install -U pip
pip install -e .[nlp,eval]
```

需要运行 LLaVA 和 GroundingDINO 时，再安装模型依赖：

```bash
pip install -e .[models]
```

云端一键安装入口：

```bash
bash scripts/setup_cloud.sh --with-models
```

如果使用 spaCy 作为对象短语抽取器，需要额外下载英文模型：

```bash
python -m spacy download en_core_web_sm
```

GroundingDINO 的安装方式会随上游仓库变化，后续实现阶段会在验证模块中保留可替换接口，并在 `models/README.md` 记录接入方式。
第四阶段的验证适配器使用官方 [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 包中 `groundingdino.util.inference` 的 `load_model`、`load_image` 和 `predict` 接口。请先安装官方 GroundingDINO 包，并在 `configs/default.yaml` 中填写 `config_path` 与 `checkpoint_path`。

## 数据准备

本项目不下载或重新发布数据集。请将数据放到 `data/` 下，或使用软链接指向已有数据目录，然后修改 `configs/datasets/*.yaml`。

推荐布局：

```text
data/
  coco/
    train2017/
    train2017_val5000/
    val2014/
    val2017/
    annotations/
  pope/
    random.jsonl
    popular.jsonl
    adversarial.jsonl
  amber/
    images/
    object_subset.jsonl
```

数据集接口目标：

- COCO/CHAIR：读取图像、COCO 标注和生成描述，计算 CHAIRs、CHAIRi、Average Length、Correct Object Coverage、误修正指标。
- POPE：读取 random/popular/adversarial 问答样本，支持 Yes-to-No 修正协议和 Accuracy、Precision、Recall、F1、Yes Ratio。
- AMBER Object Subset：只读取对象存在性相关样本，不纳入属性、关系和 OCR 项。

验证集阈值调参默认使用 COCO train2017 随机 5000 张图像，测试集不得参与调参。

可使用以下脚本创建目录或下载公开 COCO 文件：

```bash
bash scripts/prepare_data.sh
bash scripts/prepare_data.sh --download-coco-required --confirm
bash scripts/prepare_data.sh --prepare-train2017-subset --subset-size 5000 --confirm
```

生成验证 split：

```bash
python scripts/make_val_split.py \
  --coco-annotations data/coco/annotations/instances_train2017.json \
  --sample-size 5000 \
  --seed 42
```

如果你已经下载了完整 `train2017/`，subset 脚本会从完整目录复制 split 中的图像；如果没有完整
`train2017/`，可以加 `--download-missing-subset --confirm` 只下载 split 需要的图片。完整
`train2017/` 仍然兼容，只要同时传入同一个 `split_file` 即可限制验证集样本。

## 模型准备

主实验目标模型：

- LLaVA-1.5-7B：基础 caption 和 POPE Yes/No 回答生成。
- GroundingDINO：高风险对象视觉验证。

配置入口在 `configs/default.yaml`：

- `generation.model_name_or_path`
- `verification.groundingdino.config_path`
- `verification.groundingdino.checkpoint_path`

辅助下载脚本：

```bash
bash scripts/download_models.sh --confirm --install-groundingdino
```

VCD 和 OPERA 当前只作为后续预留方法接口，不在主实现阶段伪造结果。预留接口见 `src/paper_reproduce/methods/registry.py` 和对应 YAML 配置，里面写明 TODO、依赖和接入方式。

## 配置文件

全局默认配置：`configs/default.yaml`

数据集配置：

- `configs/datasets/coco_chair.yaml`
- `configs/datasets/pope.yaml`
- `configs/datasets/amber_object.yaml`

方法配置：

- `configs/methods/base.yaml`
- `configs/methods/verify_all.yaml`
- `configs/methods/random_verify.yaml`
- `configs/methods/svo.yaml`
- `configs/methods/svo_without_uncertainty.yaml`
- `configs/methods/svo_without_position.yaml`
- `configs/methods/svo_without_prior.yaml`
- `configs/methods/ablations.yaml`
- `configs/methods/vcd.yaml`
- `configs/methods/opera.yaml`

后续脚本会同时支持命令行参数和 YAML 配置文件；命令行参数优先覆盖配置文件。

## 计划运行命令

以下是最终项目的命令形态。SVO 流水线拆成多个可审计步骤，便于检查每个中间 JSONL 文件。

```bash
# Base caption generation for COCO/CHAIR
python scripts/run_caption.py --config configs/default.yaml --dataset configs/datasets/coco_chair.yaml --method configs/methods/base.yaml

# POPE inference
python scripts/run_pope.py --config configs/default.yaml --dataset configs/datasets/pope.yaml --method configs/methods/base.yaml

# Object extraction and risk scoring
python scripts/extract_objects.py --config configs/default.yaml --method configs/methods/svo.yaml --input outputs/predictions/coco_chair_base_captions.jsonl

# Static hallucination prior from COCO validation captions
python scripts/build_static_prior.py --config configs/default.yaml --captions outputs/predictions/coco_train_base_captions.jsonl --coco-annotations data/coco/annotations/instances_train2017.json

# GroundingDINO visual verification for SVO-selected objects
python scripts/verify_objects.py --config configs/default.yaml --method configs/methods/svo.yaml --input outputs/objects/coco_chair_base_captions_objects.jsonl --risk-threshold <VAL_THRESHOLD>

# Verify-All baseline with the same detector
python scripts/verify_objects.py --config configs/default.yaml --method configs/methods/verify_all.yaml --input outputs/objects/coco_chair_base_captions_objects.jsonl

# Random-Verify baseline matched to SVO verification counts
python scripts/verify_objects.py --config configs/default.yaml --method configs/methods/random_verify.yaml --input outputs/objects/coco_chair_base_captions_objects.jsonl --reference outputs/verifications/coco_chair_base_captions_objects_svo.jsonl

# Conservative caption revision
python scripts/revise_captions.py --config configs/default.yaml --input outputs/verifications/coco_chair_base_captions_objects_svo.jsonl

# POPE Yes-to-No revision
python scripts/revise_pope.py --config configs/default.yaml --predictions outputs/predictions/pope_base_pope.jsonl --verifications outputs/verifications/pope_base_objects_svo.jsonl

# CHAIR evaluation
python scripts/evaluate.py --config configs/default.yaml --dataset coco_chair --task chair --predictions outputs/revisions/coco_chair_base_captions_objects_svo_revisions.jsonl --text-field revised_caption

# POPE evaluation
python scripts/evaluate.py --config configs/default.yaml --dataset pope --task pope --predictions outputs/revisions/pope_base_pope_pope_revised.jsonl

# AMBER Object Subset evaluation
python scripts/evaluate.py --config configs/default.yaml --dataset amber_object --task amber --predictions outputs/predictions/amber_object_svo.jsonl

# Efficiency evaluation
python scripts/evaluate.py --config configs/default.yaml --dataset coco_chair --task efficiency --objects outputs/objects/coco_chair_base_captions_objects.jsonl --verifications outputs/verifications/coco_chair_base_captions_objects_svo.jsonl --base-predictions outputs/predictions/coco_chair_base_captions.jsonl

# False-correction evaluation
python scripts/evaluate.py --config configs/default.yaml --dataset coco_chair --task false_correction --predictions outputs/revisions/coco_chair_base_captions_objects_svo_revisions.jsonl

# Export tables from real metric files only
python scripts/export_results.py --config configs/default.yaml --metrics-dir outputs/metrics --out outputs/tables
```

## 结果文件格式

所有中间结果采用 JSONL，便于断点续跑和逐样本审计。详细 schema 见 `docs/result_schemas.md`。

核心输出类型：

- captions：图像描述、生成参数、耗时和可选 token 分数。
- objects：对象短语、归一化类别、文本位置和风险分数。
- verifications：被验证对象、检测置信度、是否通过视觉证据阈值。
- revisions：原描述、修正描述、删除或弱化对象、保守修正规则。
- metrics：由真实输出统计得到的 CHAIR、POPE、效率和误修正指标。

实验表格必须由 `outputs/metrics` 中的真实指标自动导出。没有真实输出时，导出工具只能生成空模板或缺失值标记，不能填入虚假数值。导出脚本会生成 Markdown、CSV 和带来源追踪的 JSON 表格。

## 测试与仓库检查

无模型烟测：

```bash
bash scripts/smoke_test.sh
```

单元测试 + 编译检查 + 烟测：

```bash
bash scripts/run_tests.sh
```

云端资产路径检查：

```bash
python scripts/check_assets.py
```

更多说明见：

- `docs/cloud_setup.md`
- `docs/data_preparation.md`
- `docs/model_zoo.md`
- `docs/reproduction.md`
- `docs/experiment_matrix.md`
- `docs/hyperparameters.md`
- `docs/troubleshooting.md`

## 实验方法清单

核心方法：

- Base：原始 LLaVA-1.5-7B，不做幻觉缓解。
- Verify-All：对描述中所有抽取对象进行 GroundingDINO 验证。
- Random-Verify：逐样本随机选择与 SVO 相同数量的对象进行验证。
- Ours/SVO：只验证风险分数超过阈值的对象。

风险评分消融：

- Ours-Full：Uncertainty + Position + Prior。
- w/o Uncertainty：去掉生成不确定性。
- w/o Position：去掉位置风险。
- w/o Prior：去掉静态幻觉先验。
- Uncertainty Only：只保留生成不确定性。
- Position Only：只保留位置风险。
- Prior Only：只保留静态幻觉先验。

预留方法：

- VCD：TODO，接入视觉对比解码实现，保持与 Base 相同 prompt 和生成参数。
- OPERA：TODO，接入 OPERA 解码干预实现，记录环境依赖和复现限制。

## 阶段计划

1. 项目结构和 README。
2. Base Caption 和 POPE 推理代码。
3. 对象抽取和风险评分代码。
4. GroundingDINO 验证代码。
5. SVO 保守修正逻辑。
6. CHAIR / POPE / 效率 / 误修正评估代码。
7. 结果自动导出脚本。

每个阶段完成后暂停检查，再进入下一阶段。
