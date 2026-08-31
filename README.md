# LLM4REC 

快手探索者 LLM-Rec 挑战赛方案代码。我们基于官方 `OneReason-0.8B` 竞赛模型，围绕层级语义 ID（SID）的生成特点改造训练数据与监督目标，最终取得 **1.0357** 分，排名 **83 / 1205（Top 6.9%）**。

- 比赛主页：[快手探索者 LLM-Rec 挑战赛](https://ks-llmrec.streamlake.com/)
- 官方资源：[Explorer_LLM_Rec_Competition](https://huggingface.co/datasets/OpenOneRec/Explorer_LLM_Rec_Competition)

## 方法概览

推荐结果由三层 SID 自回归生成。前级 Token 预测错误后，后续层即使局部正确也无法形成正确物料路径；同时，较长的 CoT 会降低关键 SID 监督在输出序列中的占比。我们从数据和损失两侧处理这一问题：

1. **层级 SID 训练数据**：在完整 SID 生成之外，构造“直接预测第一层、给定第一层预测第二层、给定前两层预测第三层”等条件预测子任务，使模型显式学习每一级 SID 及其层间依赖。
2. **前缀感知 Token 加权**：保留所有普通 SFT Token 的交叉熵，并按 SID 层级提高前级 Token 的损失权重，强化正确前缀在逐层解码中的稳定性。
3. **紧凑语义 CoT**：从用户历史对应的 Caption/Tag 中提取高频兴趣与关键行为证据，压缩冗余推理文本，在严格保留原始 gold SID 的同时提高单位上下文内的有效监督密度。

最终综合得分由初始基线 **0.85** 提升至 **1.0357**，相对提升约 **21.8%**。

## 目录

```text
configs/
  sid_weighted_lora.yaml       # LoRA、32K packing 与分层权重配置
scripts/
  build_sid_curriculum.py      # 构造完整/分层 SID 子任务
  build_compact_cot.py         # 构造可审计的紧凑语义 CoT
  train_sid_weighted.py        # 接入 LLaMA-Factory 的训练入口
  evaluate_sid_beam.py         # 官方风格的短 SID Beam 评测
src/llm4rec/
  sid_weighted_loss.py         # 分层 Token 加权损失
tests/
  test_sid_weighted_loss.py
```

## 快速开始

准备官方模型、SFT 数据以及 Caption/Tag、PID-to-SID 等资源。出于竞赛数据许可与体积考虑，本仓库不分发数据和模型权重。

### 1. 安装

先安装与 GPU 环境匹配的 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，再安装本项目：

```bash
pip install -e ".[train,test]"
```

### 2. 构造层级 SID 数据

```bash
python scripts/build_sid_curriculum.py \
  --data-root /path/to/official/resources \
  --output-dir data/material_sid_curriculum
```

每个高置信物料会生成完整 SID、SID-only 和三个层级条件预测任务；脚本同时执行 Caption 质量过滤、PID-to-SID 精确关联、重复项清理及 SID 前缀分布约束。

### 3. 构造紧凑 CoT

```bash
python scripts/build_compact_cot.py \
  --parquet /path/to/official_caption_tag.parquet \
  --output-dir data/compact_cot
```

CoT 仅使用用户输入中已经出现的历史 SID 所对应的 Caption/Tag，不读取目标物料语义，不在思考文本中泄漏 SID。脚本默认只生成压缩 CoT；`--emit-no-think` 可用于独立消融实验，并非最终成绩所依赖的设置。

### 4. 训练

在 LLaMA-Factory 的 `dataset_info.json` 中注册数据集，并修改 [训练配置](configs/sid_weighted_lora.yaml) 中的本地路径：

```bash
python scripts/train_sid_weighted.py configs/sid_weighted_lora.yaml
```

配置保留了最终方案的核心训练范式：LoRA、长上下文 packing、一个 epoch，以及前重后轻的 SID Token 加权。损失实现不会裁掉普通语言、CoT 或其他任务的监督 Token。

### 5. 小样本评测

```bash
python scripts/evaluate_sid_beam.py \
  --base /path/to/OneReason-0.8B \
  --adapter /path/to/adapter \
  --material /path/to/material_dev.jsonl \
  --rec /path/to/recommendation_dev.jsonl \
  --material-video-only
```

该脚本按 domain prompt 续写三个 SID Token，并分别检查物料理解的 Beam-64 召回，以及推荐任务 no-think / two-stage Beam-32 的候选命中情况。它用于本地趋势判断。


