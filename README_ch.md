# Med-GRADE

[English](README.md) | **中文**

**Med-GRADE** (Medical Grading and Rubric-based Assessment of Doctor-Patient Encounters) 是一个用于评估大语言模型作为**医疗对话量表评审员（LLM-as-Judge）**可靠性的基准。与侧重医学知识问答或通用偏好判别的基准不同，Med-GRADE 关注模型能否在真实医患对话上，依据临床可观察行为对 **23 项 Mini-CEX 改编量表**进行稳定、可解析的二元评分。

<p align="center">
  <img src="figs/fig-intro_01.png" alt="动机示意图" width="65%"/>
</p>


---

## 基准概览

Med-GRADE 包含 **2,076** 条经医师审核的真实在线医患对话，由 **56** 名医师依据 **23 维量表**标注，覆盖：

| 维度 | 缩写 | 条目数 | 评估内容 |
|------|------|--------|----------|
| Medical Interviewing | MI | 1.1–1.8 | 病史采集、开放式提问、通俗表达、解释依据、识别急症等 |
| Humanistic Care | HC | 2.1–2.8 | 尊重与同理、问诊组织、情绪支持、隐私保护等 |
| Diagnosis & Treatment Management | DTM | 3.1–3.7 | 信息主次判断、可信度验证、鉴别诊断、诊疗方案合理性等 |

数据集还附带临床元信息：**18** 个一级科室、**38** 个二级科室、**1,416** 个疾病领域、**9** 类咨询意图。对话平均约 **14.1** 轮、**1,454** 词。

<p align="center">
  <img src="figs/fig-frame_01.png" alt="Figure 2 — Med-GRADE 整体流程" width="85%"/>
</p>


### 任务形式

对每条样本，评审模型输入：

1. **医患对话** `input`
2. **23 维评分量表**（见 `prompt.py`）

输出为长度 **23** 的 JSON 列表，每个元素为 `0`（未做到）或 `1`（做到），顺序对应量表 1.1 → 3.7。

### 评测指标

与论文一致，`eval.py` 计算：

- **Hamming Accuracy**：23 个条目上的逐维一致率
- **Macro-F1**：23 个条目 F1 的宏平均，缓解类别不平衡

同时按三大维度（MI / HC / DTM）汇总分项指标。

<p align="center">
  <img src="figs/fig-radar_01.png" alt="Figure 3 — 不同模型家族的分维度表现" width="85%"/>
  <br>
  <sub><b>不同模型家族的分维度表现</b></sub>
</p>


---

## 仓库结构

```
Med-GRADE/
├── qa.jsonl              # 评测输入（对话 + 元信息，不含 gold label）
├── ground_truth.jsonl    # 医师标注（id + 23 维 output），评测时需要
├── prompt.py             # 23 维量表指令与输出格式后缀
├── test.py               # 调用 LLM 进行批量评审
├── eval.py               # 对照 ground truth 计算指标
├── model_list.py         # 待评测模型 API 配置（需自行创建）
└── output/               # 运行结果目录
    ├── {model}.jsonl
    ├── {model}_token_stats.json
    ├── eval_summary.csv
    └── eval_summary.xlsx
```

### 数据格式

**`qa.jsonl`**（输入，每行一条）：

```json
{
  "id": 0,
  "input": "患者：...\n医生：...",
  "department_level1": "皮肤科",
  "department_level2": null,
  "clinical_domain": "脱发",
  "consultation_intent": "治疗建议"
}
```

**`ground_truth.jsonl`**（标注，每行一条）：

```json
{
  "id": 0,
  "output": ""
}
```

`output` 必须是长度 23 的 `0/1` 列表，顺序与 `prompt.py` 中 `DEFAULT_INSTRUCTION` 一致。

---

## 环境准备

```bash
pip install openai tqdm pandas openpyxl
```

在项目根目录创建 `model_list.py`，例如：

```python
model_list = [
    {
        "save_name": "gemini-3.1-pro",
        "base_url": "https://your-api-endpoint/v1",
        "api_key": "YOUR_API_KEY",
        "model": "gemini-3.1-pro",
        "temperature": 0.7,
        "max_tokens": 1024,
    },
]
```

`test.py` 通过 `save_name` 区分不同模型的输出文件。可按模型需要配置 `disable_thinking`、`merge_system_to_user` 等字段。

---

## 使用方法

### 1. 运行 LLM 评审 — `test.py`

对 `qa.jsonl` 中每条对话调用评审模型，结果写入 `output/{save_name}.jsonl`，支持**断点续传**。

```bash
# 查看可用模型
python test.py --list-models

# 运行全部模型
python test.py

# 指定模型
python test.py -m gemini-3.1-pro gpt-5.4

# 自定义输入文件
python test.py --input qa.jsonl -m your-model
```

**输出示例**（`output/{save_name}.jsonl`）：

```json
{
  "id": 0,
  "instruction": "...",
  "input": "患者：...",
  "output": "",
  "input_tokens": 1234,
  "output_tokens": 56,
  "total_tokens": 1290
}
```

解析失败时会保留 `parse_error` 字段；API 失败时保留 `error` 字段，断点续跑时会自动重试失败样本。

### 2. 计算评测指标 — `eval.py`

将 `output/*.jsonl` 与 `ground_truth.jsonl` 对照，计算 Macro-F1 与 Hamming Accuracy。

```bash
# 评估 output/ 下全部模型
python eval.py

# 指定 ground truth 与模型
python eval.py --gt ground_truth.jsonl --output-dir output -m gemini-3.1-pro gpt-5.4

# 自定义输出路径
python eval.py -o output/eval_summary.csv --xlsx output/eval_summary.xlsx
```

**输出文件：**

- `output/eval_summary.csv` — 各模型总体指标
- `output/eval_summary.xlsx` — 含 `Summary` 与 `By Category`（MI/HC/DTM 分项）两个 sheet

解析失败的预测在指标计算中按**全错**处理（每个维度计为预测错误）。

### 3. 完整流程示例

```bash
# Step 1: 运行评审
python test.py -m gemini-3.1-pro

# Step 2: 计算指标
python eval.py -m gemini-3.1-pro
```

---

## 主要发现（Insights）

基于论文对 17 个 LLM、4 种提示策略的实验，Med-GRADE 揭示以下关键结论：


1. **整体可靠性仅属中等。** 最强模型 Gemini-3.1-Pro 的 Overall Macro-F1 为 **64.89%**，Hamming Accuracy 为 **68.28%**，距离可放心用于临床流程质检仍有明显差距。

2. **DTM 是最难维度。** 诊疗管理能力（鉴别诊断、方案合理性、依据解释）显著弱于 MI 与 HC；医学专用模型在 DTM 上平均 Macro-F1 仅约 **32.12%**，且 Hamming 与 Macro-F1 差距最大，条目级一致率可能**高估**实际可靠性。

3. **医学专用模型并不天然更强。** 部分医学模型在 HC 上表现尚可，但在需要临床推理的 DTM 条目上明显落后；问诊类 QA 微调并不等同于量表校准的评审能力。

4. **更复杂的提示不一定更好。** Chain-of-Thought 可能导致结构化输出崩溃（如 Gemini 有效解析率从 99.95% 降至 19.36%），或显著降低 Macro-F1；Few-shot / Self-refine 对某些条目有提升，但会扰动其他条目的评分边界。

---

## 局限性（Limitations）

### 基准层面（论文）

- **仅覆盖文本对话**，不包含体格检查、化验、影像等多模态临床信息。
- **数据来自特定语言与医疗体系**，跨语言、跨机构泛化需进一步验证。
- **模型与提示策略快速演进**，当前 17 个模型的结论不一定适用于未来模型。

---

