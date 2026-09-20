# DrugRec-Agent —— 基于图表示学习的可信药物推荐 Agent（原型）

> 一个大创同款方向（GNN × 药物推荐 × LLM Agent × 医学知识图谱）的**端到端可运行原型**：
> 真实医疗数据 → 图表示学习 → 药物推荐 → 带工具、带证据、带安全约束的 LLM Agent。
> **本仓库里每一个数字都来自脚本的真实运行结果**（见 `results/*.json` 与 `results/log_*.txt`），可复现。

---

## 1. 它解决什么问题

给定一位患者的一次住院记录（诊断列表 + 已知用药），系统输出：

1. **候选药物排序**（图神经网络打分，可解释到"哪条诊断支撑了它"）；
2. **安全核对**（与已知用药的真实 DDI 相互作用，Major/Moderate 直接排除或高亮）；
3. **证据链**（历史使用次数、共现诊断次数、相似病例、FDA 说明书原文）；
4. 全部过程由一个 **LLM Agent 工具调用循环**驱动，最终输出可审计的推荐清单。

覆盖招募海报的方向关键词：图表示学习 ✅ 药物推荐 ✅ 医学知识图谱 ✅ RAG 检索 ✅ LLM Agent ✅ MIMIC ✅ 可信/安全 ✅

## 2. 架构

```
┌──────────────────────────── 数据层 (data/) ────────────────────────────┐
│ MIMIC-III demo(真实ICU记录): 100患者/122次住院/266种药/581个诊断         │
│ DDInter 药物相互作用库: 1939种药 / 160,235 对 DDI(含 Major~Minor 分级)   │
│ openFDA 说明书 API(运行时查询)                                          │
└───────────────┬─────────────────────────────────────────┬──────────────┘
                │                                         │
   实验一 (src/gnn.py)                          实验二 (src/rec_model.py)
   DDI 图上的 GCN 链接预测                        MIMIC 异构图上的推荐模型
   1939节点 / 16万药对 / 2层GCN                 药物-诊断 28.8k边 + 共处方 15.3k边
   → test AUC 0.8467, AP 0.7691                 → HeteroRec(轻量图传播) + DDI 安全重排
        │  药物向量热启动(177/266种药)                    │
        └──────────────────┬──────────────────────────────┘
                           ▼
              实验三 (src/agent/) —— LLM Agent 工具调用循环
   ┌───────────────────────────────────────────────────────────────┐
   │ patient_profile       读病历(诊断+已知用药)                     │
   │ recommend_drugs       GNN 打分 top-k(自动排除 DDI 冲突候选)     │
   │ check_ddi             DDInter 两两核对(含严重程度)              │
   │ drug_info             使用统计 + DDI 伙伴数 + openFDA 说明书     │
   │ find_similar_admissions  历史相似病例检索(仅训练集, 防泄漏)      │
   └───────────────────────────────────────────────────────────────┘
                           ▼
        带证据链 + 安全说明 + 需药师复核提示的推荐清单
```

## 3. 数据来源与规模（真实数据）

| 来源 | 内容 | 规模 | 获取方式 | 许可 |
|---|---|---|---|---|
| MIMIC-III Clinical Database Demo v1.4 | 真实 ICU 住院记录（处方/诊断/患者） | 100 患者、129 次住院、10,398 条处方、1,761 条诊断 | GitHub 公开镜像（见 §5 下载命令），官方源为 PhysioNet | ODbL v1.0（MIT-LCP） |
| DDInter 药物相互作用数据库 | 8 个 ATC 大类 CSV，DDI 对 + 严重程度 | 222,383 条原始记录 → 去重 160,235 对 | ddinter.scbdd.com 公开下载 | 学术用途免费（请引用 DDInter 论文） |
| openFDA drug label API | FDA 说明书原文（适应症/警告/相互作用） | 运行时按需查询 | api.fda.gov | 美国政府公开数据 |

**药名规范化**（两库命名差异是真实痛点）：小写化 → 去括号/百分比/剂型词（如 `Morphine Sulfate (Syringe)` → `morphine sulfate`）→ 盐型剥离（`metoprolol tartrate` → `metoprolol`）→ 人工别名表（`aspirin→acetylsalicylic acid`、`albuterol→salbutamol`、`insulin→insulin human`）。
结果：**66.5% 的药物节点 / 80.9% 的处方行**成功对齐 DDInter；未对齐的主要是疫苗、保健品、复方制剂（诚实披露，见 `data/processed/stats.json`）。

## 4. 实验与结果（全部真实运行）

### 实验一：DDI 链接预测（图表示学习核心实验）

任务：判断两种药之间是否存在已知相互作用（药物组合风险预筛）。
设置：1939 个药物节点、160,235 个唯一药对；85% / 7.5% / 7.5% 边划分；邻接图只用训练边（防泄漏）；负采样 1:1 且过滤全部正样本（防假负例）；2 层 GCN（dim=64，dropout=0.2），Adam lr=0.02，早停。

| 指标 | 数值 |
|---|---|
| 验证集 AUC（best epoch 21） | **0.8503** |
| 验证集 AP | 0.7729 |
| **测试集 AUC** | **0.8467** |
| **测试集 AP** | **0.7691** |
| 训练耗时（CPU，Intel Arc 集显机器） | 60–92 秒 |
| 复现性 | 3 次独立运行指标完全一致 |

附加演示：在 MIMIC 常用药之间，「DDInter 已收录相互作用」与「未收录组合」的可分性 **AUC = 0.7893**（即模型不仅记住已收录的，还能对新组合给出有意义的分数排序）。
曲线：`results/ddi_training_curves.png`　指标：`results/ddi_link_pred.json`

> 诚实说明：点积解码器的绝对分数尺度很小（logit ≈ 0.001 级），BCE loss 在 4 位小数下看起来"平"在 0.6931；但 AUC/AP 是秩指标不受影响。下一版可加可学习温度参数让 loss 曲线更直观。

### 实验二：药物推荐 + DDI 安全重排

任务：给定诊断（+ 一半已知用药），为患者排序 266 个候选药物。
设置：按患者切分（88 次训练住院 / 34 次测试住院）；HeteroRec = 药物-诊断二部图 + 共处方图上的两轮图传播（用实验一的向量热启动）；3 个随机种子取均值±标准差。
方法：popular（流行度）/ cond_cooccur（诊断-药物共现）/ gnn / **gnn+ddi_mask**（把与已知用药有 Major/Moderate DDI 的候选直接排除后再排序）。

**场景 half（已知用药=真实用药的一半，目标=另一半；最贴近真实使用）**

| 方法 | Recall@10 | Recall@20 | Precision@10 | Major违规@10 | Major+Moderate违规@10 |
|---|---|---|---|---|---|
| popular | 0.268 | 0.415 | 0.341 | 0.144 | 0.429 |
| cond_cooccur | **0.323** | **0.516** | **0.403** | 0.159 | 0.447 |
| gnn | 0.273 ± 0.008 | 0.418 ± 0.001 | 0.342 ± 0.009 | **0.114 ± 0.011** | 0.478 ± 0.019 |
| **gnn+ddi_mask** | 0.207 ± 0.004 | 0.299 ± 0.005 | 0.253 ± 0.006 | **0.000** | **0.000** |

**场景 full（已知=目标=全部真实用药，宽松上界）**

| 方法 | Recall@10 | Recall@20 | Major违规@10 | Major+Moderate违规@10 |
|---|---|---|---|---|
| popular | 0.257 | 0.405 | 0.241 | 0.618 |
| cond_cooccur | **0.310** | **0.511** | 0.265 | 0.647 |
| gnn | 0.260 ± 0.006 | 0.402 ± 0.008 | **0.211 ± 0.018** | 0.672 ± 0.013 |
| **gnn+ddi_mask** | 0.178 ± 0.001 | 0.231 ± 0.001 | **0.000** | **0.000** |

结论（如实）：在这个 100 患者的小规模 demo 上，GNN 的命中率与基线相近、但**Major 违规率是所有不做安全约束的方法里最低的**；加入安全重排后违规率归零，代价是约 6–7 个百分点的命中率——这正是"可信推荐"里精度与安全的显式权衡点。
图：`results/rec_metrics.png`　指标：`results/rec_metrics.json`　模型：`results/rec_model.pt`

### 实验三：Agent 评测（纯 LLM 基线 vs 带工具 Agent）

设置：同一批真实测试病例；模型只见诊断 + 已知用药（half 场景）；对比"直接问 LLM"与"Agent 走工具流程（读病历→GNN 推荐→核对 DDI→查证据）"。

**5 个测试病例的汇总（均值；耗时约 10 分钟）**

| 方法 | Recall@10 | 命中数/例 | Major违规@10 | Major+Moderate违规@10 | 步数 | 工具调用 | 秒/例 |
|---|---|---|---|---|---|---|---|
| 纯 LLM 基线（无工具） | 0.135 | 1.8 | **1.4** | 4.4 | 1.0 | 0 | 53 |
| **Agent（5 个工具）** | **0.187** | **2.2** | **0.0** | **0.2** | 6.4 | 7.0 | 69 |

逐病例命中（推荐清单前 10 命中隐藏目标数）：100375: 3/9 → **4/9**；102203: 2/14 → 2/14；114648: 0/12 → **1/12**；117105: 4/20 → 2/20；118192: 0/12 → **2/12**。
基线在 3 个病例共产生 **7 处 Major 违规**，Agent **全程 0 处**（工具层的 DDI 安全重排 + 模型主动核对共同保证）。

结论（如实）：Agent 把**安全违规降为零**，命中率总体更高（0.187 vs 0.135；小样本、逐病例有波动）；代价是每例多约 7 次工具调用、多约 16 秒——这是"可信优先"的合理成本。
指标文件：`results/agent_eval.json`；每例完整对话轨迹：`results/agent_logs/`（含每次工具调用的入参和返回）。

轨迹示例（住院 142345，Agent 6 步 11 次工具调用 0 错误）：

> **Acetaminophen** | GNN评分2.336(最高)，相似病例使用率4/5，支持房颤(34例)、充血性心衰(27例) | 肾病V期需监测肝功能
> **风险提示**: 已知用药中 Insulin–Levofloxacin、Potassium Chloride–Moexipril 存在 Major 相互作用，需持续监测血糖/血钾……
> 最后一行: `推荐清单: Acetaminophen, Docusate, Senna, Pantoprazole, Multivitamins, Pneumococcal Vac Polyvalent`

完整轨迹：`results/agent_logs/*.json`

## 5. 快速开始（Windows + uv）

```bash
cd D:/File/OneDrive/Desktop/项目/drug-rec-agent

# 1) 环境(uv 秒装, 依赖全部来自缓存/镜像)
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 2) 数据下载(仓库不含原始数据; 一键脚本, 自动跳过已存在文件, 可重复运行)
.venv/Scripts/python.exe scripts/download_data.py

# 3) 按顺序运行(实测耗时)
.venv/Scripts/python.exe src/prep_data.py            # 建图与工件        ~10s
.venv/Scripts/python.exe src/gnn.py                  # 实验一 DDI 链接预测 ~90s
.venv/Scripts/python.exe src/rec_model.py            # 实验二 推荐+安全重排 ~150s
.venv/Scripts/python.exe src/agent/drug_agent.py 142345   # 单病例 Agent 演示(需 DOTS_API_KEY)
.venv/Scripts/python.exe src/agent/run_cases.py 5    # 实验三 基线 vs Agent 评测
```

> Agent 部分需要任意 OpenAI 兼容的 LLM 接口（其余脚本完全离线）。相关环境变量：`DRUG_AGENT_LLM_BASE_URL`（端点地址）、`DRUG_AGENT_LLM_MODEL`（模型名）、`DRUG_AGENT_LLM_API_KEY`（密钥，也兼容 `DOTS_API_KEY` / `OPENAI_API_KEY`）。

### 在其他电脑 / 服务器上运行（无本机依赖）

- `git clone` 仓库 → 运行 `scripts/download_data.py` → 按上面 1)~3) 步执行即可；**Linux/macOS 请把 `.venv/Scripts/python.exe` 换成 `.venv/bin/python`**。
- 环境要求：Python ≥ 3.10，**纯 CPU 可跑**（无需显卡），约 2GB 磁盘；也可以部署在服务器 / 云主机上长期运行。
- 只有 Agent 一步需要 LLM Key（任何 OpenAI 兼容端点都行，例如本地 vLLM / DeepSeek / OpenAI）；其余脚本完全离线、无需任何密钥。

## 6. PyCharm 里的使用步骤（图形界面）

1. **打开项目**：`File → Open…`，选择文件夹 `D:\File\OneDrive\Desktop\项目\drug-rec-agent`。
2. **配置解释器**：右下角点 `Python Interpreter`（或 `File → Settings → Project → Python Interpreter`）→ `Add Interpreter → Add Local Interpreter…` → `Existing` → 选 `.venv\Scripts\python.exe` → OK。
3. **运行脚本**：在左侧项目树里右键某个脚本（如 `src/gnn.py`）→ `Run 'gnn'`（快捷键 `Ctrl+Shift+F10`）；运行结果显示在底部 Run 面板。
4. **带参数运行**（如指定住院号）：`Run → Edit Configurations…` → 在 `Parameters` 里填 `142345` → Run。
5. **看结果**：`results/` 下的 JSON 与 PNG（PyCharm 里双击 PNG 直接预览）；`data/processed/stats.json` 是全部数据口径。
6. 如果 Agent 脚本报"未找到 DOTS_API_KEY"：在 `Run → Edit Configurations → Environment variables` 里加上 `DOTS_API_KEY=...`（或重启 PyCharm 让它继承系统环境变量）。

## 7. 目录结构

```
drug-rec-agent/
├─ data/
│  ├─ raw/            # 原始数据(mimic_demo/ + ddinter/, 需按 §5 下载)
│  └─ processed/      # 加工后工件: 节点表/边表/DDI表/划分/统计 + emb_ddi.npz
├─ src/
│  ├─ common.py       # 路径/种子/药名规范化
│  ├─ inspect_data.py # 原始数据体检
│  ├─ prep_data.py    # 数据层: 异构图构建
│  ├─ gnn.py          # 实验一: GCN 链接预测 + 药物向量
│  ├─ rec_model.py    # 实验二: HeteroRec 推荐 + 安全重排 + 基线
│  └─ agent/
│     ├─ tools.py       # 5 个工具(病历/GNN推荐/DDI/openFDA/相似病例)
│     ├─ drug_agent.py  # LLM 工具调用循环
│     └─ run_cases.py   # 实验三: 基线 vs Agent 评测
├─ scripts/
│  └─ download_data.py  # 一键下载原始数据(标准库, 跨平台)
├─ results/           # 指标JSON + 曲线图 + 日志 + agent_logs/ 完整轨迹
├─ requirements.txt
├─ LICENSE            # 代码 MIT; 数据许可说明见文件末尾
└─ README.md
```

## 8. 局限与升级路线（写进申请书里的诚实部分）

- **数据规模**：demo 只有 100 患者 / 122 次可用住院。升级到 MIMIC-III/IV 全量（~30 万次住院）只需替换 `data/raw/mimic_demo` 并保持 CSV 结构，代码不变；需要 PhysioNet 认证账号。
- **模型**：GCN 的绝对分数尺度小、能力有限——下一版：可学习温度、更多层、用药时序建模（按 `startdate` 切分"已有/新增"）、异构图注意力。
- **DDI 覆盖**：DDInter 未覆盖保健品/疫苗/复方，代码中已显式标注"unresolved_names 需人工核对"。
- **安全评测口径**：已知用药=真实用药一半的模拟设置；上线前需要前瞻性/回顾性临床验证与药师评审。
- **可解释性增强**：目前证据链是"诊断共现次数+相似病例"，可扩展到药-靶-通路级知识图谱（如 Hetionet/DrugBank）。

## 9. 致谢与引用

- **MIMIC-III Clinical Database Demo v1.4** — Johnson et al., MIT-LCP(ODbL v1.0)
- **DDInter 药物相互作用数据库** — scbdd.com, 请引用其 Nucleic Acids Research 论文
- **openFDA** — 美国 FDA 公开药物说明书数据
- **数据许可**：本仓库不含任何原始数据（MIT 仅覆盖代码）；下载使用时请遵守 MIMIC-III demo（ODbL v1.0）与 DDInter（仅限学术研究）各自的条款。
- 本项目为独立构建的教学/研究原型, 与上述数据提供方无隶属关系。

## 10. 诚信声明

- 本项目是**研究原型**，输出**不构成医疗建议**。
- 所有指标可复核：数字出处在 `results/*.json`；运行日志在 `results/log_*.txt`；数据口径在 `data/processed/stats.json`。
- 小样本实验结果以**趋势与机制**为主，不宣称临床有效性。
