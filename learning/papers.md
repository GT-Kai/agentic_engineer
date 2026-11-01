# 论文阅读清单

> 状态：⬜ 未读 · 🟨 在读 · ✅ 已读 · 🔁 已复现
> 优先级：P0 必读（面试会问）· P1 选读（扩展视野）

## 怎么读

**三遍法**

1. 第一遍（5 分钟）：标题、摘要、图、结论，回答"它解决什么问题"
2. 第二遍（30 分钟）：方法、实验设置、主结果，回答"怎么做的、比谁好"
3. 第三遍（2 小时+）：公式、实现细节、消融，回答"我能不能复现、哪里可以改"

**读完必须写一句话结论**，填在表格最后一列。写不出来就是没读懂。

精读的论文在 `notes/` 下建 `P-xx-标题.md`，从 [_template.md](notes/_template.md) 复制。

## 清单

### Agent 基础与评测

| 编号 | 论文 | 主题 | 优先级 | 状态 | 一句话结论 |
|---|---|---|---|---|---|
| P-01 | [ReAct](https://arxiv.org/abs/2210.03629) | 推理与行动交替 | P0 | ⬜ | |
| P-02 | [Toolformer](https://arxiv.org/abs/2302.04761) | 自监督学习调用工具 | P1 | ⬜ | |
| P-03 | [Reflexion](https://arxiv.org/abs/2303.11366) | 语言化反思与重试 | P1 | ⬜ | |
| P-04 | [τ-bench](https://arxiv.org/abs/2406.12045) | 工具-用户多轮交互评测 | P0 | ⬜ | |
| P-05 | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | 函数调用能力评测 | P0 | ⬜ | |
| P-06 | [SWE-bench](https://arxiv.org/abs/2310.06770) | 真实仓库修 issue | P0 | ⬜ | |

### Agent 环境

| 编号 | 论文 | 主题 | 优先级 | 状态 | 一句话结论 |
|---|---|---|---|---|---|
| P-07 | [WebArena](https://arxiv.org/abs/2307.13854) | 真实 Web 环境 | P1 | ⬜ | |
| P-08 | [OSWorld](https://arxiv.org/abs/2404.07972) | 桌面操作系统环境 | P1 | ⬜ | |
| P-09 | [AppWorld](https://arxiv.org/abs/2407.18901) | 应用与 API 环境 | P1 | ⬜ | |
| P-10 | [GAIA](https://arxiv.org/abs/2311.12983) | 通用助手基准 | P0 | ⬜ | |
| P-11 | [Terminal-Bench](https://github.com/laude-institute/terminal-bench) | 终端任务基准 | P1 | ⬜ | |

### 后训练基础

| 编号 | 论文 | 主题 | 优先级 | 状态 | 一句话结论 |
|---|---|---|---|---|---|
| P-12 | [InstructGPT](https://arxiv.org/abs/2203.02155) | RLHF 三段式范式 | P0 | ⬜ | |
| P-13 | [LoRA](https://arxiv.org/abs/2106.09685) | 低秩适配 | P0 | ⬜ | |
| P-14 | [QLoRA](https://arxiv.org/abs/2305.14314) | 量化微调 | P1 | ⬜ | |
| P-15 | [DPO](https://arxiv.org/abs/2305.18290) | 直接偏好优化 | P0 | ⬜ | |
| P-16 | [Tulu 3](https://arxiv.org/abs/2411.15124) | 开放后训练全流程 | P0 | ⬜ | |
| P-17 | [Llama 3 Herd](https://arxiv.org/abs/2407.21783) | 工业级后训练细节 | P1 | ⬜ | |
| P-18 | [Qwen3](https://arxiv.org/abs/2505.09388) | 混合推理与后训练 | P0 | ⬜ | |

### RL for Reasoning

| 编号 | 论文 | 主题 | 优先级 | 状态 | 一句话结论 |
|---|---|---|---|---|---|
| P-19 | [PPO](https://arxiv.org/abs/1707.06347) | 策略梯度裁剪 | P0 | ⬜ | |
| P-20 | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | GRPO 出处 | P0 | ⬜ | |
| P-21 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 纯 RL 激发推理 | P0 | ⬜ | |
| P-22 | [ScaleRL](https://arxiv.org/abs/2510.13786) | RL 算力扩展规律 | P1 | ⬜ | |
| P-23 | [ProRL](https://arxiv.org/abs/2505.24864) | 长时 RL 扩展推理边界 | P1 | ⬜ | |
| P-24 | [Absolute Zero](https://arxiv.org/abs/2505.03335) | 自博弈零数据 | P1 | ⬜ | |
| P-25 | [Kimi K2](https://arxiv.org/abs/2507.20534) | Agentic 模型训练报告 | P1 | ⬜ | |

### Agentic RL

| 编号 | 论文 | 主题 | 优先级 | 状态 | 一句话结论 |
|---|---|---|---|---|---|
| P-26 | [Search-R1](https://arxiv.org/abs/2503.09516) | 搜索工具 RL | P0 | ⬜ | |
| P-27 | [ReTool](https://arxiv.org/abs/2504.11536) | 代码工具 RL | P1 | ⬜ | |
| P-28 | [SWE-RL](https://arxiv.org/abs/2502.18449) | 软件进化数据 RL | P1 | ⬜ | |
| P-29 | [RAGEN](https://arxiv.org/abs/2504.20073) | 多轮 agent RL 分析 | P0 | ⬜ | |

### 训练系统

| 编号 | 论文 | 主题 | 优先级 | 状态 | 一句话结论 |
|---|---|---|---|---|---|
| P-30 | [HybridFlow / verl](https://arxiv.org/abs/2409.19256) | RLHF 训练框架 | P0 | ⬜ | |

## 建议顺序

1. 入门：P-01 → P-12 → P-15
2. 建立评测观：P-04 → P-05 → P-06
3. 进入 RL：P-19 → P-20 → P-21
4. 走向 agentic：P-26 → P-29 → P-30
5. 扩展：其余按兴趣

一轮下来大约 6–8 周，每周 3–4 篇。
