# 学习路线：Agent 工程 + 后训练

> v1.0 · 2026-11-01 · 配套记录：[log.md](log.md) · [papers.md](papers.md) · [experiments.md](experiments.md)

## 0. 一句话结论

**Agent 工程和后训练不是两个方向，而是同一条流水线的两端。** Agent 工程负责把模型放进真实环境、定义工具与反馈信号；后训练负责用这些轨迹反过来改模型。市面上最缺的，是能把这两端接起来的人：会造环境、会设计奖励、能跑通 RL、能做评测。

所以路线是：**以 Agent 工程入门，以后训练立身，用 Agentic RL 项目把两者焊死。**

## 1. 岗位地图

| 路线 | 典型岗位 | 日常工作 | 入门门槛 | 岗位量 | 适合谁 |
|---|---|---|---|---|---|
| A. Agent 应用工程 | AI Engineer / Agent Engineer / LLM Engineer | 工具调用、上下文工程、多步编排、RAG、评测与可观测性 | 低 | 多 | 后端、全栈转型，想快速上岸 |
| B. 后训练算法 | Post-training Engineer / Applied Research Scientist / RL Engineer | SFT、DPO、RLHF/GRPO、奖励建模、数据配比 | 高 | 中 | 有 ML 基础、能读论文、有卡 |
| C. Agentic RL 与训练 Infra | RL Infra Engineer / Training Infra / Research Engineer | 环境构建、多轮 rollout、异步 RL、推理训练协同、吞吐优化 | 中高 | 少但增长快 | 系统、分布式背景，工程强 |

三个判断：

- A 路线 3–4 个月可以够到初级岗位，是快速上岸的保底选项。
- B 路线通常要求硕士/博士学历，或很强的开源与复现记录。
- C 路线**学历要求最松、最看工程能力**，也是 A 和 B 的交集，建议作为主攻方向。

## 2. 时间与投入

| 模式 | 每周投入 | 预计周期 | 说明 |
|---|---|---|---|
| 全职学习 | 35–45h | 6–8 个月 | 按本路线完整走完 |
| 在职/实习 | 12–18h | 12–16 个月 | 砍掉阶段 1 的部分深度，项目不减 |
| 在校学生 | 20–25h | 10–12 个月 | 可结合课程与实验室算力 |

节奏建议：**70% 时间写代码、跑实验，30% 读论文与文档。** 每个阶段必须有可运行的产物，不接受"看完了"。

## 3. 总览

| 阶段 | 主题 | 周期 | 关键产出 |
|---|---|---|---|
| 0 | 校准 | 2–4 周 | 手写 nanoGPT + 一次 LoRA 微调 |
| 1 | 推理与服务 | 3–4 周 | 带 prefix caching 的 agent 服务 + 压测报告 |
| 2 | Agent 工程 | 6–8 周 | 裸写 agent harness + 评测集 |
| 3 | 后训练基础 | 6–8 周 | SFT→DPO→GRPO 三连 + 消融 |
| 4 | Agentic RL | 8–10 周 | 自建环境 + verl 多轮 RL + 提升曲线 |
| 5 | 训练 Infra | 与 4 并行 | 一次可量化的吞吐优化 |
| 6 | 求职 | 第 20 周起 | 作品集、博客、开源 PR |

## 4. 阶段详情

### 阶段 0 · 校准（2–4 周）

**目标**：具备改代码、debug 训练的手感，而不是重学一遍 ML 理论。

**详细任务**：[stages/stage-0-calibration.md](stages/stage-0-calibration.md) —— 6 个模块（M0–M5）、出口标准 E1–E6、周计划与削减策略

**学什么**

- Python 工程：typing、dataclass、asyncio、pytest、logging、Docker、git 工作流
- PyTorch：Dataset/DataLoader、autocast、梯度累积、checkpoint、显存估算
- Transformer：attention、位置编码、LayerNorm、残差；通读 nanoGPT 全部代码
- 训练基础：cross-entropy、学习率调度、warmup、过拟合与欠拟合判断

**动手**

1. 跟完 Karpathy Zero to Hero 前四讲，手写 micrograd 与 makemore
2. 跑通 nanoGPT 在 shakespeare 上的训练，改一次超参观察曲线
3. 用 transformers + peft 对 1.5B 模型做一次 LoRA SFT

**验收**

- [ ] 能说出 loss 不降的五种原因，并给出排查顺序
- [ ] 能在 24GB 卡上估算 7B 模型 LoRA 训练的显存占用
- [ ] 有一次完整训练 run 的日志与曲线

**资源**：Karpathy Zero to Hero、CS336 前两讲、nanoGPT

---

### 阶段 1 · 推理与服务（3–4 周）

**目标**：建立"每个 token 花在哪"的直觉。

**学什么**

- tokenizer、chat template、特殊 token
- prefill 与 decode、KV cache、continuous batching、PagedAttention
- 采样参数（temperature / top-p / repetition penalty）、结构化输出（JSON schema、grammar）
- vLLM / SGLang 部署、prefix caching、量化（AWQ / GPTQ / FP8）
- 成本模型：每百万 token 价格、并发与 P95 延迟

**动手**

1. 用 vLLM 部署 Qwen3-4B，压测不同并发下的 TTFT / TPOT
2. 构造 agent 典型负载（4k system prompt + 20 轮短对话），对比开/关 prefix caching
3. 写一个 200 行的最小推理服务，支持流式输出与工具调用解析

**验收**

- [ ] 能画出一条 agent 请求的耗时分解图（排队、prefill、decode、工具执行）
- [ ] 能算出某个 agent 产品的单任务成本
- [ ] 能解释为什么 agent 负载下 prefix caching 直接决定毛利

**资源**：vLLM 文档、SGLang 文档、CS336 第 10–11 讲

---

### 阶段 2 · Agent 工程（6–8 周）

**目标**：不依赖框架从零写出可靠 agent，并能量化它的好坏。

**学什么**

- 工具调用协议：function calling、JSON schema、并行调用、错误返回
- 循环设计：ReAct、Plan-Execute、reflection、sub-agent
- 上下文工程：system prompt、消息裁剪、压缩摘要、检索、记忆分层
- 工程：沙箱执行、超时、重试与幂等、限流、tracing、回放
- 评测：任务集、pass@k、LLM-as-judge 与人工校准、成本与延迟指标
- MCP：协议、server/client、工具暴露方式

**动手**

1. 裸写 agent harness（不引框架）：工具注册、主循环、流式、中断、重试
2. 接三个工具：文件系统（沙箱）、HTTP、代码执行
3. 建 20–50 题评测集，跑两个模型，出对比报告
4. 再用 LangGraph 或 OpenAI Agents SDK 实现同样功能，写一篇对比笔记

**验收**

- [ ] harness 一条命令跑出 pass@1、单任务成本、P95 延迟
- [ ] 能解释不同上下文压缩策略的取舍
- [ ] 有 trace 可视化或回放能力

**资源**：Anthropic 三篇、OpenAI 指南、MCP 规范、τ-bench、BFCL

---

### 阶段 3 · 后训练基础（6–8 周）

**目标**：独立完成 SFT → 偏好优化 → RL 的完整链路，并做出可解释的消融。

**学什么**

- SFT：数据格式、chat template、label masking、packing、数据配比与去重、decontamination
- 偏好优化：DPO / IPO / ORPO / KTO 的损失设计与适用场景
- RL：policy gradient、importance sampling、KL 正则、GAE、PPO、GRPO
- 奖励：reward model、规则奖励、可验证奖励、reward hacking
- 评测：基准选择、数据污染检查、显著性

**动手**

1. 用 5k–20k 条数据 SFT Qwen3-1.7B，对比三种数据配比
2. 在 SFT 模型上做一版 DPO，对比 baseline
3. 在数学或代码可验证任务上训一版 GRPO，记录 reward 曲线与回答长度变化
4. 做消融：学习率、KL 系数、group size、数据难度

**验收**

- [ ] 三条链路都有 baseline、曲线、失败分析
- [ ] 能解释 GRPO 相比 PPO 省掉了什么、代价是什么
- [ ] 能定位 reward hacking 的具体表现并给出缓解方案

**资源**：InstructGPT、DPO、DeepSeekMath、DeepSeek-R1、Tulu 3、TRL 文档

---

### 阶段 4 · Agentic RL（8–10 周）

**目标**：把 agent 与训练接起来，这是差异化的核心。

**学什么**

- 环境：可重置、可验证、可扩展；工具沙箱、模拟器、状态管理
- 多轮 rollout：轨迹采集、turn-level 与 trajectory-level 奖励、信用分配
- 奖励设计：结果奖励、过程奖励、rubric judge、稠密化、抗 hack
- 训练难点：稀疏奖励、长尾失败、长度爆炸、KL 崩塌、off-policy 漂移
- 数据飞轮：失败轨迹 → 新任务 → 再训练；强模型轨迹蒸馏
- 框架：verl、SkyRL、OpenRLHF 的架构差异与选择

**动手**

1. 自建一个工具环境（SQL / CLI / 轻量浏览器），带可验证奖励
2. 用 verl 接入环境，跑多轮 GRPO，训 4B 模型
3. 对比 SFT-only 与 SFT+RL，分析提升来自哪类任务
4. 写一篇复现报告，包含 reward hacking 案例与修法

**验收**

- [ ] held-out 任务上稳定超过 SFT baseline
- [ ] 能画出从 rollout 到参数更新的完整数据流
- [ ] 有一份别人能复现的实验配置

**资源**：verl、SkyRL、RAGEN、Search-R1、ReTool、SWE-RL、τ²-bench、SWE-bench

---

### 阶段 5 · 训练 Infra（与阶段 4 并行，持续）

**目标**：让训练跑得起来、跑得快、挂不了。

**学什么**

- 并行：数据/张量/流水/序列并行、ZeRO、FSDP、Megatron
- 通信：NCCL、all-reduce / all-gather、显存与带宽瓶颈
- 编排：Ray、任务队列、容错、checkpoint、断点续训
- RL 系统：colocate、权重同步、异步 rollout、rollout 与训练的配比
- 观测：GPU 利用率、MFU、OOM 排查、慢节点

**动手**

1. 用 FSDP 把 7B 模型跑在多卡上，记录 MFU
2. 给阶段 4 的训练加 checkpoint 与断点续训
3. 做一次吞吐优化（batch、序列打包、offload），量化收益

**验收**

- [ ] 能解释 8 卡训练 7B 的切分方案与通信开销
- [ ] 能独立排查一次 OOM 或 hang
- [ ] 有一次可量化的吞吐优化

**资源**：CS336 系统部分、Megatron-LM 文档、DeepSpeed 文档、verl 源码

---

### 阶段 6 · 求职（第 20 周起，与阶段 4 并行）

- **简历**：每个项目用数字说话（pass@1 x→y、成本降 z%、吞吐 +w%）
- **面试**：coding、ML system design、论文讨论、take-home
- **开源**：给 verl / TRL / vLLM / SGLang / SWE-agent 提 good first issue
- **渠道**：内推、论文作者、框架 Discord、X

## 5. 作品集

三个项目足够，但每个都要能讲 20 分钟。

### P1 · 裸写 Agent Harness

**一句话**：不依赖框架实现一个可评测的 agent 运行时。

必须包含：工具注册与 schema、主循环与中断、沙箱、记忆与压缩、重试、tracing、评测集。

- 验收：clone 后一条命令复现分数；报告含 pass@1、成本、延迟、失败分类
- 加分：支持 MCP；多模型切换；trace viewer
- 简历写法：`从零实现 agent 运行时，在 τ-bench 子集上 pass@1 从 x% 提升到 y%，单任务成本降低 z%`

### P2 · 后训练三连

**一句话**：在 1.7B–4B 模型上完成 SFT→DPO→GRPO 并做消融。

必须包含：数据构建脚本、训练配置、评测脚本、W&B 曲线、失败分析。

- 验收：三阶段对比表；至少四组消融；结论可复现
- 加分：数据合成 pipeline；回答长度与 reward 的关系分析
- 简历写法：`构建 2 万条领域数据，SFT+GRPO 后数学基准 +x 分，给出数据质量与 group size 的消融`

### P3 · Agentic RL 环境

**一句话**：自建可验证的工具环境，接 verl 做多轮 RL。

必须包含：环境实现、奖励函数、rollout 配置、训练脚本、评测。

- 验收：held-out 上超过 SFT；有 reward hacking 分析
- 加分：环境支持自动出题；多环境混合训练
- 简历写法：`自建 SQL/CLI 环境与可验证奖励，多轮 GRPO 训练 4B 模型，held-out 成功率 +x%，定位并修复三类 reward hacking`

### 写作与开源

- 每个阶段结束写一篇 800–1500 字复盘，发到博客 / 知乎 / X
- 给用过的框架提至少 3 个 PR（文档、测试、bug 都算）
- 论文复现报告至少 3 篇，其中 1 篇要有自己的消融结论

## 6. 资源库

### 6.1 课程

| 资源 | 时长 | 说明 |
|---|---|---|
| [Stanford CS336](https://stanford-cs336.github.io/) | 10 周 | 从零实现语言模型，覆盖训练与系统，最值得完整跟 |
| [Karpathy Zero to Hero](https://karpathy.ai/zero-to-hero.html) | 20h | 手写 backprop 到 GPT，阶段 0 用 |
| [OpenAI Spinning Up](https://spinningup.openai.com/) | 15h | 补 policy gradient 与 PPO 的直觉 |

### 6.2 Agent 方向

| 类型 | 资源 |
|---|---|
| 博客 | [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)、[Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)、Writing Effective Tools |
| 指南 | [OpenAI: A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) |
| 协议 | [Model Context Protocol](https://modelcontextprotocol.io/) |
| 框架 | [SWE-agent](https://github.com/SWE-agent/SWE-agent)、[OpenHands](https://github.com/All-Hands-AI/OpenHands) |

### 6.3 后训练方向

| 类型 | 资源 |
|---|---|
| 综述 | Tulu 3、Llama 3 Herd、Qwen3 技术报告 |
| 算法 | InstructGPT、DPO、PPO、DeepSeekMath（GRPO） |
| 推理 RL | DeepSeek-R1、ScaleRL、ProRL、Absolute Zero |
| 博客 | [Interconnects](https://www.interconnects.ai/)、[Lil'Log](https://lilianweng.github.io/) |

### 6.4 代码库

| 代码库 | 用途 | 重点读什么 |
|---|---|---|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 最小 GPT 实现 | 全部 |
| [TRL](https://github.com/huggingface/trl) | SFT / DPO / GRPO | Trainer 与 loss 实现 |
| [verl](https://github.com/volcengine/verl) | Agentic RL 训练 | rollout 与 trainer 的数据流 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | RLHF 全栈 | Ray 编排 |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | Agent RL | 环境接口 |
| [vLLM](https://github.com/vllm-project/vllm) | 推理引擎 | scheduler 与 prefix cache |
| [SGLang](https://github.com/sgl-project/sglang) | 推理引擎 | RadixAttention |

新框架速览（知道存在即可，不必都跑）：[slime](https://github.com/THUDM/slime)、[AReaL](https://github.com/inclusionAI/AReaL)、[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)、[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)、[rLLM](https://github.com/rllm-org/rllm)。

### 6.5 基准与环境

| 基准 | 测什么 | 仓库 |
|---|---|---|
| τ-bench / τ²-bench | 工具-用户多轮交互 | [tau-bench](https://github.com/sierra-research/tau-bench)、[tau2-bench](https://github.com/sierra-research/tau2-bench) |
| BFCL | 函数调用能力 | [排行榜](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| SWE-bench | 真实仓库修 issue | [SWE-bench](https://github.com/SWE-bench/SWE-bench) |
| Terminal-Bench | 终端任务 | [terminal-bench](https://github.com/laude-institute/terminal-bench) |
| WebArena / OSWorld / AppWorld | Web、桌面、应用操作 | [WebArena](https://github.com/web-arena-x/webarena)、[OSWorld](https://github.com/xlang-ai/OSWorld)、[AppWorld](https://github.com/StonyBrookNLP/appworld) |
| GAIA | 通用助手任务 | [GAIA](https://hf-mirror.com/datasets/gaia-benchmark/GAIA) |

## 7. 算力与成本

| 配置 | 能做什么 | 成本量级 |
|---|---|---|
| 0 卡（Kaggle / Colab） | 推理、1.5B LoRA SFT | 免费 |
| 1×24GB | 7B LoRA SFT、1.5–4B GRPO（Unsloth） | ¥2/h |
| 1×80GB | 7B 全参 SFT、4B GRPO | ¥6–8/h |
| 8×H100 | 7–14B GRPO、agentic RL | ¥120–200/h |
| 整套作品集 | 三个项目 | ¥3k–10k |

> 价格为 2026 年国内租用平台的量级估算，按实际询价调整。优先用**小模型 + 高难度任务**证明方法有效，不要用大模型 + 简单任务凑分数。

## 8. 常见坑

- **只收集框架，不写评测**：教程跟着做没问题，但最终要自己造 eval。
- **跳过 SFT 与数据，直接上 RL**：RL 会放大数据问题，而不是修复它。
- **只读论文不复现**：读懂和跑通之间隔着一条鸿沟，面试一追问就露馅。
- **不做 infra**：训练跑不起来，前面学的都是空中楼阁。
- **追新框架**：半年换一代，基础（Transformer、RL、分布式）才保值。
- **不写作**：能把一件事讲清楚，本身就是竞争力。
- **只调 prompt 不看数据**：大多数 agent 失败是任务定义与环境问题，不是 prompt 问题。

## 9. 求职准备

### 简历

- 每个项目三段：问题 → 做法 → 数字结果
- 技术栈只写真正跑过的
- 附 GitHub、博客、复现报告链接

### 面试

| 环节 | 准备方式 |
|---|---|
| Coding | 每周 2–3 题，保持手感；大厂仍有 hard |
| ML System Design | 设计一个 agent 产品 / 一条后训练流水线，练白板 |
| 论文讨论 | 精读 3 篇，能讲清动机、方法、局限、改进 |
| Take-home | 通常是训一个小模型或搭一个 agent，重点是可复现与评测 |

### 渠道

- 内推 > 官网投递 > 招聘平台
- 论文作者、框架 maintainer、Discord 社区里主动交流
- 把复盘文章发出去，让机会来找你

## 10. 可投递自检清单

- [ ] 能独立跑通 SFT 与 GRPO，并解释每一根曲线
- [ ] 能从零写一个带工具与评测的 agent
- [ ] 精读并复现过至少 3 篇关键论文
- [ ] 有一个能讲 20 分钟的深度项目
- [ ] 能在白板上设计一条后训练流水线
- [ ] 有一篇公开的技术写作
- [ ] 至少 3 个开源 PR（含文档）
- [ ] 简历上每个项目都有数字

八条中满足六条，就可以开始投递。
