# 阶段 0 · 校准：学习内容与任务清单

> 周期：第 1–4 周（2026-W45 ~ 2026-W48）· 全职 35–45h/w ｜ 在职 12–18h/w
> 上游：[roadmap.md](../../roadmap.md) 阶段 0 ｜ 状态：🟨 进行中 ｜ 建立于 2026-11-01
> 目标不是重学 ML 理论，而是**拿到"改代码、debug 训练"的手感**：给你一条不收敛的 loss 曲线，你能在一小时内定位到原因。

## 0. 出口标准

阶段 0 结束的判定，不看"看完了几讲"，看下面六件东西是否存在：

| # | 交付物 | 位置 | 判定标准 |
|---|---|---|---|
| E1 | 手写 autograd（micrograd 等价物） | `learning/stages/stage0/micrograd/` | 与 `torch.autograd` 梯度校验误差 < 1e-6 |
| E2 | 手写 GPT + 因果性测试 | `learning/stages/stage0/nanogpt/` | 位置 t 的输出对 t+1 的输入不敏感；参数量 ≈10.65M |
| E3 | 两次可对比的训练 run | [experiments.md](../../experiments.md) EXP-001/002 | 有曲线、有超参、有"和 baseline 比怎么样"的结论 |
| E4 | 显存估算表 | `learning/stages/stage0/vram.md` | 对手写 GPT 与 7B LoRA 各估一次，实测误差 < 20% |
| E5 | LoRA SFT 三连 | [experiments.md](../../experiments.md) EXP-003~005 | r / target_modules / lr 各一组消融，方向能解释 |
| E6 | 阶段复盘 | `learning/notes/stage0-retro.md` | 能回答第 7 节全部自测题 |

对应的三条验收项（来自 [roadmap.md:68](../../roadmap.md)）：能说出 loss 不降的五种原因并给出排查顺序、能在 24GB 卡上估算 7B LoRA 显存、有一次完整 run 的日志与曲线。三条都要落到 E3/E4/E5 上，不能只写在脑子里。

## 1. 模块总览

| 模块 | 主题 | 对应材料 | 天数（全职） | 核心产出 |
|---|---|---|---|---|
| M0 | 环境与工具链 | 自建 | 0.5 | 可复现的运行环境 + 测试骨架 |
| M1 | 反向传播 | ZTH 第 1 讲 | 2 | 手写 autograd + 梯度校验 |
| M2 | 语言模型与训练直觉 | ZTH 第 2–4 讲 | 3–4 | makemore 复现 + 调参曲线 |
| M3 | Transformer 与 GPT | ZTH 第 7 讲 + nanoGPT 源码 | 4–5 | 手写 GPT + 因果性测试 |
| M4 | 训练工程 | 自建 | 2–3 | 混合精度、累积、调度、断点续训、显存表 |
| M5 | LoRA SFT | peft / transformers | 3–4 | 三条消融实验 + 显存结论 |

材料链接：[Zero to Hero 主页](https://karpathy.ai/zero-to-hero.html)、[nn-zero-to-hero 代码](https://github.com/karpathy/nn-zero-to-hero)、[nanoGPT](https://github.com/karpathy/nanoGPT)。代码统一放 `learning/stages/stage0/`，一条命令可复现（`make` 或 `run.sh`），环境版本写进 `env.md`。

## 2. M0 · 环境与工具链（0.5 天）

**学什么**：venv/uv、`typing` + `dataclass`、`asyncio` 基本并发、`pytest`、`logging`（不要 `print`）、Docker 最小镜像、git 分支与 PR 流程、CUDA/cuDNN 校验。

**任务**

- [ ] T0.1 建环境，记录 `python -V`、`torch.__version__`、`torch.version.cuda`、`torch.cuda.get_device_name(0)`、显存大小 → `learning/stages/stage0/env.md`
- [ ] T0.2 写 3 个 pytest 用例：dataclass 序列化往返、`asyncio.gather` 并发正确性、tensor shape 断言；`pytest -q` 全绿
- [ ] T0.3 写 20 行 Dockerfile（`nvidia/cuda` 基础镜像 + 装 torch），容器内跑通 T0.1 的检查脚本
- [ ] T0.4 训练脚本统一用 `logging` + `argparse`，日志同时落文件（后续所有 run 都靠它做对比）
- [ ] T0.5 建一个 `learning/stages/stage0/Makefile`，目标：`env` / `test` / `train` / `sample`

**验收**：`make test` 在裸机与容器里都能过；`env.md` 里的显存数字和 `nvidia-smi` 对得上。

**卡点提示**：torch 装成 CPU 版是最常见的坑，`torch.cuda.is_available()` 为 False 时先查 wheel 的 cu 版本而不是重装驱动。

## 3. M1 · 反向传播：micrograd（2 天）

**学什么**：计算图、链式法则、拓扑排序、`+=` 累加梯度、`tanh`/`relu` 的局部导数、为什么需要 zero_grad。材料：ZTH 第 1 讲《building micrograd》（2h25m）。

**任务**

- [ ] T1.1 先不看视频，凭理解写 `Value` 类：`__add__/__mul__/__pow__/__tanh__/__relu__`、`backward()`（拓扑排序 + 累加）→ 骨架在 [`mygrad/`](mygrad/)，前向已给，补 6 处 TODO；`pytest -q` 从 4 passed / 14 failed 打到 18 passed
- [ ] T1.2 用它在二维 toy 数据上训一个 2 层 MLP 做二分类，画出决策边界 → `assets/stage0-micrograd.png`
- [ ] T1.3 写 `tests/test_gradcheck.py`：同一网络分别用自写 autograd 与 `torch.autograd` 求梯度，断言 `max|diff| < 1e-6`（用 float64）
- [ ] T1.4 写 300 字笔记：为什么必须拓扑排序、为什么梯度是 `+=` 而不是 `=`、`zero_grad` 漏了会怎样

**验收**：T1.3 通过；能对着代码口述一遍"loss → w2 → h → w1"的梯度回流路径。

**卡点提示**：坑集中在三处——忘 `+=`（`x*x + x` 立刻挂）、拓扑序反了（深层图用到没算完的 grad）、`pow` 的局部导数漏乘 `out.grad`；详见 [`mygrad/README.md`](mygrad/README.md)。softmax + 交叉熵建议合并手推，别拆成两步再梯度校验。

## 4. M2 · 语言模型与训练直觉：makemore（3–4 天）

**学什么**：字符级语言建模框架（train/sample/eval）、NLL loss、train/dev/test 切分、embedding、隐藏层宽度、学习率与 batch size 的相互作用、激活与梯度尺度、BatchNorm 的前向与反向。材料：第 2 讲 bigram（1h57m）、第 3 讲 MLP（1h15m）、第 4 讲 Activations & Gradients（1h55m）。

**任务**

- [ ] T2.1 复现 bigram 计数版 + 神经版 bigram，loss 量级应落在 ~2.4（nats，`names.txt` 32k 人名），记录 train/val
- [ ] T2.2 复现 MLP 版（embedding → concat → hidden → logits），扫 3 组 hidden size 与 3 组 lr，画对比曲线 → `assets/stage0-makemore-lr.png`
- [ ] T2.3 **手动实现 BatchNorm 的前向与反向**，与 `nn.BatchNorm1d` 逐元素对比（误差 < 1e-5）；再实现"没有 running stats 的 batch norm"看推理时的崩坏
- [ ] T2.4 画激活与梯度的直方图：不缩放 vs 缩放（`*0.2` 初始化 / `*0.01` 输出层），各存一张 → `assets/stage0-act-grad.png`
- [ ] T2.5 做一次过拟合实验：把训练集缩到 200 条，直到 train loss ≈ 0，记录 val loss 与 epoch 数，写"怎么判断过拟合"的三条信号
- [ ] T2.6（可选，时间够再做）第 5 讲 Backprop Ninja（1h55m）：手推整个 2 层 MLP 的反向，与 autograd 对齐

**验收**：能说出"深层网络训练脆弱"的三种表现（激活饱和、梯度消失/爆炸、初始 loss 偏离 `-log(1/n)`）及各自的诊断图；`nn.BatchNorm1d` 的 `running_mean` 更新公式能手写。

**卡点提示**：MLP 版收敛慢是正常的（10 万步量级），别在 5000 步就判定"实现错了"；先确认初始 loss ≈ 理论值再谈收敛。

## 5. M3 · Transformer 与 GPT（4–5 天）

**学什么**：self-attention 的 QKV 与缩放、causal mask、多头拼接、残差 + LayerNorm（pre-norm）、位置编码、weight tying、dropout；然后通读 nanoGPT 全部代码。材料：第 7 讲《Let's build GPT》（1h56m）+ `nanoGPT/model.py`、`train.py`、`config/`。

**任务**

- [ ] T3.1 照视频手写 `model.py`（BigramLanguageModel → 单头 → 多头 → 残差/LayerNorm → 完整 GPT），参数量对到 ≈10.65M（`n_layer=6, n_head=6, n_embd=384`）
- [ ] T3.2 写**因果性测试**：随机输入，改 `x[t+1:]`，断言 `logits[:t+1]` 不变（对比 `atol=1e-5`）→ 进 `tests/`
- [ ] T3.3 写 attention 的 shape 测试：`(B,T,C) → (B,T,C)`，mask 后上三角的权重和为 0
- [ ] T3.4 跑 nanoGPT 的 `shakespeare_char` 配置，记录最终 val loss、总耗时、tokens/s → EXP-001
- [ ] T3.5 改**一个**超参（`n_layer` 6→4 或 lr ×3），同 seed 重跑，对比曲线并解释差异 → EXP-002
- [ ] T3.6 采样 5 段 200 字符文本贴进 EXP 记录（看得出是"莎士比亚体"才算学到了分布）

**验收**：T3.2/T3.3 通过；EXP-001/002 里能回答"和 baseline 比差多少、为什么"。

**卡点提示**：因果性测试用 `atol=1e-5` 而不是精确相等（浮点累加顺序不同）；`nanoGPT` 默认是 `shakespeare_char` 之外还有 `shakespeare` 字符级/词级两套配置，别拿错 data 目录。

## 6. M4 · 训练工程（2–3 天）

**学什么**：`Dataset`/`DataLoader` 与 map-style 数据的坑、`autocast(bf16)`、梯度累积的等价条件、checkpoint 与 resume、`torch.cuda.max_memory_allocated`、cosine + warmup、显存构成（权重 / 梯度 / 优化器状态 / 激活）。

**任务**

- [ ] T4.1 给 T3 的 GPT 加 bf16 autocast + 梯度累积，验证 `grad_accum=4 & batch=8` 与 `batch=32` 的 loss 轨迹一致（前若干步差值 < 1e-2），记录不严格相等的原因
- [ ] T4.2 加 warmup + cosine 调度，与 constant lr 对比最终 val loss（同步数）
- [ ] T4.3 实现 checkpoint/resume：训到一半 kill，恢复后 loss 与不间断训练对齐（差 < 0.02）
- [ ] T4.4 写 `vram.py`：输入 `(params, seq_len, batch, dtype, 是否 LoRA, r)`，输出权重/梯度/优化器/激活四部分的估算表，并用 `max_memory_allocated` 校验，误差 > 20% 就回去改公式
- [ ] T4.5 故意制造不收敛（`lr=10`、或忘 `zero_grad`、或把 mask 去掉），记录现象与定位手段 → 汇总成 `learning/notes/stage0-retro.md` 里的**"loss 不降的五种原因 + 排查顺序"**
- [ ] T4.6 在 24GB 卡上估算 7B 模型 LoRA（r=16）训练需要多少显存，给出"能不能跑、要什么配置"的一段结论

**验收**：E4 的显存表两项都完成且误差达标；T4.5 的清单能默写（这是 roadmap 的验收项之一）。

**卡点提示**：有 BatchNorm 时梯度累积与大 batch **不等价**（统计量按 micro-batch 算），这条要能说清，也是面试常问；bf16 一般不需要 loss scaling，fp16 需要。

## 7. M5 · LoRA SFT（3–4 天）

**学什么**：LoRA 的低秩分解 `ΔW = BA`、`alpha/r` 缩放、`target_modules`（qkv vs 全线性）、为什么省显存（优化器状态少，但**激活不变**）、`peft` + `transformers Trainer` 的用法、SFT 数据格式与 loss mask。

**任务**

- [ ] T5.1 准备 1000 条指令数据（单轮即可），写 `dataset.py`：prompt/completion 拼接 + 只对 completion 算 loss
- [ ] T5.2 Qwen3-1.7B + `peft` 跑通 SFT（bf16 + gradient checkpointing），记录显存峰值、吞吐（tokens/s）、train loss 曲线 → EXP-003
- [ ] T5.3 消融 A：`r=8` vs `r=16`（其余不变）→ EXP-004
- [ ] T5.4 消融 B：`target_modules=["q_proj","v_proj"]` vs 全部线性层 → EXP-005
- [ ] T5.5 消融 C：`lr=1e-4` vs `2e-4`；三条消融一次只改一个变量
- [ ] T5.6 held-out 20 条对比 base vs tuned，用"人工 1–5 分 + 规则指标（长度、拒答率、格式合规率）"，结论必须落到数字上
- [ ] T5.7 合并 / 不合并 adapter 各推理一次，确认输出一致（容差内），并记录合并后的显存与延迟差异
- [ ] T5.8 把"7B LoRA 在 24GB 卡"的估算与本次 1.7B 实测显存对齐，修正 T4.4 的公式

**验收**：EXP-003~005 三条记录齐全且都回答了"和 baseline 比怎么样"；能解释"LoRA 省的是哪部分显存、不省哪部分"。

**卡点提示**：只对 completion 算 loss 时，mask 写反会导致 loss 虚低但效果全无；`gradient_checkpointing` 必须配 `use_cache=False`，否则显存反而涨。Qwen3 系列建议用官方 chat template，别自己拼特殊 token。

## 8. 周计划

| 周 | 全职业余安排 | 在职安排 | 里程碑 |
|---|---|---|---|
| W45 | M0 + M1 全量（约 25h） | M0（3h）+ M1 前半（6h） | 梯度校验通过 |
| W46 | M2 全量（约 28h） | M2 前半（8h） | makemore 复现 + 调参曲线 |
| W47 | M3 全量（约 32h） | M3 前半（9h） | 手写 GPT + 因果性测试 |
| W48 | M4 + M5（约 35h） | M3 后半 + M4 前半（9h） | EXP-001/002 |
| W49 | 缓冲 + 复盘（在职模式顺延） | M4 后半 + M5 前半（9h） | 显存表 |
| W50 | — | M5 + 复盘（9h） | EXP-003~005 + 复盘 |

时间分配原则：**视频/读码 30%，亲手写 50%，写记录 20%**。视频一律 1.5–2 倍速看第一遍，写代码时回去按需查片段，不要二刷。

## 9. 削减策略（只有 2 周时）

按此顺序砍，砍到能保住出口标准为止：

1. 砍 M4 的 T4.2/T4.3（调度与断点续训，阶段 3 还会再遇到）
2. 砍 M2 的 T2.6（Backprop Ninja）与第 4 讲的直方图部分
3. 砍 M5 的消融 C（lr 扫描），保留 r 与 target_modules 两条
4. **不可砍**：M1、M3 的因果性测试、M5 的 T5.2/T5.6、E4 显存表

## 10. 自测题（答不上就回去补对应模块）

1. 为什么 attention 要除以 `sqrt(d_k)`？不除会发生什么？（→ M3）
2. 因果 mask 加在 softmax 之前还是之后？加 `-inf` 与乘 0 有什么区别？（→ M3）
3. LayerNorm 和 BatchNorm 在序列模型上的差别，为什么 GPT 用前者？（→ M2/M3）
4. 梯度累积在什么条件下等价于大 batch？什么时候不等价？（→ M4）
5. bf16 为什么通常不需要 loss scaling，fp16 需要？（→ M4）
6. LoRA 省的是权重、梯度、优化器状态还是激活？r 变大时哪一项先爆？（→ M5）
7. 训练 loss 突然变成 NaN，按什么顺序排查？（→ M4，答案要能对上 T4.5 的清单）
8. 初始 loss 应该约等于多少？偏离说明什么？（→ M2）
9. 为什么 micrograd 里梯度要 `+=`？不累加会错在哪个算子？（→ M1）
10. 你的两个 run 差 0.05 val loss，怎么判断是噪声还是真差异？（→ M3/M5）

## 11. 与后续阶段的接口

阶段 0 的产物会被后面直接复用，别写一次性代码：

- 手写 GPT + 训练循环 → 阶段 3 的 SFT 基线、阶段 4 的 rollout 模型
- 显存估算脚本 `vram.py` → 阶段 1 的服务容量规划、阶段 5 的吞吐优化
- "loss 不降"排查清单 → 阶段 3/4 的 debug 手册
- 日志与实验记录规范 → 阶段 4 的多轮 RL 对比必须靠它

完成后：更新 [roadmap.md](../../roadmap.md) 阶段 0 的验收勾选与 [README.md](../../README.md) 的进度总览，在 [log.md](../../log.md) 补一条周记录，并把复盘写进 `learning/notes/stage0-retro.md`。
