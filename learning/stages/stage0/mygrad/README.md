# mygrad · T1.1 手写 Value

照 Zero to Hero 第 1 讲，自己写一版 scalar autograd。**前向已给，你补反向。**

## 现状

`pytest -q` 当前是 **4 passed / 14 failed**：前向的 4 个用例是绿的，反向全部停在 `TODO 6`。

```
value.py
  TODO 1  __add__      局部导数
  TODO 2  __mul__      局部导数
  TODO 3  __pow__      局部导数
  TODO 4  tanh()       局部导数
  TODO 5  relu()       局部导数
  TODO 6  backward()   拓扑排序 + 逆序累加
```

## 建议顺序

1. **先写 `backward()`（TODO 6）**，哪怕 5 个算子的导数还没写。理由：拓扑序是这个练习的主线，写完它测试会从"卡在 TODO 6"变成"卡在 TODO 1"，报错位置就开始给你指路了。
2. 然后按 `add → mul → pow → tanh → relu` 的顺序补，每补一个跑一次 `pytest -q`。
3. 最后跑 `test_readme_expression`，它复刻的是 micrograd README 里那个刻意拼出来的长表达式，数字必须是 `24.7041 / 138.8338 / 645.5773`。

## 命令

```bash
cd learning/stages/stage0/mygrad
pytest -q                        # 全部
pytest -q -k readme              # 只跑终局用例
pytest -q -k "gradcheck"         # 只跑与 torch 的梯度校验
```

## 卡住时看哪里

按这个顺序，**逐级升级**，不要一上来就翻源码：

1. 报错信息本身就写了是哪个算子、哪条路径（`value.py:行号`）
2. `tests/test_value.py` 里对应用例的注释——每个用例都在测一个具体语义
3. 拿铅笔在纸上画 3 个节点的小图，手推一遍 `d(out)/d(输入)`
4. 最后才看 `../micrograd/micrograd/engine.py`。对照时重点看两处差异：他的 `backward()` 怎么排拓扑序、他的 `_backward` 闭包怎么拿到 `out.grad`

## 三个常见错法（写之前先有意识）

- **忘了 `+=`**：`self.grad = ...` 会让 `test_gradient_accumulation_same_node_twice` 挂掉，因为 `x*x + x` 里 x 有两条路径
- **拓扑序反了**：先遍历子节点再遍历父节点，或者用 `_prev` 的集合顺序直接递归，深层图上会用到还没算完的 grad
- **`pow` 的导数漏了 `out.grad`**：所有局部导数都要乘上游梯度，这是链式法则的全部内容

## 做完之后

- T1.2：在 `mygrad/` 里加 `nn.py`（`Neuron` / `Layer` / `MLP`），用 `make_moons` 训一次二分类，出决策边界图
- T1.3：把 `tests/test_value.py` 的 gradcheck 扩成随机表达式 + 随机输入的模糊测试
- T1.4：写 300 字笔记——为什么必须拓扑排序、为什么是 `+=`、`zero_grad` 漏了会怎样

写完对照 `../micrograd`，两份实现的差异就是这一讲的收获。
