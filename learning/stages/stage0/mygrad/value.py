"""mygrad：标量级自动微分引擎（T1.1 练习骨架）

对应 Zero to Hero 第 1 讲《building micrograd》。前向（建图）已经写好，
你要补两件事：

  1. 每个算子闭包里的局部导数 —— 6 处 `TODO`
  2. `Value.backward()` —— 拓扑排序 + 逆序累加

三条规则（视频里反复强调）：

  * 前向建图：新节点的 `_prev` 记下参与运算的父节点
  * 反向只走一次：每个节点把自己的 grad 按局部导数分给父节点
  * 梯度一律 `+=`：一个节点可能被多条路径用到，覆盖式赋值会丢梯度

验收：`pytest -q` 全绿，且 test_readme_expression 的数字与 micrograd README
一致（g=24.7041、dg/da=138.8338、dg/db=645.5773）。

提示：先别看 micrograd/engine.py，写完再对照，差异才是收获。
"""

from __future__ import annotations

import math


class Value:
    """一个标量节点：包装 float，并记录它在计算图中的位置。"""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None  # 局部分发函数；叶子节点默认空操作
        self._prev = set(_children)     # 计算图：指向父节点
        self._op = _op                  # 仅用于调试打印
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------------
    # 前向：已实现。读的时候只关注一件事——"图是在哪一行建起来的"
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # TODO 1：加法把 out.grad 原样分给 self 和 other
            raise NotImplementedError("TODO 1: '+' 的局部导数")

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO 2：d(ab)/da = b，d(ab)/db = a。注意 out.grad 已经是"上游传下来的部分"
            raise NotImplementedError("TODO 2: '*' 的局部导数")

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "只支持数字指数；Value 做指数先不处理"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            # TODO 3：d(x**n)/dx = n * x**(n-1)。n 可能是负数或小数
            raise NotImplementedError("TODO 3: '**' 的局部导数")

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        e = math.exp(2 * x)
        out = Value((e - 1) / (e + 1), (self,), "tanh")

        def _backward():
            # TODO 4：d(tanh)/dx = 1 - tanh(x)**2。用 out.data 算比重新 exp 更快
            raise NotImplementedError("TODO 4: tanh 的局部导数")

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # TODO 5：x>0 梯度为 1，否则 0。x==0 处约定为 0（与 micrograd 一致）
            raise NotImplementedError("TODO 5: relu 的局部导数")

        out._backward = _backward
        return out

    # ---------------- 下面这些是"语法糖"，全部复用上面的算子 ----------------

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Value(other) / self

    # ------------------------------------------------------------------
    # 反向：核心，留给你写
    # ------------------------------------------------------------------

    def backward(self):
        """从 self 出发做反向传播。

        要做的三件事（顺序别乱）：

          1. self.grad = 1.0  —— d(self)/d(self) = 1，链式法则的种子
          2. 收集 self 的所有祖先节点，排成拓扑序（父节点一定排在子节点之前）
             典型写法：DFS + visited 集合，节点加入列表的时机是"子节点都处理完"之后
          3. 逆序遍历，逐个调用 node._backward()

        注意：这里**不清零**梯度。连续调用两次 backward 会累加，这是刻意的语义
        （见 tests/test_value.py::test_backward_twice_accumulates），训练循环里
        每步要自己 reset。
        """
        raise NotImplementedError("TODO 6: 拓扑排序 + 逆序调用 _backward")
