"""T1.1 的验收测试：跑通全部用例，就算这个 Value 写对了。

运行（在 learning/stages/stage0/mygrad/ 下）：

    pytest -q

建议顺序：先看 __add__ 那组，再 __mul__，再 backward —— 每次只让一组变绿。
"""

import pytest
import torch

from mygrad import Value


# ----------------------------------------------------------------------
# 1. 前向：这些现在就是绿的，用来确认你改代码时没把建图逻辑改坏
# ----------------------------------------------------------------------

def test_add_forward():
    assert (Value(2.0) + Value(3.0)).data == 5.0
    assert (Value(2.0) + 3.0).data == 5.0
    assert (2.0 + Value(3.0)).data == 5.0


def test_mul_forward():
    assert (Value(4.0) * Value(3.0)).data == 12.0
    assert (Value(4.0) * 3.0).data == 12.0
    assert (3.0 * Value(4.0)).data == 12.0


def test_pow_and_div_forward():
    assert (Value(3.0) ** 2).data == 9.0
    assert (Value(3.0) ** -1).data == pytest.approx(1 / 3)
    assert (Value(6.0) / 2.0).data == 3.0
    assert (10.0 / Value(2.0)).data == 5.0


def test_relu_tanh_forward():
    assert Value(-2.0).relu().data == 0.0
    assert Value(3.0).relu().data == 3.0
    assert Value(0.5).tanh().data == pytest.approx(0.46211715726000974)
    assert Value(0.0).tanh().data == 0.0


# ----------------------------------------------------------------------
# 2. 单个算子的反向：局部导数
# ----------------------------------------------------------------------

def test_backward_add():
    a, b = Value(2.0), Value(3.0)
    (a + b).backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_backward_mul():
    a, b = Value(2.0), Value(3.0)
    (a * b).backward()
    assert a.grad == 3.0  # d(ab)/da = b
    assert b.grad == 2.0  # d(ab)/db = a


def test_backward_pow():
    x = Value(3.0)
    (x**2).backward()
    assert x.grad == 6.0  # 2 * x

    y = Value(2.0)
    (y**-1).backward()
    assert y.grad == pytest.approx(-0.25)  # -1 * y**-2


def test_backward_tanh():
    x = Value(0.5)
    out = x.tanh()
    out.backward()
    assert x.grad == pytest.approx(1 - out.data**2)


def test_backward_relu():
    x = Value(-2.0)
    x.relu().backward()
    assert x.grad == 0.0

    y = Value(2.0)
    y.relu().backward()
    assert y.grad == 1.0


# ----------------------------------------------------------------------
# 3. 图的结构：累加、拓扑序、重复调用
# ----------------------------------------------------------------------

def test_gradient_accumulation_same_node_twice():
    """x 被用了两次，两条路径的梯度必须相加，不能覆盖。"""
    x = Value(3.0)
    y = x * x + x        # y = x² + x, dy/dx = 2x + 1 = 7
    y.backward()
    assert x.grad == 7.0


def test_shared_subgraph_deep():
    """(a*b) 被 tanh 和加法各用一次：梯度和靠拓扑序 + 累加共同保证。"""
    a, b = Value(2.0), Value(3.0)
    h = a * b
    y = h.tanh() + h

    ta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    th = ta * tb
    ty = torch.tanh(th) + th
    ty.backward()

    y.backward()
    assert a.grad == pytest.approx(ta.grad.item(), abs=1e-9)
    assert b.grad == pytest.approx(tb.grad.item(), abs=1e-9)


def test_backward_twice_accumulates():
    """不清零就连调两次，梯度翻倍——这正是训练循环里必须 reset 的原因。"""
    x = Value(3.0)
    y = x * 2
    y.backward()
    assert x.grad == 2.0
    y.backward()
    assert x.grad == 4.0


# ----------------------------------------------------------------------
# 4. 梯度校验：同一段表达式分别用 mygrad 与 torch 求导（T1.3 的雏形）
# ----------------------------------------------------------------------

CASES = [
    ("mul_add", lambda a, b: a * b + a),
    ("pow", lambda a, b: a**2 * b),
    ("tanh", lambda a, b: (a * b).tanh()),
    ("relu", lambda a, b: (a + b).relu() * a),
    ("composite", lambda a, b: ((a * b).tanh() + a.relu()) * (b**2)),
]


@pytest.mark.parametrize("name,expr", CASES, ids=[c[0] for c in CASES])
def test_gradcheck_against_torch(name, expr):
    a0, b0 = 0.7, -1.3
    a, b = Value(a0), Value(b0)
    y = expr(a, b)
    y.backward()

    ta = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
    ty = expr(ta, tb)
    ty.backward()

    assert y.data == pytest.approx(ty.item(), abs=1e-9)
    assert a.grad == pytest.approx(ta.grad.item(), abs=1e-6)
    assert b.grad == pytest.approx(tb.grad.item(), abs=1e-6)


# ----------------------------------------------------------------------
# 5. 终局：micrograd README 里那个刻意拼出来的长表达式
# ----------------------------------------------------------------------

def test_readme_expression():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f

    assert g.data == pytest.approx(24.7041, abs=1e-4)

    g.backward()
    assert a.grad == pytest.approx(138.8338, abs=1e-4)
    assert b.grad == pytest.approx(645.5773, abs=1e-4)
