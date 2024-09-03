from enum import Enum
from dataclasses import dataclass
from typing import Union


class Op(Enum):
    MATMUL = "@"
    ADD = "+"
    SUB = "-"
    DOT = "*"
    NOOP = ""
    POW = "**"
    POINTWISE = "p"


BINARY_OPS = {
    "__mul__": Op.DOT,
    "__matmul__": Op.MATMUL,
    "__add__": Op.ADD,
    "__sub__": Op.SUB,
    "__pow__": Op.POW,
}


@dataclass
class Int:
    x: int

    def print(self):
        return self.x
    
    def __add__(self, other):
        if isinstance(other, int):
            return Int(self.x + other)
        elif isinstance(other, Int):
            return Int(self.x + other.x)
        else:
            return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, int):
            return Int(self.x - other)
        elif isinstance(other, Int):
            return Int(self.x - other.x)
        else:
            return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, int):
            return Int(self.x * other)
        elif isinstance(other, Int):
            return Int(self.x * other.x)
        else:
            return NotImplemented
    
    def __div__(self, other):
        if isinstance(other, int):
            return Int(self.x / other)
        elif isinstance(other, Int):
            return Int(self.x / other.x)
        else:
            return NotImplemented
    
    def __str__(self):
        return f"{self.print()}"


class Tensor:
    number = 0

    def __init__(self, symbol="m", children=[], op=Op.NOOP):
        self.symbol = f"{symbol}_{self.number}"
        Tensor.number += 1
        self.children = children
        self.op = op

        for i in range(len(children)):
            if isinstance(children[i], int):
                children[i] = Int(children[i])

    def print(self):
        """
        Print out the expression that was constructed to compute this tensor
        """
        if self.op in BINARY_OPS.values():
            return f"({self.children[0].print()}) {self.op.value} ({self.children[1].print()})"
        else:
            return self.symbol
        
    def __str__(self):
        return f"{self.print()}"

    def grad(self, obj):
        if obj not in self.descendants():
            return "0"
        if self.op in BINARY_OPS.values():
            if self.op in [Op.ADD, Op.SUB]:
                return f"{self.children[0].grad(obj)} {self.op.value} {self.children[1].grad(obj)}"
            elif self.op in [Op.DOT, Op.MATMUL]:
                return f"({self.children[0].grad(obj)} {self.op.value} {self.children[1]}) + ({self.children[0]} {self.op.value} {self.children[1].grad(obj)})"
            elif self.op is Op.POW:
                return f"({self.children[1]} * {self.children[0]}) {Op.POW.value} {self.children[1] - 1} {Op.DOT.value} {self.children[0].grad(obj)}"
        elif self.op is Op.NOOP:
            # is a noop (leaf node)
            return "1"

    def descendants(self):
        yield self
        for child in self.children:
            yield child
            yield from child.descendants()


class ActivationFunction(Tensor):
    symbol: str = "σ"
    op: Op = Op.POINTWISE

    def __init__(self, data: Tensor):
        self.children = [data]

    def print(self):
        return f"{self.symbol}({self.children[0].print()})"

    def grad(self, obj):
        if obj in self.children[0].descendants():
            return f"{self.symbol}'({self.children[0].print()}) * {self.children[0].grad(obj)}"
        else:
            return "0"
        
def binop(op):
    return lambda self, other: Tensor(children=[self, other], op=op)


for function in BINARY_OPS:
    setattr(
        Tensor, function, binop(BINARY_OPS[function]),
    )


if __name__ == "__main__":
    tensor = Tensor().__matmul__(Tensor())
    print(tensor.print())
    weight = Tensor(symbol="w")
    expr = (ActivationFunction(weight @ Tensor(symbol="x")) - Tensor()) ** 2

    print(expr.print())
    print(expr.grad(weight))
