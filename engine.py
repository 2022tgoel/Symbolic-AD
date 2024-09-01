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

    def print(self, is_root=False):
        return self.x


class Tensor:
    number = 0

    def __init__(self, symbol="w", children=[], op=Op.NOOP):
        self.symbol = f"{symbol}_{self.number}"
        Tensor.number += 1
        self.children = children
        self.op = op

        for i in range(len(children)):
            if isinstance(children[i], int):
                children[i] = Int(children[i])

    def print(self, is_root=True):
        '''
        Print out the expression that was constructed to compute this tensor
        '''
        if self.op in BINARY_OPS.values():
            return f"{(self.symbol + ' = ') if is_root else ''}({self.children[0].print(False)}) {self.op.value} ({self.children[1].print(False)})"
        else:
            return self.symbol
        
    def grad(self, obj):
        if obj not in self.descendants:
            return "0"
        if self.op in [Op.ADD, Op.SUB]:
            return f"{self.children[0].grad(obj)} {self.op.value} {self.children[1].grad(obj)}"
        elif self.op is Op.DOT:
            return 
         

    def descendants(self):
        for child in self.children:
            yield child
            yield from child.descendants()



def binop(op):
    return lambda self, other: Tensor(children=[self, other], op=op)


for function in BINARY_OPS:
    setattr(
        Tensor, function, binop(BINARY_OPS[function]),
    )


@dataclass
class ActivationFunction:
    data: Union[Tensor, Int]
    symbol: str = "σ"

    def print(self):
        return f"{self.symbol}({self.data.print()})"
    
    def grad(self, obj):
        if obj in self.data.descendants():
            return f"{self.symbol}'({self.data.print()}) * {self.data.grad()}"
        else:
            return "0"


if __name__ == "__main__":
    tensor = Tensor().__matmul__(Tensor())
    print(tensor.print())
    weight = Tensor()
    expr = (ActivationFunction(weight @ Tensor(symbol='x')) - Tensor()) ** 2

    print(expr.print())
    print(expr.grad(weight))
