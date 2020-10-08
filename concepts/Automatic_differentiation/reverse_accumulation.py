class Overloader:
    opcount = 0
    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None
    
    def _reverse_autodiff(self):
        if self.grad_value is None:
            Overloader.opcount += 1
            self.grad_value = 0
            for partial, child in self.children:
                self.grad_value += partial * child._reverse_autodiff()
        return self.grad_value
    
    def __add__(self, other):
        z = Overloader(self.value + other.value)
        self.children.append((1.0, z))
        return z

    def __mul__(self, other):
        z = Overloader(self.value * Overloader(other).value)
        self.children.append((Overloader(other).value, z))
        return z

    def __truediv__(self, other):
        other = other if isinstance(other, Overloader) else Overloader(other)
        z = Overloader(self.value / other.value)
        other.children.append((-self.value / other.value**2, z))
        return z

    def __rtruediv__(self, other):
        return Overloader(other).__truediv__(self)

    def _apply(self, func):
        func(self)

def reverse_autodiff(result, var):
    result.grad_value = 1
    var._apply(lambda x: x._reverse_autodiff())
    return var.grad_value


