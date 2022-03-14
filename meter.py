
class Meter():
    def __init__(self, agg_func):
        self.values = []
        self.agg_func = agg_func

    def __len__(self):
        return len(self.values)

    def get(self):
        return self.agg_func(self.values)

    def update(self, val):
        self.values.append(val)
        return self.get()


class FoldableMeter(Meter):
    def __init__(self, agg_func, init_val):
        super().__init__(agg_func)
        self.accu_val = init_val

    def get(self):
        return self.accu_val

    def update(self, val):
        self.values.append(val)
        self.accu_val = self.agg_func([self.accu_val, val])
        return self.get()


class MaxMeter(FoldableMeter):
    def __init__(self):
        super().__init__(max, 0)


class MinMeter(FoldableMeter):
    def __init__(self):
        super().__init__(min, 1e6)


class AvgMeter(FoldableMeter):
    def __init__(self):
        super().__init__(sum, 0)

    def get(self):
        return super().get() / len(self)


class BatchAvgMeter(Meter):
    def __init__(self):
        super().__init__(None)
        self.weights = []

    def get(self):
        return sum(self.values) / sum(self.weights)

    def update(self, val, weight):
        self.values.append(val)
        self.weights.append(weight)
        return self.get()
