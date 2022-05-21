
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


class HasBestMixin(object):
    def __init__(self):
        self._is_best = False

    def is_best(self):
        return self._is_best


class MaxMeter(FoldableMeter, HasBestMixin):
    def __init__(self):
        super().__init__(max, -1e6)

    def update(self, val):
        self._is_best = val >= self.get()
        return super().update(val)


class MinMeter(FoldableMeter, HasBestMixin):
    def __init__(self):
        super().__init__(min, 1e6)

    def update(self, val):
        self._is_best = val <= self.get()
        return super().update(val)


class SumMeter(FoldableMeter):
    def __init__(self):
        super().__init__(sum, 0)


class AvgMeter(SumMeter):
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


class Meaner():
    def update(self, val, weight):
        return val / weight
