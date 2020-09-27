import paddle


def build_lr_scheduler(name, learning_rate, total_iteraion=0):
    if name == 'AdamOptimizer':
        return LinearDecay(learning_rate, total_iteraion)
    else:
        raise NotImplementedError


class LinearDecay(paddle.fluid.dygraph.learning_rate_scheduler.LearningRateDecay):
    def __init__(self, learning_rate, total_iteraion):
        super(LinearDecay, self).__init__()
        self.learning_rate = learning_rate
        self.total_iteraion = total_iteraion
        self.base_learning_rate = learning_rate
       

    def step(self):
        if self.step_num + 2 > (self.total_iteraion // 2):
            self.learning_rate -= self.base_learning_rate / (self.total_iteraion // 2)

        return self.create_lr_var(self.learning_rate)