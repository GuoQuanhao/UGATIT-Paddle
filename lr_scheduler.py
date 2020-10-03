import paddle


def build_lr_scheduler(name, learning_rate, total_iteraion=0, start_iteration=1):
    if name == 'AdamOptimizer':
        return LinearDecay(learning_rate, total_iteraion, start_iteration)
    else:
        raise NotImplementedError


class LinearDecay(paddle.fluid.dygraph.learning_rate_scheduler.LearningRateDecay):
    def __init__(self, learning_rate, total_iteraion, start_iteration):
        super(LinearDecay, self).__init__()
        self.learning_rate = learning_rate
        self.total_iteraion = total_iteraion
        self.base_learning_rate = learning_rate
        self.start_iteration = start_iteration
        self.flag = True
       

    def step(self):
        if self.step_num + self.start_iteration > (self.total_iteraion // 2):
            if self.start_iteration > 500000 and self.flag:
                for i in range(self.start_iteration-(self.start_iteration // 100000) * 100000):
                    self.learning_rate -= self.base_learning_rate / (self.total_iteraion // 2)
                self.flag = False
            else:
                self.learning_rate -= self.base_learning_rate / (self.total_iteraion // 2)
            print(self.learning_rate)
        

        return self.create_lr_var(self.learning_rate)
