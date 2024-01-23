import math

class LearningRateScheduler:
    def __init__(self, initial_lr, total_epochs, name, kwargs):
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.params = kwargs
        self.update_func = self._get_update_func(name)

    def _get_update_func(self, name):
        if name == 'exponential_decay':
            return self.exponential_decay
        elif name == 'fixed':
            return self.fixed_learning_rate
        elif name == 'step_decay':
            return self.step_decay
        elif name == 'linear_decay':
            return self.linear_decay
        elif name == 'warmup':
            return self.warmup
        elif name == 'cosine_annealing':
            return self.cosine_annealing
        else:
            raise ValueError(f"Unknown approach: {name}")
        

    def fixed_learning_rate(self, epoch):
        return self.initial_lr

    def exponential_decay(self, epoch):
        decay_rate = self.params.get('decay_rate', 0.99)
        return self.initial_lr * (decay_rate ** epoch)

    def step_decay(self, epoch):
        step_size = self.params.get('step_size', 10)
        decay_rate = self.params.get('decay_rate', 0.5)
        return self.initial_lr * (decay_rate ** (epoch // step_size))

    def linear_decay(self, epoch):
        return self.initial_lr * (1 - epoch / self.total_epochs)

    def warmup(self, epoch):
        warmup_epochs = self.params.get('warmup_epochs', 5)
        if epoch < warmup_epochs:
            return self.initial_lr * (epoch / warmup_epochs)
        else:
            return self.initial_lr

    def cosine_annealing(self, epoch):
        min_lr = self.params.get('min_lr', 0.001)
        return min_lr + (self.initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / self.total_epochs)) / 2