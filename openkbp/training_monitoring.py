class Averager:

    def __init__(self):
        self.N = 0
        self.total = 0.0

    def record(self, entry):
        self.N += 1
        self.total += entry

    def average(self):
        return self.total/self.N

class training_metrics:
    def __init__(self):
        self.adversarial_loss = []
        self.discriminator_loss = []
