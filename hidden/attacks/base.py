class BaseAttack(object):
    def __init__(self, args):
        raise NotImplementedError("Subclass should implement __init__ method")

    def setup(self):
        pass

    def attack(self, x, **kwargs):
        raise NotImplementedError("Subclass should implement attack method")

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)
