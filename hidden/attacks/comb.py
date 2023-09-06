class CombAttack(object):
    """A combination of attacks.

    Args:
        attack_list (list): A list of attacks to be applied sequentially.
    """

    def __init__(self, attack_list):
        self.attack_list = attack_list

    def __call__(self, x):
        for attack in self.attack_list:
            x = attack(x)
        return x
