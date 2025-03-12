from dataclasses import dataclass, field
import itertools


@dataclass
class Game:
    digits: list
    num_digits: int
    allow_duplicates: bool
    all_secrets: list = field(init=False)

    def __post_init__(self):
        self.digits = sorted(list(self.digits))
        if self.allow_duplicates:
            self.all_secrets = [
                tuple(p) for p in itertools.product(self.digits, repeat=self.num_digits)
            ]
        else:
            self.all_secrets = [
                tuple(p) for p in itertools.permutations(self.digits, self.num_digits)
            ]
