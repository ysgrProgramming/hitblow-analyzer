from dataclasses import dataclass


@dataclass
class Game:
    digits: list
    num_digits: int
    allow_duplicates: bool

    def __post_init__(self):
        self.digits = sorted(self.digits)
