from dataclasses import dataclass, field


@dataclass
class Game:
    digits: list
    num_digits: int
    allow_duplicates: bool
    all_secrets: list = field(init=False)

    def __post_init__(self):
        self.digits = sorted(self.digits)
