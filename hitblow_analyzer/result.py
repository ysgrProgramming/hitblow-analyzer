from dataclasses import dataclass, field
import json
from typing import Optional


@dataclass
class Result:
    best_question: Optional[tuple] = None
    expected_moves: float = 0
    # children: キーは応答 (tuple)、値は Result または Result のリスト
    children: dict = field(default_factory=dict)
    candidates: set = field(default_factory=set)

    def to_dict(self):
        # best_question は terminal branch なら null となる（None）
        best_question_str = (
            "".join(map(str, self.best_question))
            if self.best_question is not None
            else None
        )
        children_dict = {}
        for response, child in self.children.items():
            response_str = f"{response[0]}H{response[1]}B"
            if isinstance(child, list):
                children_dict[response_str] = [c.to_dict() for c in child]
            else:
                children_dict[response_str] = (
                    child.to_dict() if child is not None else None
                )
        candidates_str = " ".join("".join(map(str, c)) for c in self.candidates)
        result = {
            "best_question": best_question_str,
            "expected_moves": self.expected_moves,
            "candidates": candidates_str,
            "children": children_dict,
        }
        return result

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
