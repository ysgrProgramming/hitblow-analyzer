import pytest
from hitblow_analyzer import Game, Solver


@pytest.mark.parametrize(
    "digits, num_digits, allow_duplicates, expected_file",
    [
        # テストケース例:
        #   - 使用数字 {0, 1, 2}
        #   - 桁数 2
        #   - 重複禁止
        #   - 期待する出力は "expected_output_case1.txt" 内の内容
        ({0, 1}, 3, True, "tests/solver_outputs/solver_output_case1.json"),
        ({0, 1, 2}, 3, False, "tests/solver_outputs/solver_output_case2.json"),
    ],
)
def test_solver_solve(digits, num_digits, allow_duplicates, expected_file):
    """
    指定されたゲーム設定に対して、Solver.solve() の出力が
    期待するファイルの内容と一致するかを検証するテストです。
    """
    game = Game(digits, num_digits, allow_duplicates)
    solver = Solver()
    result = solver.solve(game)
    # Result クラスで __str__ または適切なシリアライズが実装されている前提です
    result_str = result.to_json()

    with open(expected_file, "r", encoding="utf-8") as f:
        expected_output = f.read().strip()

    assert result_str == expected_output
