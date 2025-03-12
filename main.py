from hitblow_analyzer import Game, Solver

if __name__ == "__main__":
    game = Game(digits={0, 1, 2, 3, 4}, num_digits=2, allow_duplicates=False)
    solver = Solver()
    result = solver.solve(game)
    print(result.to_json())
