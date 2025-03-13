from hitblow_analyzer import Game, Solver

if __name__ == "__main__":
    game = Game(digits={0, 1, 2}, num_digits=3, allow_duplicates=True)
    solver = Solver()
    result = solver.solve(game)
    print(result.to_json())
