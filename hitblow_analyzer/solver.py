import itertools
import numpy as np
from collections import Counter
from .result import Result


class Solver:
    def precompute_lookup_tables(self, game):
        """
        ゲーム設定 (game) に基づいて、lookupテーブルを前計算します。
            - self.all_secrets: 全ての秘密（かつ質問）の数列（タプル）のリスト
            - self.all_secrets_dict: 秘密（質問）からそのインデックスへの写像
            - self.hit_table, self.blow_table: 質問×秘密の応答（ヒット・ブロー）を表す2次元ndarray
        """
        self.game = game
        self.all_secrets_dict = {
            secret: idx for idx, secret in enumerate(self.all_secrets)
        }

        # 2次元lookup tableの作成（質問は全ての秘密と同じ集合）
        n_secrets = len(self.all_secrets)
        hit_table = [[0] * n_secrets for _ in range(n_secrets)]
        blow_table = [[0] * n_secrets for _ in range(n_secrets)]
        for qi, question in enumerate(self.all_secrets):
            for si, secret in enumerate(self.all_secrets):
                # ヒット数：同じ位置に同じ数字がある数
                hits = sum(1 for q, s in zip(question, secret) if q == s)
                # ブロー数：各数字の共通個数の合計からヒット数を引いた値
                q_count = Counter(question)
                s_count = Counter(secret)
                total_common = sum(min(q_count[d], s_count[d]) for d in q_count)
                blows = total_common - hits
                hit_table[qi][si] = hits
                blow_table[qi][si] = blows

        self.hit_table = np.array(hit_table)
        self.blow_table = np.array(blow_table)

    @staticmethod
    def canonicalize_question_history(history_questions, num_digits):
        """
        これまでの質問履歴（各質問はタプル）について、桁の位置入れ替えと数字のリラベリングを全パターン試し、
        辞書順で最小となる正規化表現を返します。これにより、対称な質問列は同一視できます。
        """
        positions_perms = list(itertools.permutations(range(num_digits)))

        def normalize_digits(seq):
            mapping = {}
            next_val = 0
            normalized = []
            for question in seq:
                norm_question = []
                for d in question:
                    if d not in mapping:
                        mapping[d] = next_val
                        next_val += 1
                    norm_question.append(mapping[d])
                normalized.append(tuple(norm_question))
            return tuple(normalized)

        best = None
        for perm in positions_perms:
            transformed = tuple(tuple(q[i] for i in perm) for q in history_questions)
            normalized = normalize_digits(transformed)
            if best is None or normalized < best:
                best = normalized
        return best

    def next_question_candidates_from_history(self, history_questions, num_digits):
        """
        これまでの質問履歴（history_questions: 質問のタプルの並び）から、
        対称性により同一と見なされる候補質問を重複排除して生成します。
        キャッシュキーは、履歴の正規化表現のみ（回答情報は含まない）です。
        戻り値は、候補質問のインデックスのリスト（numpy配列）です。
        """
        canon = self.canonicalize_question_history(history_questions, num_digits)

        if canon in self._next_question_cache:
            return self._next_question_cache[canon]

        rep_candidates = {}
        for candidate in self.all_secrets:
            if candidate in history_questions:
                continue  # 既に実施した質問は除外
            new_history = history_questions + (candidate,)
            canon_new = self.canonicalize_question_history(new_history, num_digits)
            candidate_idx = self.all_secrets_dict[candidate]
            if canon_new not in rep_candidates:
                rep_candidates[canon_new] = candidate_idx
        candidate_list = np.array(list(rep_candidates.values()))
        self._next_question_cache[canon] = candidate_list
        return candidate_list

    def solve_state(self, history, candidates, depth_limit=float("inf")):
        """
        局面を再帰的に解いて、(最小期待手数, Result) を返します。
          - history: (質問インデックス, 回答) のペアのタプル列
          - candidates: 現在の候補解（秘密の数列のインデックスを格納したnumpy配列）
          - depth_limit: 反復深化のための残り探索深度
        終了条件:
          - 候補解が1個の場合は terminal branch として解を返す。
          - 深さ制限 (depth_limit == 0) に達した場合は、十分な解が見つかっていないと判断し、
            expected_moves を float("inf") として返します。
        """
        if depth_limit == 0:
            return Result(
                best_question=None,
                expected_moves=float("inf"),
                candidates=candidates,
                children={},
            )

        if len(candidates) == 1:
            return Result(
                best_question=candidates[0],
                expected_moves=1.0,
                candidates=candidates,
                children={
                    (self.game.num_digits, 0): Result(
                        best_question=None,
                        expected_moves=0,
                        candidates=candidates,
                        children={},
                    )
                },
            )

        best_expected = float("inf")
        best_result = Result(
            best_question=None,
            expected_moves=float("inf"),
            candidates=candidates,
            children={},
        )
        # 履歴の質問は、各ペアの質問インデックスを実際の質問タプルに変換して取得する
        history_questions = tuple(self.all_secrets[q_idx] for (q_idx, _) in history)
        candidate_question_indices = self.next_question_candidates_from_history(
            history_questions,
            self.game.num_digits,
        )

        for q_idx in candidate_question_indices:
            # 各候補秘密に対するヒット・ブローを一括取得
            hits = self.hit_table[q_idx, candidates]
            blows = self.blow_table[q_idx, candidates]
            responses = np.stack((hits, blows), axis=1)  # shape: (n_candidates, 2)
            unique_responses, inverse_indices = np.unique(
                responses, axis=0, return_inverse=True
            )
            response_groups = {}
            for group_idx, response in enumerate(unique_responses):
                group_candidates = candidates[inverse_indices == group_idx]
                response_groups[tuple(response)] = group_candidates

            expected_cost = 0.0
            children_results = {}
            for response, group in response_groups.items():
                if response == (self.game.num_digits, 0):
                    child_result = Result(
                        best_question=None,
                        expected_moves=0,
                        candidates=group,
                        children={},
                    )
                    children_results[response] = child_result
                else:
                    new_history = history + ((q_idx, response),)
                    child_result = self.solve_state(new_history, group, depth_limit - 1)
                    children_results[response] = child_result
                    expected_cost += len(group) * child_result.expected_moves
            expected_cost = expected_cost / len(candidates) + 1.0
            if expected_cost < best_expected:
                best_expected = expected_cost
                best_result = Result(
                    best_question=q_idx,
                    expected_moves=best_expected,
                    candidates=candidates,
                    children=children_results,
                )

        return best_result

    def restore_result(self, result):
        """
        再帰的にResultオブジェクト内の候補解と質問を、
        内部のインデックスから元のタプル形式に戻します。
        子のResultも同様に処理します。
        """
        result.candidates = [self.all_secrets[idx] for idx in result.candidates]
        if result.best_question is not None:
            result.best_question = self.all_secrets[result.best_question]
        for response, child in result.children.items():
            result.children[response] = self.restore_result(child)
        return result

    def solve(self, game):
        """
        ゲーム設定 (game) を受け取り、Solverによる最適戦略の探索を開始します。
        反復深化法により、深さ制限を徐々に拡大しながら完全解が得られるまで探索します。
        条件を満たす解が得られない場合は、expected_movesが float("inf") となります。
        最終的に、Result内の候補解と質問を元の形式に戻して返します。
        """
        self.game = game
        self._next_question_cache = {}
        if game.allow_duplicates:
            self.all_secrets = [
                tuple(p) for p in itertools.product(game.digits, repeat=game.num_digits)
            ]
        else:
            self.all_secrets = [
                tuple(p) for p in itertools.permutations(game.digits, game.num_digits)
            ]
        self.precompute_lookup_tables(game)
        print("Precomputation done.")
        initial_candidates = np.arange(len(self.all_secrets))
        initial_history = ()

        depth_limit = 1
        while True:
            print(f"Solving with depth limit = {depth_limit}")
            result = self.solve_state(initial_history, initial_candidates, depth_limit)
            if result.expected_moves < float("inf"):
                break
            depth_limit += 1

        restored_result = self.restore_result(result)
        print("Solving and restoring done.")
        return restored_result
