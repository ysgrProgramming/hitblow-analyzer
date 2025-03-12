import itertools
import numpy as np
from collections import Counter
from .result import Result


class Solver:
    def precompute_lookup_tables(self, game):
        """
        ゲーム設定 (game) に基づいて、以下のlookupテーブルを前計算します。
          - self.all_secrets: 全ての秘密の数列（タプル）のリスト
          - self.secret_dict: 秘密の数列からそのインデックスへの写像（内部利用用）
          - self.all_questions: 全ての可能な質問（数列）のリスト
          - self.all_questions_dict: 質問からそのインデックスへの写像（O(1)アクセス）
          - self.hit_table: 質問インデックスと秘密インデックスの組に対するヒット数の2次元配列（ndarray）
          - self.blow_table: 質問インデックスと秘密インデックスの組に対するブロー数の2次元配列（ndarray）
        """
        self.game = game

        # 秘密の数列リストを生成
        if isinstance(game.all_secrets, list):
            self.all_secrets = game.all_secrets
        else:
            self.all_secrets = list(game.all_secrets)
        self.secret_dict = {secret: idx for idx, secret in enumerate(self.all_secrets)}

        # 全ての可能な質問を生成
        digits_list = list(game.digits)
        if game.allow_duplicates:
            all_questions_iter = itertools.product(digits_list, repeat=game.num_digits)
        else:
            all_questions_iter = itertools.permutations(digits_list, game.num_digits)
        self.all_questions = list(all_questions_iter)
        self.all_questions_dict = {
            question: idx for idx, question in enumerate(self.all_questions)
        }

        # 2次元対応表（lookup table）の作成（リストで生成した後、ndarrayに変換）
        n_questions = len(self.all_questions)
        n_secrets = len(self.all_secrets)
        hit_table = [[0] * n_secrets for _ in range(n_questions)]
        blow_table = [[0] * n_secrets for _ in range(n_questions)]
        for qi, question in enumerate(self.all_questions):
            for si, secret in enumerate(self.all_secrets):
                # ヒット数：同じ位置に同じ数字がいくつあるか
                hits = sum(1 for q, s in zip(question, secret) if q == s)
                # ブロー数：各数字の共通数の最小値の合計からヒット数を引いた値
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
        これまでの質問履歴（各質問はタプル）について、
        桁の位置入れ替えと数字のリラベリングをすべて試し、
        辞書順で最小となる正規化表現を返します。
        これにより、対称な質問列は同一視できます。
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

    def next_question_candidates_from_history(
        self, history_questions, digits, num_digits, allow_duplicates
    ):
        """
        これまでの質問履歴（history_questions: 質問のインデックスのタプル）に対し、
        新たな候補質問のうち、対称性により同一と見なされるものは重複排除し、
        候補となる質問のインデックスのリストを返します。

        内部的には、history_questionsを実際の質問（タプル）に変換して、canonical化を行っています。
        """
        # 履歴の質問インデックスから実際の質問タプルを取得
        history_tuples = tuple(self.all_questions[q_idx] for q_idx in history_questions)
        digits_list = list(digits)
        if allow_duplicates:
            all_candidates = itertools.product(digits_list, repeat=num_digits)
        else:
            all_candidates = itertools.permutations(digits_list, num_digits)
        rep_candidates = {}
        for candidate in all_candidates:
            if candidate in history_tuples:
                continue  # すでに実施した質問は除外
            new_history = history_tuples + (candidate,)
            canon = self.canonicalize_question_history(new_history, num_digits)
            candidate_idx = self.all_questions_dict[candidate]
            if canon in rep_candidates:
                # 同じ正規化形の場合、辞書順で小さいものを選択
                if candidate < rep_candidates[canon][0]:
                    rep_candidates[canon] = (candidate, candidate_idx)
            else:
                rep_candidates[canon] = (candidate, candidate_idx)
        return [candidate_idx for (_, candidate_idx) in rep_candidates.values()]

    def solve_state(self, history, candidates, alpha=float("inf")):
        """
        局面を再帰的に解いて、(最小期待手数, Result) を返します。
          - history: (質問インデックス, 回答) のペアのタプル列
          - candidates: 現在の候補解集合（秘密の数列のインデックスのfrozenset）

        終了条件:
          - 候補解が1個の場合は terminal branch として扱い、
            Result の best_question は None、expected_moves は 0 とする。
          - full hit ( (num_digits, 0) ) の場合も terminal branch として扱います。
        """
        if len(candidates) == 1:
            result = Result(
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
            return result

        best_expected = float("inf")
        best_result = None
        # 履歴中の質問インデックスを抽出
        history_questions = tuple(q_idx for (q_idx, _) in history)
        candidate_question_indices = self.next_question_candidates_from_history(
            history_questions,
            self.game.digits,
            self.game.num_digits,
            self.game.allow_duplicates,
        )

        for q_idx in candidate_question_indices:
            # ベクトル化：各候補秘密に対するヒット数とブロー数を一度に取得
            hits = self.hit_table[q_idx, candidates]
            blows = self.blow_table[q_idx, candidates]
            responses = np.stack((hits, blows), axis=1)  # shape: (n_candidates, 2)
            # ユニークな応答とグループ分けのために、uniqueとそのインデックスを取得
            unique_responses, inverse_indices = np.unique(
                responses, axis=0, return_inverse=True
            )
            response_groups = {}
            for group_idx, response in enumerate(unique_responses):
                # group_candidate_indices: この応答を得た候補秘密のインデックス（ndarray）
                group_candidates = candidates[inverse_indices == group_idx]
                response_groups[tuple(response)] = group_candidates

            expected_cost = 0  # 質問1回分のコスト
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
                    child_result = self.solve_state(new_history, group, alpha - 1.0)
                    children_results[response] = child_result
                    expected_cost += len(group) * child_result.expected_moves
                    if expected_cost >= alpha * len(candidates):
                        break
            expected_cost = expected_cost / len(candidates) + 1.0
            if expected_cost < best_expected:
                best_expected = expected_cost
                best_result = Result(
                    best_question=q_idx,
                    expected_moves=best_expected,
                    candidates=candidates,
                    children=children_results,
                )
                alpha = expected_cost

        return best_result

    def restore_result(self, result):
        """
        再帰的にResultオブジェクト内の候補解と質問を、
        内部で管理しているインデックスから元の形式（秘密や質問のタプル）に戻します。
        子のResultについても同様に処理します。
        """
        # 候補解を元の秘密のタプルに変換
        result.candidates = [self.all_secrets[idx] for idx in result.candidates]
        # best_questionも元の質問に変換（Noneならそのまま）
        if result.best_question is not None:
            result.best_question = self.all_questions[result.best_question]
        # 子Resultを再帰的に変換
        for response, child in result.children.items():
            result.children[response] = self.restore_result(child)
        return result

    def solve(self, game):
        """
        ゲーム設定 (game) を受け取り、Solver による最適戦略の探索を開始します。
        まず、lookupテーブルを前計算し、初期局面（候補解は秘密のインデックス全体）から再帰的に解を求めます。
        最終的に、Result内の候補解や質問を元の形式に戻して返します。
        """
        self.game = game
        self.precompute_lookup_tables(game)
        print("Precomputation done.")
        initial_candidates = np.arange(
            len(self.all_secrets)
        )  # 候補は秘密のインデックスの集合
        initial_history = ()
        result = self.solve_state(initial_history, initial_candidates)
        print("Solving done.")
        # 最終的に、候補解（および子の候補解）を元の形式に戻す
        restored_result = self.restore_result(result)
        print("Restoring done.")
        return restored_result
