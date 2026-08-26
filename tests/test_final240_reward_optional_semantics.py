import json
import unittest
from pathlib import Path

from web_agent_site.engine.reward import (
    candidate_options_for_evaluation,
    evaluate_purchase,
)
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.reward_features import (
    _SEMANTIC_CLAUSE_MARKER,
    _query_semantic_segments,
)
from web_agent_site.engine.comparators import normalize_text
from web_agent_site.engine.variant_price import resolve_variant_price

ROOT = Path(__file__).resolve().parents[1]


class Final240RewardOptionalSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        task_ids = [
            json.loads(line)["task_id"]
            for line in (ROOT / "data/evaluation/slices.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        products = [
            json.loads(line)
            for line in (
                ROOT / "data/shopsimulator_official/fine_items_eval_standard.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        cls.features = {}
        cls.products = {}
        cls.instructions = {}
        for task_id in task_ids:
            product = products[task_id]
            instruction = product["instructions"][0]
            cls.products[task_id] = product
            cls.instructions[task_id] = instruction
            cls.features[task_id] = compile_reward_features(instruction, product)

    def test_known_final240_concessions_are_audit_only(self):
        expected_optional = {
            901: {"混凝土"},
            450: {"零食"},
            1399: {"拍vlog", "直播"},
            1144: {"室内装饰"},
            1016: {"加宽"},
            126: {"双吸盘"},
        }
        for task_id, expected in expected_optional.items():
            with self.subTest(task_id=task_id):
                features = self.features[task_id]
                self.assertTrue(
                    expected.issubset(
                        set(features["optional_core_function_preferences"])
                    )
                )
                self.assertTrue(
                    expected.isdisjoint(set(features["expected_core_functions"]))
                )
                contract = features["query_constraint_contract"]["constraints"]
                for value in expected:
                    matching = next(
                        item
                        for item in contract
                        if item.get("expected") == value
                    )
                    self.assertEqual(
                        matching["role"],
                        "query_preference_reference",
                    )
                    self.assertEqual(matching["query_evidence"], [])
                    self.assertTrue(matching["optional_query_evidence"])

    def test_final240_required_capabilities_are_not_weakened(self):
        expected_required = {
            407: {"电磁炉专用", "无涂层", "不粘锅"},
            1389: {"女骑行", "单肩", "学生"},
            1038: {"猪", "牛", "羊", "鸡", "鸭", "鹅", "兔", "狗"},
            409: {"多功能"},
        }
        for task_id, expected in expected_required.items():
            with self.subTest(task_id=task_id):
                self.assertTrue(
                    expected.issubset(
                        set(self.features[task_id]["expected_core_functions"])
                    )
                )

    def test_every_preference_reference_is_excluded_from_strict_evidence(self):
        for task_id, features in self.features.items():
            for constraint in features["query_constraint_contract"]["constraints"]:
                if constraint.get("role") != "query_preference_reference":
                    continue
                with self.subTest(
                    task_id=task_id,
                    expected=constraint.get("expected"),
                ):
                    self.assertEqual(constraint.get("query_evidence"), [])
                    self.assertTrue(constraint.get("optional_query_evidence"))

    def test_final240_hard_soft_and_negative_semantics_are_explicit(self):
        cases = {
            176: ("budget_upper", None, "hard"),
            1218: ("price_range", None, "soft"),
            1016: ("core_function", "加宽", "soft"),
            407: ("core_function", "无涂层", "hard"),
            122: ("core_function", "免打孔", "hard"),
            1250: ("option", "白色 无内搭短裤", "hard"),
            309: ("core_function", "无赠品", "ignore"),
        }
        for task_id, (constraint_type, expected, strength) in cases.items():
            with self.subTest(task_id=task_id):
                constraint = next(
                    item
                    for item in self.features[task_id][
                        "query_constraint_contract"
                    ]["constraints"]
                    if item.get("constraint_type") == constraint_type
                    and (expected is None or item.get("expected") == expected)
                )
                self.assertEqual(constraint["strength"], strength)
                self.assertTrue(constraint["semantics_reason"])

    def test_final240_has_no_scored_unresolved_constraint_semantics(self):
        unresolved = []
        for task_id, features in self.features.items():
            for constraint in features["query_constraint_contract"]["constraints"]:
                if (
                    constraint.get("strength") == "needs_review"
                    and constraint.get("enforcement") == "scored"
                ):
                    unresolved.append((task_id, constraint))
        self.assertEqual(unresolved, [])

    def test_every_final240_semantic_clause_is_preserved_in_contract(self):
        uncovered = []
        for task_id, features in self.features.items():
            query = self.instructions[task_id]["instruction"]
            constraints = features["query_constraint_contract"]["constraints"]
            for clause in _query_semantic_segments(query):
                normalized_clause = normalize_text(clause)
                if not clause or not _SEMANTIC_CLAUSE_MARKER.search(
                    normalized_clause
                ):
                    continue
                if not any(
                    normalized_clause
                    in normalize_text(constraint.get("query_quote"))
                    for constraint in constraints
                ):
                    uncovered.append((task_id, clause))
        self.assertEqual(uncovered, [])

    def test_known_unscored_semantics_are_retained_without_reward_failures(self):
        expected_clauses = {
            292: ("直径差不多在180毫米", "soft"),
            1332: ("不要人工合成的那种", "hard"),
            336: ("必须是原厂家店铺的真货", "hard"),
            516: ("珠子尺寸在 4mm-7mm 之间", "hard"),
            604: ("65cm以上比较合适", "soft"),
            699: ("座椅不需要带有高度或角度调节功能", "ignore"),
        }
        for task_id, (expected_clause, expected_strength) in expected_clauses.items():
            with self.subTest(task_id=task_id):
                contract = self.features[task_id][
                    "query_constraint_contract"
                ]["constraints"]
                matching = [
                    item
                    for item in contract
                    if item.get("constraint_type") == "query_clause"
                    and normalize_text(expected_clause)
                    in normalize_text(item.get("query_quote"))
                ]
                self.assertTrue(matching)
                self.assertTrue(
                    all(item.get("enforcement") == "audit_only" for item in matching)
                )
                self.assertTrue(
                    all(item.get("strength") == expected_strength for item in matching)
                )

    def test_words_containing_bie_are_not_mistaken_for_negative_constraints(self):
        false_positive_clauses = {
            1065: "特别适合钓鱼或垂钓时使用",
            1070: "颜色分别为蓝色和粉色",
            679: "我准备去别的国家游玩",
            884: "轻松告别狮子头",
            937: "底部方方正正特别稳",
        }
        for task_id, false_clause in false_positive_clauses.items():
            with self.subTest(task_id=task_id):
                query_clauses = [
                    item
                    for item in self.features[task_id][
                        "query_constraint_contract"
                    ]["constraints"]
                    if item.get("constraint_type") == "query_clause"
                ]
                self.assertFalse(
                    any(
                        normalize_text(false_clause)
                        in normalize_text(item.get("query_quote"))
                        for item in query_clauses
                    )
                )

    def test_all_final240_gold_variants_are_gold_purchases(self):
        non_gold = {}
        for task_id, features in self.features.items():
            product = self.products[task_id]
            instruction = self.instructions[task_id]
            goal = {
                "asin": product["asin"],
                "category": product["category"],
                "instruction_text": instruction["instruction"],
                **features,
            }
            selected, _ = candidate_options_for_evaluation(
                product,
                goal["required_options_by_key"],
            )
            result = evaluate_purchase(
                product,
                goal,
                selected_options=selected,
                price_resolution=resolve_variant_price(product, selected),
            )
            if result.reward_type != "gold_purchase":
                non_gold[task_id] = result.to_dict()

        self.assertEqual(non_gold, {})


if __name__ == "__main__":
    unittest.main()
