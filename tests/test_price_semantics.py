import unittest

from shopping_grpo.evaluation.rubric import extract_price_candidates
from shopping_grpo.price_semantics import approximate_price_bounds, reward_price_constraint


class PriceSemanticsTest(unittest.TestCase):
    def test_around_price_uses_exact_twenty_percent_bounds(self):
        self.assertEqual(approximate_price_bounds(100.0), (80.0, 120.0))
        self.assertEqual(approximate_price_bounds(20.0), (16.0, 24.0))

    def test_chinese_explicit_upper_budget_remains_hard(self):
        candidates = extract_price_candidates("价格不超过三百元")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["constraint_type"], "budget_upper")
        self.assertEqual(candidates[0]["operator"], "lte")
        self.assertEqual(candidates[0]["expected_value"]["value"], 300.0)
        self.assertEqual(candidates[0]["hardness_hint"], "hard")

    def test_chinese_price_range_is_parsed(self):
        candidates = extract_price_candidates("价格三百元到四百元之间")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["constraint_type"], "price_range")
        self.assertEqual(
            candidates[0]["expected_value"],
            {"min": 300.0, "max": 400.0, "currency": "CNY"},
        )

    def test_open_ended_prices_are_soft_preferences(self):
        for query, expected in (
            ("价格300多元就行", 300.0),
            ("价格三百多元就行", 300.0),
            ("价格不超过三百多元", 300.0),
            ("预算两千来块", 2000.0),
            ("价格4100多元", 4100.0),
            ("大概三百元出头", 300.0),
        ):
            with self.subTest(query=query):
                candidates = extract_price_candidates(query)
                self.assertEqual(len(candidates), 1)
                candidate = candidates[0]
                self.assertEqual(candidate["constraint_type"], "price_preference")
                self.assertEqual(candidate["operator"], "approximately")
                self.assertEqual(candidate["expected_value"]["value"], expected)
                self.assertTrue(candidate["expected_value"]["open_ended_above"])
                self.assertEqual(candidate["hardness_hint"], "soft")

    def test_around_price_is_a_soft_preference_in_rubric(self):
        candidates = extract_price_candidates("价格在170元上下")
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["constraint_type"], "price_preference")
        self.assertEqual(candidate["operator"], "approximately")
        self.assertEqual(candidate["expected_value"]["value"], 170.0)
        self.assertEqual(candidate["hardness_hint"], "soft")

    def test_colloquial_price_language_never_becomes_a_strict_upper_gate(self):
        for query in (
            "预算约300元",
            "价格300元左右",
            "售价300元上下",
            "大概300元",
            "接近300块",
            "300元前后",
            "价位在300元附近",
            "差不多300元",
        ):
            with self.subTest(query=query):
                constraint = reward_price_constraint(query)
                self.assertIsNotNone(constraint)
                self.assertEqual(constraint["operator"], "approximately")
                self.assertTrue(constraint["approximate"])

        for query in (
            "预算三百来块",
            "价格300多元",
            "大概300元出头",
        ):
            with self.subTest(query=query):
                self.assertIsNone(reward_price_constraint(query))

    def test_price_slightly_above_is_not_misread_as_an_upper_limit(self):
        constraint = reward_price_constraint("价格三十元多一点")

        self.assertIsNone(constraint)

    def test_per_unit_price_basis_is_preserved_for_reward(self):
        constraint = reward_price_constraint("大概需50包，每包价格约1.5元左右。")

        self.assertEqual(
            constraint["basis"],
            {"kind": "per_unit", "unit": "包", "source_text": "每包价格"},
        )

    def test_only_explicit_upper_language_compiles_to_a_strict_cap(self):
        for query in (
            "价格不超过300元",
            "预算控制在300元以内",
            "售价最多300元",
            "价格上限300元",
        ):
            with self.subTest(query=query):
                constraint = reward_price_constraint(query)
                self.assertIsNotNone(constraint)
                self.assertIn(constraint["operator"], {"lt", "lte"})
                self.assertFalse(constraint["approximate"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
