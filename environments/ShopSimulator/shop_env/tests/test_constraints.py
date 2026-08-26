import unittest

from web_agent_site.engine.constraints import (
    CONSTRAINT_CONTRACT_VERSION,
    compile_task_constraint_contract,
    deterministic_price_upper,
    explicit_budget_from_instruction,
)


class GoalV2Test(unittest.TestCase):
    def test_explicit_budget_is_used_as_written(self):
        self.assertEqual(
            explicit_budget_from_instruction("预算在1000元以下。"),
            1000.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格不超过 2199 元"),
            2199.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格大概在60块钱以内"),
            60.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格在70元左右"),
            84.0,
        )
        self.assertAlmostEqual(
            explicit_budget_from_instruction("价格在170元上下"),
            204.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格在130-140元之间"),
            140.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格30元到40元之间"),
            40.0,
        )
        self.assertIsNone(
            explicit_budget_from_instruction("预算4k+"),
        )
        self.assertEqual(
            explicit_budget_from_instruction("预算在1万元左右"),
            12000.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("预算1万2以内"),
            12000.0,
        )

    def test_chinese_numerals_preserve_explicit_budget_semantics(self):
        self.assertEqual(
            explicit_budget_from_instruction("价格不超过三百元"),
            300.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("预算两千五百元以内"),
            2500.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格在三百元左右"),
            360.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格三百元到四百元之间"),
            400.0,
        )

    def test_open_ended_price_language_is_not_a_hard_upper_budget(self):
        for instruction in (
            "价格300多元就行",
            "价格三百多元就行",
            "价格4100多元",
            "价格不超过三百多元",
            "预算两千多块",
            "售价一万余元即可",
            "三百多元就行",
            "大概三百元出头",
            "预算4k+",
        ):
            with self.subTest(instruction=instruction):
                self.assertIsNone(explicit_budget_from_instruction(instruction))

    def test_missing_budget_has_deterministic_fallback(self):
        first = deterministic_price_upper("123", "买一个枕头", 999)
        second = deterministic_price_upper("123", "买一个枕头", 999)
        self.assertEqual(first, second)

    def test_task_annotations_compile_to_complete_constraint_contract(self):
        contract = compile_task_constraint_contract(
            {
                "instruction": "买一台支持热洗的洗地机",
                "attributes": ["智能", "热洗", "智能"],
                "instruction_options": ["白色"],
                # This target-only field must not become a constraint.
                "hidden_target_brand": "石头",
            }
        )
        hard = contract["hard_constraints"]
        self.assertTrue(hard["complete"])
        self.assertEqual(hard["contract_version"], CONSTRAINT_CONTRACT_VERSION)
        self.assertEqual(hard["core_functions"], ["智能", "热洗"])
        self.assertEqual(hard["brand"], [])
        self.assertEqual(hard["model"], [])
        self.assertEqual(hard["key_specs"], [])
        self.assertEqual(hard["annotated_option_count"], 1)
        self.assertEqual(contract["weighted_preferences"], [])

    def test_missing_annotation_schema_is_fail_closed(self):
        contract = compile_task_constraint_contract(
            {
                "instruction": "买一台洗地机",
                "attributes": ["洗地"],
            }
        )
        self.assertFalse(contract["hard_constraints"]["complete"])


if __name__ == "__main__":
    unittest.main()
