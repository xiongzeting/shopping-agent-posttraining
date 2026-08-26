import unittest

from shopping_grpo.environment.actions import (
    action_reject_reason,
    product_ids,
    resolve_action_parameters,
)
from shopping_grpo.environment.candidate_memory import (
    new_candidate_memory,
    update_candidate_memory,
)
from shopping_grpo.evaluation.rollout import EVALUATION_TOOL_SCHEMAS


class ActionValidationTest(unittest.TestCase):
    def test_select_option_rejects_navigation_button(self):
        """规格工具不能把页面导航按钮伪装成一个规格值。"""
        observation = '商品页\n\n可点击的按钮: ["< Prev", "糖果粉"]'

        reason = action_reject_reason("select_option", {"value": "< Prev"}, observation)

        self.assertEqual(reason, "select_option_is_navigation_button")

    def test_select_option_allows_current_product_option(self):
        option_id = "opt_0123456789abcdef"
        observation = f'商品页\n\n可点击的按钮: ["< Prev", "{option_id}"]'

        self.assertIsNone(action_reject_reason("select_option", {"value": option_id}, observation))

    def test_select_option_keeps_stable_id_unchanged(self):
        option_id = "opt_0123456789abcdef"
        observation = (
            'available_options: {"颜色分类": '
            f'[{{"option_id": "{option_id}", "label": "800g  轻门专用"}}]}}\n'
            f'可点击的按钮: ["{option_id}"]'
        )
        resolved = resolve_action_parameters(
            "select_option",
            {"value": option_id},
            observation,
        )
        self.assertEqual(resolved, {"value": option_id})

    def test_select_option_rejects_already_selected_id_without_execution(self):
        option_id = "opt_0123456789abcdef"
        observation = (
            'selected_options: {"颜色": '
            f'{{"option_id": "{option_id}", "label": "黑色"}}}}\n'
            f'可点击的按钮: ["{option_id}", "Buy Now"]'
        )

        self.assertEqual(
            action_reject_reason(
                "select_option",
                {"value": option_id},
                observation,
                evaluation_extensions=True,
            ),
            "option_already_selected",
        )

    def test_select_option_rejects_label_even_when_label_is_visible(self):
        observation = '商品页\n\n可点击的按钮: ["A B"]'
        reason = action_reject_reason(
            "select_option",
            {"value": "A B"},
            observation,
        )
        self.assertEqual(reason, "select_option_requires_stable_id")

    def test_removed_view_tool_is_unknown_even_if_legacy_button_is_visible(self):
        """废除的 view 工具不能因旧页面仍显示同名按钮而重新变为合法。"""
        observation = '商品页\n\n可点击的按钮: ["Description", "Buy Now"]'

        reason = action_reject_reason("view_description", {}, observation)

        self.assertEqual(reason, "unknown_tool")

    def test_rejects_schema_extra_argument_before_executing_tool(self):
        """无参数工具携带垃圾字段时，不能静默丢弃字段后继续执行。"""
        observation = '商品页\n\n可点击的按钮: ["Buy Now"]'

        reason = action_reject_reason("buy_now", {"string": "true"}, observation)

        self.assertEqual(reason, "schema_extra_arguments:string")

    def test_rejects_missing_required_argument_before_execution(self):
        reason = action_reject_reason("search_products", {}, "搜索功能是否可用: True")

        self.assertEqual(reason, "schema_missing_arguments:query")

    def test_rejects_wrong_argument_type_before_execution(self):
        reason = action_reject_reason(
            "open_product",
            {"asin": 1234567890},
            "1|1234567890|99|测试商品",
        )

        self.assertEqual(reason, "schema_wrong_type:asin:string")

    def test_rejects_blank_required_string_before_execution(self):
        observation = '商品页\n\n可点击的按钮: ["黑色"]'

        reason = action_reject_reason("select_option", {"value": "  "}, observation)

        self.assertEqual(reason, "schema_empty_string:value")

    def test_rejects_unknown_tool_before_execution(self):
        self.assertEqual(action_reject_reason("invented_tool", {}, ""), "unknown_tool")

    def test_rejects_non_object_arguments_before_execution(self):
        self.assertEqual(
            action_reject_reason("buy_now", [], ""),
            "schema_arguments_not_object",
        )

    def test_structured_observation_accepts_real_eleven_digit_product_id(self):
        asin = "35842622441"
        observation = (
            "[SHOPPING_OBSERVATION_V2]\n"
            "1|35842622441|158.0|泰国乳胶枕\n"
            '可点击的按钮: ["back to search", "35842622441"]'
        )
        self.assertEqual(product_ids(observation), [asin])
        self.assertIsNone(
            action_reject_reason("open_product", {"asin": asin}, observation)
        )

    def test_unrelated_long_number_is_not_treated_as_product(self):
        observation = (
            "商品描述包含电话号码 13800138000 和价格 12345678"
            '\n\n可点击的按钮: ["back to search"]'
        )
        self.assertEqual(product_ids(observation), [])

    def test_search_guard_allows_repeated_and_synonymous_rewrites(self):
        observation = "搜索功能是否可用: True"
        self.assertIsNone(
            action_reject_reason(
                "search_products",
                {"query": "黑色 电饭煲"},
                observation,
                evaluation_extensions=True,
            )
        )
        self.assertIsNone(
            action_reject_reason(
                "search_products",
                {"query": "黑色电饭煲 5升"},
                observation,
                evaluation_extensions=True,
            )
        )

    def test_late_phase_guard_does_not_block_page_legal_actions(self):
        self.assertIsNone(
            action_reject_reason(
                "search_products",
                {"query": "新商品"},
                "搜索功能是否可用: True",
                step_count=35,
                evaluation_extensions=True,
            )
        )
        self.assertIsNone(
            action_reject_reason(
                "back_to_search",
                {},
                '可点击的按钮: ["Back to Search"]',
                step_count=40,
                evaluation_extensions=True,
            )
        )

    def test_full_candidate_memory_does_not_block_page_legal_exploration(self):
        memory = new_candidate_memory(max_entries=4)
        for index in range(1, 5):
            update_candidate_memory(
                memory,
                {
                    "observation_version": "shopping-observation-v2",
                    "page_type": "product_detail",
                    "product": {
                        "asin": f"{index:012d}",
                        "title": f"候选{index}",
                        "brand": "品牌",
                        "category": "类目",
                        "price": 100 + index,
                    },
                    "selected_options": {},
                },
                step_count=index,
            )

        for name, arguments, observation in (
            ("search_products", {"query": "第四个候选"}, "搜索功能是否可用: True"),
            (
                "open_product",
                {"asin": "000000000004"},
                '1|000000000004|100|品牌|类目|属性|候选4\n可点击的按钮: ["000000000004"]',
            ),
            ("next_page", {}, '可点击的按钮: ["Next >"]'),
            ("prev_page", {}, '可点击的按钮: ["< Prev"]'),
            ("back_to_search", {}, '可点击的按钮: ["Back to Search"]'),
        ):
            self.assertIsNone(
                action_reject_reason(
                    name,
                    arguments,
                    observation,
                    tool_schemas=EVALUATION_TOOL_SCHEMAS,
                    candidate_memory=memory,
                    evaluation_extensions=True,
                )
            )

if __name__ == "__main__":
    unittest.main()
