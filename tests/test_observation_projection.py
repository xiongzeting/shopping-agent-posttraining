"""Observation projection preserves actionable state and bounds visible tokens."""

import json
import unittest

from shopping_grpo.environment.actions import (
    action_reject_reason,
    clickable_buttons,
    product_ids,
)
from shopping_grpo.environment.candidate_memory import (
    CANDIDATE_MEMORY_START,
    detach_candidate_memory,
    new_candidate_memory,
    update_candidate_memory,
)
from shopping_grpo.environment.projection import (
    ObservationProjectionError,
    TRUNCATION_MARKER,
    project_observation,
)
from shopping_grpo.environment.observation import (
    LOOP_RECOVERY_NOTICE_PREFIX,
    add_step_budget_notice,
    render_structured_observation,
)


def search_page(product_count=12, page=1):
    segments = [
        "Instruction:",
        "find a useful product",
        "Back to Search",
        f"Page {page} (Total results: 40)",
        "Next >",
    ]
    buttons = ["back to search", "next >"]
    for index in range(product_count):
        asin = f"{100000000000 + index}"
        segments.extend([asin, f"product title {index} " + "x" * 40, f"{index + 1}.0"])
        buttons.append(asin)
    return (
        " [SEP] ".join(segments)
        + "\n\n搜索功能是否可用: False"
        + "\n\n可点击的按钮: "
        + json.dumps(buttons, ensure_ascii=False)
    )


class ObservationProjectionTest(unittest.TestCase):
    def test_search_projection_preserves_every_current_page_product(self):
        raw = search_page(product_count=20)
        visible, meta = project_observation(
            "search_products",
            raw,
            parameters={"query": "useful product"},
            count_tokens=len,
            token_budget=1200,
            search_top_k=20,
        )

        self.assertLessEqual(len(visible), 1200)
        self.assertTrue(meta.truncated)
        self.assertTrue(meta.critical_footer_preserved)
        self.assertIn(TRUNCATION_MARKER, visible)
        self.assertEqual(
            {button.casefold() for button in clickable_buttons(visible) if not button.isdigit()},
            {"back to search", "next >"},
        )
        self.assertEqual(
            set(product_ids(visible)),
            {button for button in clickable_buttons(visible) if button.isdigit()},
        )
        self.assertEqual(product_ids(visible), product_ids(raw))

    def test_guard_accepts_last_product_instead_of_creating_blind_spot(self):
        raw = search_page(product_count=20)
        visible, _ = project_observation(
            "search_products",
            raw,
            parameters={"query": "useful product"},
            count_tokens=len,
            token_budget=1200,
            search_top_k=20,
        )
        last_asin = product_ids(raw)[-1]

        self.assertIsNone(
            action_reject_reason("open_product", {"asin": last_asin}, visible)
        )

    def test_second_environment_page_preserves_products_21_through_40(self):
        raw = search_page(product_count=20, page=2)
        visible, _ = project_observation(
            "next_page",
            raw,
            count_tokens=len,
            token_budget=1200,
            search_top_k=20,
        )

        self.assertIn("page: Page 2", visible)
        self.assertEqual(product_ids(visible), product_ids(raw))

    def test_capacity_mismatch_fails_instead_of_silently_dropping_products(self):
        raw = search_page(product_count=20)
        with self.assertRaisesRegex(ObservationProjectionError, "page capacity"):
            project_observation(
                "search_products",
                raw,
                count_tokens=len,
                token_budget=1200,
                search_top_k=10,
            )

    def test_short_product_page_is_identity_projection(self):
        raw = (
            "Product [SEP] price: 20 [SEP] Buy Now"
            "\n\n搜索功能是否可用: False"
            '\n\n可点击的按钮: ["back to search", "buy now"]'
        )
        visible, meta = project_observation(
            "open_product",
            raw,
            count_tokens=len,
            token_budget=448,
        )

        self.assertEqual(visible, raw)
        self.assertFalse(meta.truncated)

    def test_generic_projection_keeps_complete_footer(self):
        raw = (
            "Description " + "detail " * 200 + "TAIL_SPECIFICATION"
            + "\n\n搜索功能是否可用: False"
            + '\n\n可点击的按钮: ["back to search", "< prev"]'
        )
        visible, meta = project_observation(
            "view_description",
            raw,
            count_tokens=len,
            generic_token_budget=300,
        )

        self.assertLessEqual(len(visible), 300)
        self.assertEqual(
            clickable_buttons(visible),
            ["back to search", "< prev"],
        )
        self.assertIn("TAIL_SPECIFICATION", visible)
        self.assertTrue(meta.critical_footer_preserved)

    def test_projection_preserves_step_budget_notice_within_budget(self):
        raw = (
            "Description " + "detail " * 200 + "TAIL_SPECIFICATION"
            + "\n\n搜索功能是否可用: False"
            + '\n可点击的按钮: ["back to search", "< prev"]'
        )
        raw = add_step_budget_notice(raw, step_count=35, max_steps=45)
        visible, meta = project_observation(
            "view_description",
            raw,
            count_tokens=len,
            generic_token_budget=300,
        )

        self.assertLessEqual(len(visible), 300)
        self.assertIn("步数提醒: 已执行 35/45 步，仅剩 10 步", visible)
        self.assertIn("请开始收敛", visible)
        self.assertIn("当前可调用工具仍以最新页面实际暴露的列表为准", visible)
        self.assertNotIn("绝对禁止新搜索、翻页和打开未核验商品", visible)
        self.assertEqual(
            clickable_buttons(visible),
            ["back to search", "< prev"],
        )
        self.assertTrue(meta.critical_footer_preserved)

    def test_projection_preserves_loop_recovery_notice_within_budget(self):
        raw = (
            "Description " + "detail " * 200 + "TAIL_SPECIFICATION"
            + "\n\n\u641c\u7d22\u529f\u80fd\u662f\u5426\u53ef\u7528: False"
            + '\n\u53ef\u70b9\u51fb\u7684\u6309\u94ae: ["back to search", "< prev"]'
        )
        raw = add_step_budget_notice(
            raw,
            step_count=20,
            max_steps=45,
            no_progress_steps=3,
        )
        visible, meta = project_observation(
            "view_description",
            raw,
            count_tokens=len,
            generic_token_budget=360,
        )

        self.assertLessEqual(len(visible), 360)
        self.assertIn(LOOP_RECOVERY_NOTICE_PREFIX, visible)
        self.assertEqual(
            clickable_buttons(visible),
            ["back to search", "< prev"],
        )
        self.assertTrue(meta.critical_footer_preserved)

    def test_long_page_without_action_footer_fails_closed(self):
        with self.assertRaisesRegex(ObservationProjectionError, "action footer"):
            project_observation(
                "unknown",
                "x" * 1000,
                count_tokens=len,
                generic_token_budget=128,
            )

    def test_structured_search_projection_preserves_all_twenty_products(self):
        products = [
            {
                "rank": index,
                "asin": f"{index:012d}",
                "title": "很长的商品标题" * 20,
                "brand": "品牌",
                "category": "类目",
                "price": index,
                "key_attributes": ["属性"],
            }
            for index in range(1, 21)
        ]
        raw = render_structured_observation(
            {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_results",
                "search_available": False,
                "actions": [
                    "back to search",
                    "next >",
                    *[product["asin"] for product in products],
                ],
                "query": "商品",
                "normalized_query": "商品",
                "page": 1,
                "total_pages": 2,
                "total_results": 40,
                "rank_start": 1,
                "rank_end": 20,
                "products": products,
            }
        )
        visible, _ = project_observation(
            "search_products",
            raw,
            count_tokens=len,
            token_budget=1400,
            search_top_k=20,
        )
        self.assertEqual(product_ids(visible), product_ids(raw))
        self.assertLessEqual(len(visible), 1400)

    def test_structured_projection_preserves_mixed_catalog_id_lengths(self):
        asins = ["12345678", "123456789", "1234567890", "35842622441", "123456789012"]
        products = [
            {
                "rank": index,
                "asin": asin,
                "title": "很长的商品标题" * 20,
                "brand": "品牌",
                "category": "类目",
                "price": index,
                "key_attributes": ["属性"],
            }
            for index, asin in enumerate(asins, start=1)
        ]
        raw = render_structured_observation(
            {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_results",
                "search_available": False,
                "actions": ["back to search", *asins],
                "query": "商品",
                "normalized_query": "商品",
                "page": 1,
                "total_pages": 1,
                "total_results": len(products),
                "rank_start": 1,
                "rank_end": len(products),
                "products": products,
            }
        )
        visible, _ = project_observation(
            "search_products",
            raw,
            count_tokens=len,
            token_budget=700,
            search_top_k=20,
        )

        self.assertEqual(product_ids(visible), asins)

    def test_long_search_projection_uses_additive_candidate_memory_budget(self):
        memory = new_candidate_memory()
        for index in range(1, 7):
            update_candidate_memory(
                memory,
                {
                    "observation_version": "shopping-observation-v2",
                    "page_type": "product_detail",
                    "product": {
                        "asin": f"{900000000000 + index}",
                        "title": "历史候选长标题" * 20,
                        "brand": "历史品牌",
                        "category": "历史类目",
                        "price": index * 100,
                        "key_attributes": ["历史属性" * 10],
                        "features": ["历史功能" * 10],
                        "attributes": ["历史规格" * 10],
                    },
                    "selected_options": {},
                },
                step_count=index,
            )
        products = [
            {
                "rank": index,
                "asin": f"{index:012d}",
                "title": "当前搜索页超长商品标题" * 20,
                "brand": "当前品牌",
                "category": "当前类目",
                "price": index,
                "key_attributes": ["当前属性"],
            }
            for index in range(1, 21)
        ]
        raw = render_structured_observation(
            {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_results",
                "search_available": False,
                "actions": ["back to search", "next >", *[p["asin"] for p in products]],
                "query": "商品",
                "normalized_query": "商品",
                "page": 1,
                "total_pages": 2,
                "total_results": 40,
                "rank_start": 1,
                "rank_end": 20,
                "products": products,
            },
            candidate_memory=memory,
            step_count=5,
        )
        raw = add_step_budget_notice(
            raw,
            step_count=36,
            max_steps=45,
            no_progress_steps=3,
        )

        visible, meta = project_observation(
            "search_products",
            raw,
            count_tokens=len,
            token_budget=3200,
            search_top_k=20,
        )

        visible_page, visible_memory = detach_candidate_memory(visible)
        self.assertLessEqual(len(visible_page), 3200)
        self.assertLessEqual(len(visible), 3200 + 1024)
        self.assertTrue(meta.truncated)
        self.assertEqual(meta.page_token_budget, 3200)
        self.assertEqual(meta.candidate_memory_token_budget, 1024)
        self.assertEqual(meta.candidate_memory_tokens, len(visible_memory))
        self.assertIn(CANDIDATE_MEMORY_START, visible)
        self.assertIn("候选记忆提示:", visible_memory)
        self.assertIn("后续商品仍可正常搜索和核验", visible_memory)
        self.assertLess(visible.index("page_type:"), visible.index("步数提醒:"))
        self.assertLess(visible.index("步数提醒:"), visible.index("query:"))
        self.assertLess(visible.index("query:"), visible.index(CANDIDATE_MEMORY_START))
        self.assertLess(
            visible.index(CANDIDATE_MEMORY_START),
            visible.index("可点击的按钮:"),
        )
        self.assertEqual(product_ids(visible), [p["asin"] for p in products])
        self.assertEqual(
            [button for button in clickable_buttons(visible) if button.isdigit()],
            [p["asin"] for p in products],
        )
        self.assertNotIn("900000000001", product_ids(visible))

    def test_candidate_memory_budget_failure_is_explicit(self):
        memory = new_candidate_memory()
        for index in range(1, 7):
            update_candidate_memory(
                memory,
                {
                    "observation_version": "shopping-observation-v2",
                    "page_type": "product_detail",
                    "product": {
                        "asin": f"{800000000000 + index}",
                        "title": "x" * 200,
                        "brand": "b" * 100,
                        "category": "c" * 100,
                        "price": index,
                        "key_attributes": ["k" * 100] * 10,
                        "features": ["f" * 100] * 10,
                        "attributes": ["a" * 100] * 10,
                    },
                    "selected_options": {},
                },
                step_count=index,
            )
        raw = render_structured_observation(
            search_state := {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_results",
                "search_available": False,
                "actions": ["back to search", "123456789012"],
                "query": "商品",
                "normalized_query": "商品",
                "page": 1,
                "total_pages": 1,
                "total_results": 1,
                "rank_start": 1,
                "rank_end": 1,
                "products": [
                    {
                        "rank": 1,
                        "asin": "123456789012",
                        "title": "当前商品" * 50,
                        "brand": "品牌",
                        "category": "类目",
                        "price": 1,
                        "key_attributes": ["属性"],
                    }
                ],
            },
            candidate_memory=memory,
            step_count=5,
        )
        self.assertEqual(search_state["page_type"], "search_results")

        with self.assertRaisesRegex(
            ObservationProjectionError,
            "candidate memory exceeds its additive",
        ):
            project_observation(
                "search_products",
                raw,
                count_tokens=len,
                token_budget=512,
                candidate_memory_token_budget=64,
                search_top_k=20,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
