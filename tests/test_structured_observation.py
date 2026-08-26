import unittest

from shopping_grpo.environment.actions import action_reject_reason, product_ids
from shopping_grpo.environment.candidate_memory import (
    CANDIDATE_MEMORY_END,
    CANDIDATE_MEMORY_START,
    detach_candidate_memory,
    new_candidate_memory,
    update_candidate_memory,
)
from shopping_grpo.environment.observation import (
    CANDIDATE_CONVERGENCE_NOTICE_PREFIX,
    LOOP_RECOVERY_NOTICE_PREFIX,
    StructuredObservationError,
    add_step_budget_notice,
    render_structured_observation,
)


def search_state(count=20):
    products = [
        {
            "rank": index,
            "asin": f"{index:012d}",
            "title": f"商品 {index}",
            "brand": "品牌",
            "category": "类目",
            "price": index,
            "key_attributes": ["属性"],
        }
        for index in range(1, count + 1)
    ]
    return {
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
        "rank_end": count,
        "products": products,
    }


def product_state():
    option_id = "opt_0123456789abcdef"
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "product_detail",
        "search_available": False,
        "actions": ["< prev", option_id, "buy now"],
        "product": {
            "asin": "123456789012",
            "title": "黑色测试商品",
            "brand": "测试品牌",
            "category": "测试商品",
            "price": 120,
            "key_attributes": ["黑色"],
            "features": ["支持防水", "简约设计"],
            "attributes": ["高度66cm", "咖啡色"],
        },
        "selected_options": {
            "颜色": {"option_id": option_id, "label": "黑色"}
        },
        "available_options": {
            "颜色": [{"option_id": option_id, "label": "黑色"}]
        },
    }


class StructuredObservationTest(unittest.TestCase):
    def test_all_twenty_products_are_visible_and_guard_actionable(self):
        visible = render_structured_observation(search_state())
        self.assertEqual(len(product_ids(visible)), 20)
        self.assertIsNone(
            action_reject_reason(
                "open_product",
                {"asin": "000000000020"},
                visible,
            )
        )

    def test_step_budget_notice_starts_at_35_without_hiding_actions(self):
        visible = render_structured_observation(search_state(1))

        self.assertEqual(
            add_step_budget_notice(visible, step_count=29, max_steps=45),
            visible,
        )
        first = add_step_budget_notice(visible, step_count=30, max_steps=45)
        self.assertEqual(first, visible)
        warned = add_step_budget_notice(visible, step_count=35, max_steps=45)
        self.assertIn("步数提醒: 已执行 35/45 步，仅剩 10 步", warned)
        self.assertIn("请开始收敛", warned)
        self.assertIn("利用已有核验信息", warned)
        self.assertIn("当前可调用工具仍以最新页面实际暴露的列表为准", warned)
        self.assertNotIn("绝对禁止", warned)
        self.assertIn(
            "已执行 36/45 步，仅剩 9 步",
            add_step_budget_notice(visible, step_count=36, max_steps=45),
        )
        step40 = add_step_budget_notice(visible, step_count=40, max_steps=45)
        self.assertIn("强烈建议立即比较", step40)
        self.assertIn("若仍需探索", step40)
        self.assertLess(
            warned.index("步数提醒:"),
            warned.index("可点击的按钮:"),
        )
        self.assertEqual(warned.splitlines()[0], "[SHOPPING_OBSERVATION_V2]")
        self.assertEqual(warned.splitlines()[1], "page_type: search_results")
        self.assertTrue(warned.splitlines()[2].startswith("步数提醒:"))

    def test_terminal_observation_never_gets_step_budget_notice(self):
        terminal = (
            "[SHOPPING_OBSERVATION_V2]\npage_type: terminal"
            "\n\n搜索功能是否可用: False"
            "\n可点击的按钮: []"
        )
        self.assertEqual(
            add_step_budget_notice(terminal, step_count=40, max_steps=45),
            terminal,
        )

    def test_full_candidate_memory_is_neutral_and_keeps_loop_notice(self):
        memory = new_candidate_memory(max_entries=3)
        for index in range(1, 4):
            state = product_state()
            state["product"] = dict(state["product"], asin=f"{index:012d}")
            update_candidate_memory(memory, state, step_count=index)
        visible = render_structured_observation(
            search_state(1),
            candidate_memory=memory,
        )

        warned = add_step_budget_notice(
            visible,
            step_count=12,
            max_steps=45,
            no_progress_steps=5,
            candidate_count=3,
        )

        self.assertIn(CANDIDATE_CONVERGENCE_NOTICE_PREFIX, warned)
        _, memory_block = detach_candidate_memory(warned)
        self.assertIn("后续商品仍可正常搜索和核验", memory_block)
        self.assertIn("不会写入或替换本候选记忆", memory_block)
        self.assertNotIn("绝对禁止继续搜索", warned)
        self.assertIn(LOOP_RECOVERY_NOTICE_PREFIX, warned)
        self.assertEqual(warned.count(CANDIDATE_CONVERGENCE_NOTICE_PREFIX), 1)
        self.assertNotIn(CANDIDATE_CONVERGENCE_NOTICE_PREFIX, warned.split(CANDIDATE_MEMORY_START)[0])

    def test_loop_recovery_notice_starts_after_three_no_progress_steps(self):
        visible = render_structured_observation(search_state(1))

        self.assertEqual(
            add_step_budget_notice(
                visible,
                step_count=10,
                max_steps=45,
                no_progress_steps=2,
            ),
            visible,
        )
        warned = add_step_budget_notice(
            visible,
            step_count=10,
            max_steps=45,
            no_progress_steps=3,
        )
        warned_twice = add_step_budget_notice(
            warned,
            step_count=10,
            max_steps=45,
            no_progress_steps=3,
        )

        self.assertEqual(warned.count(LOOP_RECOVERY_NOTICE_PREFIX), 1)
        self.assertEqual(warned_twice.count(LOOP_RECOVERY_NOTICE_PREFIX), 1)
        self.assertLess(
            warned.index(LOOP_RECOVERY_NOTICE_PREFIX),
            warned.index("\u53ef\u70b9\u51fb\u7684\u6309\u94ae:"),
        )
        self.assertEqual(warned.splitlines()[0], "[SHOPPING_OBSERVATION_V2]")
        self.assertEqual(warned.splitlines()[1], "page_type: search_results")
        self.assertTrue(warned.splitlines()[2].startswith(LOOP_RECOVERY_NOTICE_PREFIX))
        self.assertTrue(warned.endswith(visible.splitlines()[-1]))
        self.assertIn("现在必须立即改变策略", warned)
        self.assertIn("已进入 Loop 高风险状态", warned)

        strong = add_step_budget_notice(
            visible,
            step_count=10,
            max_steps=45,
            no_progress_steps=5,
        )
        self.assertIn("现在必须立即改变策略", strong)

    def test_terminal_observation_never_gets_loop_recovery_notice(self):
        terminal = render_structured_observation(
            {
                "observation_version": "shopping-observation-v2",
                "page_type": "terminal",
                "search_available": False,
                "actions": [],
            }
        )

        self.assertEqual(
            add_step_budget_notice(
                terminal,
                step_count=40,
                max_steps=45,
                no_progress_steps=6,
            ),
            terminal,
        )

    def test_action_asin_mismatch_fails_closed(self):
        state = search_state(2)
        state["actions"].remove("000000000002")
        with self.assertRaisesRegex(StructuredObservationError, "actionable"):
            render_structured_observation(state)

    def test_hidden_reward_payload_is_rejected(self):
        state = search_state(1)
        state["reward"] = 1.0
        with self.assertRaisesRegex(StructuredObservationError, "forbidden"):
            render_structured_observation(state)

    def test_candidate_judgment_payload_is_rejected(self):
        state = search_state(1)
        state["candidate_state"] = {
            "current_candidate": {"fully_satisfied": True},
            "best_candidate": {"public_match_score": 1.0},
        }
        with self.assertRaisesRegex(StructuredObservationError, "forbidden"):
            render_structured_observation(state)

    def test_product_renders_stable_options_without_candidate_judgment(self):
        visible = render_structured_observation(product_state())

        self.assertIn('"option_id": "opt_0123456789abcdef"', visible)
        self.assertIn('"label": "黑色"', visible)
        self.assertIn("features: 支持防水, 简约设计", visible)
        self.assertIn("attributes: 高度66cm, 咖啡色", visible)
        self.assertIn("规格状态: 1/1 个规格轴已选择；当前完整价格: 120", visible)
        for forbidden in (
            "current_candidate:",
            "best_candidate:",
            "missing_conditions",
            "satisfied_conditions",
            "fully_satisfied",
            "public_match_score",
        ):
            self.assertNotIn(forbidden, visible)

    def test_product_omits_empty_optional_evidence_fields(self):
        state = product_state()
        state["product"]["features"] = []
        state["product"]["attributes"] = []

        visible = render_structured_observation(state)

        self.assertNotIn("\nfeatures:", visible)
        self.assertNotIn("\nattributes:", visible)

    def test_candidate_memory_follows_products_across_pages(self):
        memory = new_candidate_memory()
        first = product_state()
        first["product"]["asin"] = "111111111111"
        second = product_state()
        second["product"]["asin"] = "222222222222"

        first_visible = render_structured_observation(
            first,
            candidate_memory=memory,
            step_count=1,
        )
        second_visible = render_structured_observation(
            second,
            candidate_memory=memory,
            step_count=2,
        )
        search_visible = render_structured_observation(
            search_state(1),
            candidate_memory=memory,
            step_count=3,
        )

        self.assertIn(CANDIDATE_MEMORY_START, first_visible)
        _, first_memory = detach_candidate_memory(first_visible)
        self.assertIn("当前详情候选：C1｜ASIN 111111111111", first_memory)
        self.assertIn(CANDIDATE_MEMORY_START, second_visible)
        _, second_memory = detach_candidate_memory(second_visible)
        self.assertIn("111111111111", second_memory)
        self.assertIn("当前详情候选：C2｜ASIN 222222222222", second_memory)
        self.assertNotIn("C2｜222222222222｜@", second_memory)
        self.assertIn("111111111111", search_visible)
        self.assertIn("222222222222", search_visible)
        self.assertLess(
            search_visible.index(CANDIDATE_MEMORY_END),
            search_visible.index("搜索功能是否可用:"),
        )
        self.assertTrue(search_visible.startswith("[SHOPPING_OBSERVATION_V2]"))
        self.assertTrue(search_visible.endswith(render_structured_observation(search_state(1)).splitlines()[-1]))

        warned_search = add_step_budget_notice(
            search_visible,
            step_count=36,
            max_steps=45,
            no_progress_steps=3,
        )
        self.assertLess(warned_search.index("page_type:"), warned_search.index("步数提醒:"))
        self.assertLess(warned_search.index("步数提醒:"), warned_search.index("query:"))
        self.assertLess(warned_search.index("query:"), warned_search.index(CANDIDATE_MEMORY_START))
        self.assertLess(warned_search.index(CANDIDATE_MEMORY_END), warned_search.index("可点击的按钮:"))

    def test_candidate_memory_can_be_updated_without_being_projected(self):
        memory = new_candidate_memory(max_entries=4)

        visible = render_structured_observation(
            product_state(),
            candidate_memory=memory,
            step_count=1,
            show_candidate_memory=False,
        )

        self.assertEqual(len(memory["entries"]), 1)
        self.assertEqual(memory["entries"][0]["asin"], "123456789012")
        self.assertNotIn(CANDIDATE_MEMORY_START, visible)
        self.assertNotIn("C1", visible)

    def test_candidate_memory_does_not_change_current_page_buttons(self):
        memory = new_candidate_memory()
        detail = product_state()
        detail["product"]["asin"] = "111111111111"
        render_structured_observation(detail, candidate_memory=memory, step_count=1)

        visible = render_structured_observation(
            search_state(2),
            candidate_memory=memory,
            step_count=2,
        )

        self.assertEqual(product_ids(visible), ["000000000001", "000000000002"])
        self.assertEqual(
            action_reject_reason(
                "open_product",
                {"asin": "111111111111"},
                visible,
            ),
            "click_not_in_previous_observation",
        )

    def test_candidate_memory_records_public_search_location(self):
        memory = new_candidate_memory()
        source_search = search_state(1)
        detail = product_state()
        detail["product"]["asin"] = "000000000001"

        render_structured_observation(
            source_search,
            candidate_memory=memory,
            step_count=1,
        )
        render_structured_observation(
            detail,
            candidate_memory=memory,
            step_count=2,
        )
        visible = render_structured_observation(
            source_search,
            candidate_memory=memory,
            step_count=3,
        )

        self.assertIn("@商品/P1/R1", visible)
        self.assertIn("C1｜000000000001｜", visible)

    def test_product_rejects_option_id_not_actionable_on_current_page(self):
        state = product_state()
        state["actions"].remove("opt_0123456789abcdef")

        with self.assertRaisesRegex(StructuredObservationError, "actionable option IDs"):
            render_structured_observation(state)

    def test_catalog_product_ids_from_eight_through_twelve_digits_are_supported(self):
        asins = ["12345678", "123456789", "1234567890", "35842622441", "123456789012"]
        state = search_state(len(asins))
        for product, asin in zip(state["products"], asins, strict=True):
            product["asin"] = asin
        state["actions"] = ["back to search", "next >", *asins]

        visible = render_structured_observation(state)

        self.assertEqual(product_ids(visible), asins)
        for asin in asins:
            self.assertIsNone(
                action_reject_reason("open_product", {"asin": asin}, visible)
            )

    def test_product_ids_outside_catalog_length_are_rejected(self):
        for invalid in ("1234567", "1234567890123"):
            with self.subTest(invalid=invalid):
                state = search_state(1)
                state["products"][0]["asin"] = invalid
                state["actions"] = ["back to search", "next >", invalid]
                with self.assertRaisesRegex(StructuredObservationError, "invalid"):
                    render_structured_observation(state)


if __name__ == "__main__":
    unittest.main()
