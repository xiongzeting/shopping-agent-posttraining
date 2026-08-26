import hashlib
import unittest

from web_agent_site.engine.comparators import (
    FAIL,
    PASS,
    UNVERIFIABLE,
    compare_brand,
    compare_core_functions,
    compare_model,
    compare_numeric_spec,
)
from web_agent_site.engine.goal import align_goal_reward_features
from web_agent_site.engine.reward import (
    DEFAULT_REWARDS,
    calculate_step_penalty,
    evaluate_abstain,
    evaluate_candidate_eligibility,
    evaluate_purchase,
    fixed_termination,
)
from web_agent_site.engine.reward_features import (
    _constraint_semantics,
    _query_semantic_segments,
    compile_reward_features,
)
from web_agent_site.engine.variant_price import (
    compare_required_options,
    resolve_variant_price,
)


def product(asin="111111111111", *, model="A20", attributes=None):
    return {
        "asin": asin,
        "title": f"石头 {model} 智能洗地机",
        "brand": "石头",
        "shop_name": "石头旗舰店",
        "category": "家电›清洁电器›洗地机",
        "attribute": attributes or ["智能洗地", "热洗"],
        "pricing": [1999],
        "customization_options": {
            "颜色分类": [
                {"value": "白色", "price": 1999},
                {"value": "黑色", "price": 1999},
            ],
            "尺码": [
                {"value": "L", "price": 1899},
                {"value": "XL", "price": 1999},
            ],
        },
    }


INSTRUCTION = {
    "instruction": "购买支持热洗的白色 XL 洗地机，预算2200元",
    "attributes": ["洗地", "热洗"],
    "instruction_options": ["白色", "XL"],
}


def goal(target=None):
    target = target or product()
    return {
        "asin": target["asin"],
        "category": target["category"],
        "price_upper": 2200,
        "instruction_text": INSTRUCTION["instruction"],
        **compile_reward_features(INSTRUCTION, target),
    }


def align_manual_goal(task_goal):
    instruction_text = task_goal["instruction_text"]
    instruction_hash = hashlib.sha256(instruction_text.encode("utf-8")).hexdigest()
    task_goal["instruction_sha256"] = instruction_hash
    task_goal["query_constraint_contract"] = {
        "schema_version": "shopping-query-constraints-v1",
        "instruction_sha256": instruction_hash,
        "constraints": [],
    }
    return task_goal


class RewardV3Test(unittest.TestCase):
    def test_hidden_annotation_cannot_add_a_requirement_to_visible_instruction(self):
        target = product(attributes=["双枪", "家用", "无涂层"])
        instruction = {
            "instruction": "找一款家用无涂层洗地机，预算2200元以内。",
            "attributes": ["双枪", "家用", "无涂层"],
            "instruction_options": [],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }

        self.assertNotIn("双枪", task_goal["expected_core_functions"])
        result = evaluate_purchase(target, task_goal, selected_options={}, price=1999)
        self.assertEqual(result.reward_type, "gold_purchase")

    def test_optional_option_language_does_not_block_a_valid_alternative(self):
        target = product(attributes=[])
        instruction = {
            "instruction": "购买一台洗地机，红色也可以，预算2200元以内。",
            "attributes": [],
            "instruction_options": ["红色"],
        }
        target["customization_options"]["颜色分类"] = [
            {"value": "红色", "price": 1999},
            {"value": "黑色", "price": 1999},
        ]
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=[])
        alternative["customization_options"]["颜色分类"] = [
            {"value": "黑色", "price": 1999},
        ]

        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={"颜色分类": "黑色"},
            price=1999,
        )

        self.assertEqual(result.reward_type, "valid_alternative_purchase")
        option_constraint = next(
            item
            for item in task_goal["query_constraint_contract"]["constraints"]
            if item["constraint_type"] == "option"
        )
        self.assertEqual(option_constraint["role"], "query_preference_reference")
        self.assertEqual(option_constraint["query_evidence"], [])
        self.assertEqual(option_constraint["optional_query_evidence"], ["红色"])

    def test_bare_capability_can_remain_required(self):
        target = product(attributes=["热洗"])
        instruction = {
            "instruction": "购买一台可以用于热洗的洗地机，预算2200元以内。",
            "attributes": ["热洗"],
            "instruction_options": [],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=["常温清洁"])

        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={},
            price=1999,
        )

        self.assertEqual(task_goal["expected_core_functions"], ["热洗"])
        self.assertEqual(result.reward_type, "wrong_purchase")

    def test_audit_only_hard_clause_cannot_turn_alternative_into_wrong_purchase(self):
        target = product(attributes=[])
        instruction = {
            "instruction": "购买一台洗地机，不要人工合成的那种。",
            "attributes": [],
            "instruction_options": [],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=[])

        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={},
            price=1999,
        )
        detail = result.to_dict()
        audit_clause = next(
            item
            for item in detail["constraint_audit_results"]
            if item.get("constraint_type") == "query_clause"
        )

        self.assertEqual(audit_clause["strength"], "hard")
        self.assertEqual(audit_clause["enforcement"], "audit_only")
        self.assertEqual(audit_clause["status"], UNVERIFIABLE)
        self.assertEqual(result.reward_type, "valid_alternative_purchase")
        self.assertFalse(
            any(
                item.get("constraint_type") == "query_clause"
                for item in detail["constraint_results"]
            )
        )

    def test_additive_and_soft_preferences_do_not_become_requirements(self):
        target = product(attributes=["热洗", "加宽"])
        instruction = {
            "instruction": "购买一台洗地机，也可以用于热洗，最好是加宽设计。",
            "attributes": ["热洗", "加宽"],
            "instruction_options": [],
        }
        features = compile_reward_features(instruction, target)

        self.assertEqual(features["expected_core_functions"], [])
        self.assertEqual(
            features["optional_core_function_preferences"],
            ["热洗", "加宽"],
        )
        preference_constraints = [
            item
            for item in features["query_constraint_contract"]["constraints"]
            if item.get("role") == "query_preference_reference"
        ]
        self.assertEqual(
            {item["expected"] for item in preference_constraints},
            {"热洗", "加宽"},
        )

    def test_required_evidence_wins_when_same_value_is_also_a_preference(self):
        target = product(attributes=["热洗"])
        instruction = {
            "instruction": "必须支持热洗，外观最好也标明热洗模式。",
            "attributes": ["热洗"],
            "instruction_options": [],
        }
        features = compile_reward_features(instruction, target)

        self.assertEqual(features["expected_core_functions"], ["热洗"])
        self.assertEqual(features["optional_core_function_preferences"], [])

    def test_mixed_hard_and_soft_constraints_without_punctuation_are_isolated(self):
        instruction_text = "绝对不要5升以上最好要黑色的电饭煲"

        capacity = _constraint_semantics(
            instruction_text=instruction_text,
            role="matching_dimension",
            constraint_type="core_function",
            expected="5升以上",
            query_evidence=["5升以上"],
        )
        color = _constraint_semantics(
            instruction_text=instruction_text,
            role="query_preference_reference",
            constraint_type="option",
            expected="黑色",
            optional_query_evidence=["黑色"],
        )

        self.assertEqual((capacity["strength"], capacity["polarity"]), ("hard", "forbid"))
        self.assertNotIn("最好", capacity["query_quote"])
        self.assertEqual((color["strength"], color["polarity"]), ("soft", "prefer"))
        self.assertTrue(color["query_quote"].startswith("最好"))

    def test_reversed_mixed_soft_and_hard_constraints_are_isolated(self):
        instruction_text = "最好要黑色但绝对不要5升以上的电饭煲"

        color = _constraint_semantics(
            instruction_text=instruction_text,
            role="query_preference_reference",
            constraint_type="option",
            expected="黑色",
            optional_query_evidence=["黑色"],
        )
        capacity = _constraint_semantics(
            instruction_text=instruction_text,
            role="matching_dimension",
            constraint_type="core_function",
            expected="5升以上",
            query_evidence=["5升以上"],
        )

        self.assertEqual((color["strength"], color["polarity"]), ("soft", "prefer"))
        self.assertNotIn("绝对", color["query_quote"])
        self.assertEqual((capacity["strength"], capacity["polarity"]), ("hard", "forbid"))

    def test_soft_qualified_boundary_stays_in_one_semantic_scope(self):
        cases = {
            "尽量别超过760元": ["尽量别超过760元"],
            "预期价格不超过100元": ["预期价格不超过100元"],
            "价格在120元以内最好": ["价格在120元以内最好"],
        }
        for instruction_text, expected in cases.items():
            with self.subTest(instruction_text=instruction_text):
                self.assertEqual(
                    _query_semantic_segments(instruction_text),
                    expected,
                )

    def test_not_required_does_not_weaken_later_required_evidence(self):
        target = product(attributes=["高度调节", "扶手"])
        instruction = {
            "instruction": "座椅不需要带有高度调节功能但必须有扶手。",
            "attributes": ["高度调节", "扶手"],
            "instruction_options": [],
        }
        features = compile_reward_features(instruction, target)

        self.assertNotIn("高度调节", features["expected_core_functions"])
        self.assertIn("高度调节", features["optional_core_function_preferences"])
        self.assertIn("扶手", features["expected_core_functions"])
        handle_constraint = next(
            item
            for item in features["query_constraint_contract"]["constraints"]
            if item.get("expected") == "扶手"
        )
        self.assertEqual(handle_constraint["strength"], "hard")

    def test_ambiguous_final240_negative_patterns_are_classified_contextually(self):
        cases = {
            "要不用工具也能快速拆装的": ("hard", "require"),
            "安装时无需破坏原车线路就能改装": ("hard", "forbid"),
            "不需要有售后": ("ignore", "indifferent"),
            "不用参与活动": ("ignore", "indifferent"),
            "不用减脂": ("ignore", "indifferent"),
            "不用花里胡哨的": ("hard", "forbid"),
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                semantics = _constraint_semantics(
                    instruction_text=query,
                    role="query_semantic_audit",
                    constraint_type="query_clause",
                    expected=query,
                    query_evidence=[query],
                )
                self.assertEqual(
                    (semantics["strength"], semantics["polarity"]),
                    expected,
                )

    def test_narrative_plan_is_not_a_product_requirement(self):
        target = product(attributes=["学生相机", "旅游"])
        instruction = {
            "instruction": "学生准备去旅游了，购买一台白色相机。",
            "attributes": ["旅游"],
            "instruction_options": ["白色"],
        }
        features = compile_reward_features(instruction, target)

        self.assertEqual(features["expected_core_functions"], [])
        self.assertEqual(features["optional_core_function_preferences"], [])

    def test_compound_requirement_can_be_verified_across_attributes(self):
        candidate = product(attributes=["白参菌", "专利"])

        self.assertEqual(
            compare_core_functions(["专利白参菌"], candidate)["status"],
            PASS,
        )

    def test_public_option_fragments_accept_visible_aliases_and_size_token(self):
        target = product(attributes=[])
        target["customization_options"] = {
            "颜色分类": [{"value": "深灰色", "price": 145}],
            "尺码": [{"value": "M", "price": 145}],
        }
        instruction = {
            "instruction": "购买深灰色M码洗地机，预算不超过160元。",
            "attributes": [],
            "instruction_options": ["深灰色", "M"],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=[])
        alternative["customization_options"] = {
            "颜色": [{"value": "F511【枫叶深灰】", "price": 145}],
            "尺码": [{"value": "M【推荐90-110斤】", "price": 145}],
        }
        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={
                "颜色": "F511【枫叶深灰】",
                "尺码": "M【推荐90-110斤】",
            },
            price=145,
        )

        self.assertEqual(result.reward_type, "valid_alternative_purchase")

    def test_expected_price_boundary_is_soft(self):
        target = product(attributes=["热洗"])
        instruction = {
            "instruction": "购买支持热洗的洗地机，预期价格在100元以内。",
            "attributes": ["热洗"],
            "instruction_options": [],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=["热洗"])
        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={},
            price=150,
        )
        price_result = next(
            item
            for item in result.to_dict()["constraint_results"]
            if item["constraint_type"] == "budget_upper"
        )

        self.assertEqual(price_result["strength"], "soft")
        self.assertEqual(price_result["status"], FAIL)
        self.assertEqual(result.reward_type, "partial_alternative_purchase")

    def test_hidden_model_token_cannot_match_price_digits(self):
        target = product(attributes=["平躺"])
        target["customization_options"] = {
            "型号": [{"value": "A30", "price": 2349}],
        }
        instruction = {
            "instruction": "购买支持平躺的洗地机，价格2300元到2400元。",
            "attributes": ["平躺"],
            "instruction_options": ["A30"],
        }
        features = compile_reward_features(instruction, target)
        option = next(
            item
            for item in features["query_constraint_contract"]["constraints"]
            if item["constraint_type"] == "option"
        )

        self.assertEqual(option["role"], "gold_variant_reference")
        self.assertEqual(option["query_evidence"], [])
        self.assertEqual(
            compare_core_functions(["平躺"], product(attributes=["躺平"]))["status"],
            PASS,
        )

    def test_composite_option_scores_only_public_required_components(self):
        target = product(attributes=[])
        target["customization_options"] = {
            "颜色分类": [
                {"value": "蓝色+推杆+赠品", "price": 1999},
            ]
        }
        instruction = {
            "instruction": "购买蓝色并带推杆的款式，赠品也可以有。",
            "attributes": [],
            "instruction_options": ["蓝色+推杆+赠品"],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=[])
        alternative["customization_options"] = {
            "颜色分类": [
                {"value": "蓝色+推杆", "price": 1999},
            ]
        }

        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={"颜色分类": "蓝色+推杆"},
            price=1999,
        )

        self.assertEqual(result.reward_type, "valid_alternative_purchase")
        option_constraint = next(
            item
            for item in task_goal["query_constraint_contract"]["constraints"]
            if item["constraint_type"] == "option"
        )
        self.assertEqual(option_constraint["strength"], "ignore")

    def test_alternative_option_can_express_requirement_on_a_different_axis(self):
        target = product(attributes=[])
        target["customization_options"] = {
            "魔方种类": [{"value": "三阶", "price": 108}],
        }
        instruction = {
            "instruction": "购买三阶魔方，预算120元以内。",
            "attributes": [],
            "instruction_options": ["三阶"],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=[])
        alternative["title"] = "竞赛磁力三阶魔方"
        alternative["customization_options"] = {
            "颜色分类": [{"value": "威龙V11-三阶魔方-磁力版", "price": 108}],
        }

        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={"颜色分类": "威龙V11-三阶魔方-磁力版"},
            price=108,
        )

        self.assertEqual(result.reward_type, "valid_alternative_purchase")

    def test_visible_product_text_cannot_override_conflicting_selected_quantity(self):
        target = product(attributes=[])
        target["title"] = "五罐装炼乳"
        target["customization_options"] = {
            "包装": [{"value": "5罐", "price": 100}],
        }
        instruction = {
            "instruction": "购买5罐装炼乳，预算100元以内。",
            "attributes": [],
            "instruction_options": ["5罐"],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=[])
        alternative["title"] = "炼乳 3罐/5罐可选"
        alternative["customization_options"] = {
            "数量": [
                {"value": "3罐", "price": 60},
                {"value": "5罐", "price": 100},
            ],
        }

        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={"数量": "3罐"},
            price=60,
        )

        self.assertEqual(result.reward_type, "reward_unverifiable")
        self.assertFalse(result.reward_valid)

    def test_explicit_purchase_price_recovers_stale_unverifiable_resolution(self):
        stale_resolution = {
            "status": UNVERIFIABLE,
            "price": None,
            "method": "effective_price_axis_unselected",
        }
        result = evaluate_purchase(
            product(),
            goal(),
            selected_options={"颜色分类": "白色", "尺码": "XL"},
            price_resolution=stale_resolution,
            price=1999,
        )

        self.assertEqual(result.reward_type, "gold_purchase")
        self.assertEqual(
            result.to_dict()["evidence"]["price_resolution"]["price"],
            1999,
        )

    def test_approximate_price_uses_twenty_percent_bounds_in_reward(self):
        target = product(attributes=[])
        instruction = {
            "instruction": "购买一台洗地机，价格20元左右。",
            "attributes": [],
            "instruction_options": [],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }

        for accepted in (16.0, 20.0, 24.0):
            with self.subTest(accepted=accepted):
                result = evaluate_purchase(
                    target,
                    task_goal,
                    selected_options={},
                    price=accepted,
                )
                self.assertEqual(result.reward_type, "gold_purchase")
        for rejected in (15.99, 24.01):
            with self.subTest(rejected=rejected):
                result = evaluate_purchase(
                    target,
                    task_goal,
                    selected_options={},
                    price=rejected,
                )
                self.assertEqual(result.reward_type, "gold_purchase")
                price_result = next(
                    item
                    for item in result.to_dict()["constraint_results"]
                    if item["constraint_type"] == "price_range"
                )
                self.assertEqual(price_result["strength"], "soft")
                self.assertEqual(price_result["status"], FAIL)

    def test_soft_qualified_price_boundary_produces_partial_alternative(self):
        target = product(attributes=["热洗"])
        instruction = {
            "instruction": "购买支持热洗的洗地机，价格最好在100元以下。",
            "attributes": ["热洗"],
            "instruction_options": [],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=["热洗"])
        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={},
            price=150,
        )
        detail = result.to_dict()
        price_result = next(
            item
            for item in detail["constraint_results"]
            if item["constraint_type"] == "budget_upper"
        )

        self.assertEqual(price_result["strength"], "soft")
        self.assertEqual(price_result["status"], FAIL)
        self.assertEqual(result.reward_type, "partial_alternative_purchase")
        self.assertEqual(result.reward, DEFAULT_REWARDS["partial_purchase_base"])
        self.assertFalse(detail["purchase_success"])

    def test_soft_qualified_price_boundary_keeps_satisfying_alternative_valid(self):
        target = product(attributes=["热洗"])
        instruction = {
            "instruction": "购买支持热洗的洗地机，尽量别超过100元。",
            "attributes": ["热洗"],
            "instruction_options": [],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=["热洗"])
        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={},
            price=90,
        )

        self.assertEqual(result.reward_type, "valid_alternative_purchase")
        self.assertEqual(result.reward, DEFAULT_REWARDS["valid_alternative_purchase"])

    def test_explicit_price_boundary_remains_hard(self):
        target = product(attributes=["热洗"])
        instruction = {
            "instruction": "购买支持热洗的洗地机，价格必须在100元以下。",
            "attributes": ["热洗"],
            "instruction_options": [],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        alternative = product("222222222222", attributes=["热洗"])
        result = evaluate_purchase(
            alternative,
            task_goal,
            selected_options={},
            price=150,
        )

        self.assertEqual(result.reward_type, "wrong_purchase")

    def test_one_approximate_component_does_not_soften_an_exact_composite_option(self):
        target = product(attributes=[])
        target["customization_options"] = {
            "尺码": [{"value": "180/84A/XL", "price": 499}],
        }
        instruction = {
            "instruction": "购买身高180，腰围84A左右的洗地机。",
            "attributes": [],
            "instruction_options": ["180/84A/XL"],
        }
        features = compile_reward_features(instruction, target)
        option_constraint = next(
            item
            for item in features["query_constraint_contract"]["constraints"]
            if item.get("constraint_type") == "option"
        )

        self.assertEqual(option_constraint["strength"], "hard")

    def test_step_penalty_boundaries_are_cumulative(self):
        expected = {
            14: 0.0,
            15: 0.0,
            16: -0.01,
            20: -0.05,
            21: -0.07,
            25: -0.15,
            26: -0.18,
            30: -0.30,
            31: -0.34,
            35: -0.50,
            36: -0.55,
            40: -0.75,
            41: -0.81,
            45: -1.05,
        }
        for steps, penalty in expected.items():
            with self.subTest(steps=steps):
                self.assertAlmostEqual(calculate_step_penalty(steps), penalty)

    def test_step_penalty_applies_to_every_valid_terminal_type(self):
        selected = {"颜色分类": "白色", "尺码": "XL"}
        gold = evaluate_purchase(
            product(), goal(), selected_options=selected, step_count=21
        )
        abstain = evaluate_abstain(
            effective_result_sets=2,
            opened_candidates=2,
            known_acceptable_candidates=0,
            step_count=21,
        )
        loop = fixed_termination("repeat_loop", step_count=21)

        self.assertAlmostEqual(gold.reward, 0.93)
        self.assertAlmostEqual(abstain.reward, -0.47)
        self.assertAlmostEqual(loop.reward, -0.67)
        detail = gold.to_dict()
        self.assertEqual(detail["base_terminal_utility"], 1.0)
        self.assertEqual(detail["step_count"], 21)
        self.assertAlmostEqual(detail["step_penalty"], -0.07)

    def test_reward_invalid_terminal_is_not_step_penalized(self):
        unverifiable_product = product()
        unverifiable_product.pop("category")
        result = evaluate_purchase(
            unverifiable_product,
            goal(),
            selected_options={"颜色分类": "白色", "尺码": "XL"},
            step_count=45,
        )
        self.assertEqual(result.reward_type, "reward_unverifiable")
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.to_dict()["step_penalty"], 0.0)

    def test_reward_order_prevents_panic_buying(self):
        self.assertEqual(
            DEFAULT_REWARDS["valid_alternative_purchase"],
            DEFAULT_REWARDS["partial_purchase_base"]
            + DEFAULT_REWARDS["partial_purchase_scale"],
        )
        self.assertEqual(DEFAULT_REWARDS["max_steps"], 0.0)
        self.assertGreater(
            DEFAULT_REWARDS["early_abstain"],
            DEFAULT_REWARDS["repeat_loop"],
        )
        self.assertGreater(
            DEFAULT_REWARDS["repeat_loop"],
            DEFAULT_REWARDS["wrong_purchase"],
        )

    def test_gold_and_full_alternative_are_both_successful(self):
        selected = {"颜色分类": "白色", "尺码": "XL"}
        gold = evaluate_purchase(product(), goal(), selected_options=selected)
        alternative = evaluate_purchase(
            product("222222222222"),
            goal(),
            selected_options=selected,
        )
        self.assertEqual(gold.reward_type, "gold_purchase")
        self.assertEqual(gold.reward, 1.0)
        self.assertEqual(
            alternative.reward_type,
            "valid_alternative_purchase",
        )
        self.assertEqual(
            alternative.reward,
            DEFAULT_REWARDS["valid_alternative_purchase"],
        )
        self.assertTrue(alternative.to_dict()["purchase_success"])

    def test_missing_instruction_hash_is_audited_without_invalidating_reward(self):
        task_goal = goal()
        task_goal.pop("instruction_sha256")
        result = evaluate_purchase(
            product(),
            task_goal,
            selected_options={"颜色分类": "白色", "尺码": "XL"},
        )
        self.assertEqual(result.reward_type, "gold_purchase")
        self.assertTrue(result.reward_valid)
        self.assertFalse(
            result.to_dict()["evidence"]["instruction_contract_integrity"]["valid"]
        )

    def test_changed_instruction_is_realigned_before_reward(self):
        task_goal = goal()
        task_goal["instruction_text"] = "购买支持热洗的白色 XL 洗地机，预算不超过2200元"
        task_goal = align_goal_reward_features(task_goal, product())

        result = evaluate_purchase(
            product(),
            task_goal,
            selected_options={"颜色分类": "白色", "尺码": "XL"},
        )

        self.assertEqual(result.reward_type, "gold_purchase")
        self.assertTrue(
            result.to_dict()["evidence"]["instruction_contract_integrity"]["valid"]
        )

    def test_lightweight_features_detect_explicit_brand_and_model(self):
        target = product()
        features = compile_reward_features(
            {
                "instruction": "购买石头 A20 洗地机",
                "attributes": [],
                "instruction_options": [],
            },
            target,
        )
        self.assertEqual(features["expected_brand"], ["石头"])
        self.assertEqual(features["expected_model"], ["a20"])

    def test_reward_v4_exposes_query_constraint_provenance_and_results(self):
        task_goal = goal()
        contract = task_goal["query_constraint_contract"]
        self.assertEqual(
            contract["schema_version"],
            "shopping-query-constraints-v1",
        )
        self.assertTrue(
            any(
                item["constraint_type"] == "budget_upper"
                and item["source"] == "query.explicit_budget"
                for item in contract["constraints"]
            )
        )

        detail = evaluate_purchase(
            product(),
            task_goal,
            selected_options={"颜色分类": "白色", "尺码": "XL"},
        ).to_dict()

        self.assertEqual(detail["reward_version"], "shopsimulator-reward-v4")
        self.assertEqual(
            detail["query_constraint_version"],
            "shopping-query-constraints-v1",
        )
        self.assertGreater(detail["constraint_summary"]["total"], 0)
        self.assertEqual(
            detail["constraint_summary"]["status_counts"]["fail"],
            0,
        )

    def test_wrong_option_gets_continuous_partial_reward(self):
        result = evaluate_purchase(
            product(),
            goal(),
            selected_options={"颜色分类": "白色", "尺码": "L"},
        )
        self.assertEqual(result.reward_type, "wrong_purchase")
        self.assertEqual(result.reward, DEFAULT_REWARDS["wrong_purchase"])
        self.assertLessEqual(result.reward, 0.25)

    def test_public_numeric_option_constraint_keeps_gold_on_target_asin(self):
        target = product()
        target["customization_options"] = {
            "颜色分类": [
                {"value": "小号手工枕高9cm", "price": 56},
                {"value": "多功能手工枕高12cm", "price": 59},
                {"value": "大号手工枕高16cm", "price": 65},
            ]
        }
        task_goal = align_manual_goal({
            "asin": target["asin"],
            "category": target["category"],
            "price_upper": 60,
            "instruction_text": "想要大小在15厘米以下的护颈枕，价格60元以内。",
            "expected_brand": [],
            "expected_model": [],
            "expected_core_functions": [],
            "required_options_by_key": {
                "color": {
                    "value": "多功能手工枕高12cm",
                    "source_axis": "颜色分类",
                }
            },
            "unresolved_option_requirements": [],
        })

        result = evaluate_purchase(
            target,
            task_goal,
            selected_options={"颜色分类": "小号手工枕高9cm"},
        )

        self.assertEqual(result.reward_type, "gold_purchase")
        self.assertTrue(result.target_asin_match)
        self.assertEqual(result.weighted_score, 1.0)
        evidence = result.to_dict()["evidence"]
        self.assertFalse(evidence["exact_target_variant_match"])
        option_result = evidence["preference_scoring"]["dimensions"]["key_options"]
        self.assertEqual(
            option_result["results"][0]["comparator"],
            "instruction_numeric_option_constraint_v1",
        )

    def test_public_numeric_option_constraint_does_not_accept_out_of_range_variant(self):
        target = product()
        target["customization_options"] = {
            "颜色分类": [
                {"value": "多功能手工枕高12cm", "price": 59},
                {"value": "大号手工枕高16cm", "price": 59},
            ]
        }
        task_goal = align_manual_goal({
            "asin": target["asin"],
            "category": target["category"],
            "price_upper": 60,
            "instruction_text": "想要大小在15厘米以下的护颈枕，价格60元以内。",
            "expected_brand": [],
            "expected_model": [],
            "expected_core_functions": [],
            "required_options_by_key": {
                "color": {"value": "多功能手工枕高12cm"}
            },
            "unresolved_option_requirements": [],
        })

        result = evaluate_purchase(
            target,
            task_goal,
            selected_options={"颜色分类": "大号手工枕高16cm"},
        )

        self.assertEqual(result.reward_type, "wrong_purchase")

    def test_per_unit_price_uses_selected_variant_quantity(self):
        target = product(attributes=[])
        target["customization_options"] = {
            "颜色分类": [
                {"value": "12g*50包", "price": 85},
            ]
        }
        instruction = {
            "instruction": "需50包小包装，每包价格约1.5元左右。",
            "attributes": [],
            "instruction_options": ["12g*50包"],
        }
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }

        result = evaluate_purchase(
            target,
            task_goal,
            selected_options={"颜色分类": "12g*50包"},
            price=85,
        )

        self.assertEqual(result.reward_type, "gold_purchase")
        price_evidence = result.to_dict()["evidence"]["price_comparison"]["evidence"][
            "evidence"
        ]
        self.assertEqual(price_evidence["unit_count"], 50)
        self.assertEqual(price_evidence["variant_total_price"], 85.0)

    def test_exact_target_variant_remains_strict_gold(self):
        target = product()
        target["customization_options"] = {
            "颜色分类": [
                {"value": "小号手工枕高9cm", "price": 56},
                {"value": "多功能手工枕高12cm", "price": 59},
            ]
        }
        task_goal = align_manual_goal({
            "asin": target["asin"],
            "category": target["category"],
            "price_upper": 60,
            "instruction_text": "想要大小在15厘米以下的护颈枕，价格60元以内。",
            "expected_brand": [],
            "expected_model": [],
            "expected_core_functions": [],
            "required_options_by_key": {
                "color": {"value": "多功能手工枕高12cm"}
            },
            "unresolved_option_requirements": [],
        })

        result = evaluate_purchase(
            target,
            task_goal,
            selected_options={"颜色分类": "多功能手工枕高12cm"},
        )

        self.assertEqual(result.reward_type, "gold_purchase")
        self.assertTrue(result.to_dict()["evidence"]["exact_target_variant_match"])

    def test_brand_can_use_safe_title_and_exact_attribute_evidence(self):
        target = product()
        target.update(
            {
                "title": "飞利浦电动牙刷头替换 英伦绿4支",
                "brand": "健康护理海外购",
                "shop_name": "健康护理海外购",
                "attribute": ["飞利浦", "电动牙刷", "替换"],
                "customization_options": {
                    "颜色分类": [{"value": "英伦绿4支", "price": 740}]
                },
            }
        )
        task_goal = align_manual_goal({
            "asin": target["asin"],
            "category": target["category"],
            "price_upper": 814,
            "instruction_text": "找飞利浦电动牙刷替换刷头，英伦绿4支。",
            "expected_brand": ["philips"],
            "expected_model": [],
            "expected_core_functions": [],
            "required_options_by_key": {"color": {"value": "英伦绿4支"}},
            "unresolved_option_requirements": [],
        })

        result = evaluate_purchase(
            target,
            task_goal,
            selected_options={"颜色分类": "英伦绿4支"},
        )

        self.assertEqual(result.reward_type, "gold_purchase")
        brand_result = result.to_dict()["evidence"]["preference_scoring"]["dimensions"]["brand"]
        self.assertEqual(brand_result["score"], 1.0)
        self.assertEqual(
            brand_result["results"][0]["comparator"],
            "explicit_brand_alias_evidence_v2",
        )

    def test_compatibility_phrase_alone_does_not_prove_product_brand(self):
        candidate = product()
        candidate.update(
            {
                "title": "兼容飞利浦电动牙刷的第三方替换刷头",
                "brand": "通用",
                "shop_name": "通用配件店",
                "attribute": ["兼容飞利浦"],
            }
        )
        task_goal = goal(candidate)
        task_goal.update(
            {
                "expected_brand": ["philips"],
                "expected_model": [],
                "expected_core_functions": [],
            }
        )

        result = evaluate_purchase(
            candidate,
            task_goal,
            selected_options={"颜色分类": "白色", "尺码": "XL"},
        )

        self.assertEqual(result.reward_type, "wrong_purchase")

    def test_nearly_unmatched_alternative_is_partial(self):
        unrelated = product(
            "222222222222",
            attributes=["普通清洁"],
        )
        unrelated["title"] = "同类普通清洁设备"
        result = evaluate_purchase(
            unrelated,
            goal(),
            selected_options={"颜色分类": "黑色", "尺码": "L"},
        )
        self.assertEqual(result.reward_type, "wrong_purchase")
        self.assertEqual(result.reward, DEFAULT_REWARDS["wrong_purchase"])

    def test_target_asin_must_also_satisfy_explicit_price(self):
        cross_category = product()
        cross_category["category"] = "数码›电脑›笔记本电脑"
        selected = {"颜色分类": "白色", "尺码": "XL"}
        wrong_category = evaluate_purchase(
            cross_category,
            goal(),
            selected_options=selected,
        )
        over_budget_goal = goal()
        over_budget_goal["price_upper"] = 1900
        over_budget_goal["price_constraint"] = {
            "operator": "lte",
            "value": 1900,
            "approximate": False,
        }
        over_budget = evaluate_purchase(
            product(),
            over_budget_goal,
            selected_options=selected,
        )
        self.assertEqual(wrong_category.reward_type, "wrong_purchase")
        self.assertEqual(wrong_category.reward, -1.0)
        self.assertEqual(over_budget.reward_type, "wrong_purchase")
        self.assertTrue(
            {"category", "q0002"}.issubset(set(over_budget.hard_gates))
        )
        price_rows = [
            row
            for row in over_budget.to_dict()["constraint_results"]
            if row.get("constraint_type") == "budget_upper"
        ]
        self.assertTrue(price_rows)
        self.assertTrue(all(row["status"] == FAIL for row in price_rows))

    def test_missing_user_budget_does_not_invent_an_upper_bound(self):
        no_budget_goal = goal()
        no_budget_goal["price_upper"] = None
        no_budget_goal["price_constraint"] = None
        result = evaluate_purchase(
            product(),
            no_budget_goal,
            selected_options={"颜色分类": "白色", "尺码": "XL"},
            price=99999,
        )
        self.assertEqual(result.reward_type, "gold_purchase")
        price_comparison = result.to_dict()["evidence"]["price_comparison"]
        self.assertEqual(price_comparison["status"], "pass")
        self.assertEqual(
            price_comparison["comparator"],
            "budget_not_declared_v1",
        )

    def test_option_comparison_uses_key_and_exact_value(self):
        target_goal = goal()
        candidate = product("222222222222")
        candidate["customization_options"]["尺码"] = [
            {"value": "XXL", "price": 1999}
        ]
        gate = compare_required_options(
            candidate,
            target_goal["required_options_by_key"],
            {"颜色": "白色", "鞋码": "XXL"},
        )
        self.assertEqual(gate["status"], FAIL)

    def test_stale_unresolved_option_mutation_cannot_bypass_frozen_contract(self):
        task_goal = goal()
        task_goal["required_options_by_key"] = {}
        task_goal["unresolved_option_requirements"] = [
            {"value": "白色", "reason": "axis_not_found", "axes": []}
        ]
        result = evaluate_purchase(
            product(),
            task_goal,
            selected_options={"任意规格轴": "白色"},
            price=1999,
        )
        self.assertEqual(result.reward_type, "wrong_purchase")

    def test_unique_effective_price_axis_is_used(self):
        resolution = resolve_variant_price(
            product(),
            {"颜色分类": "白色", "尺码": "XL"},
        )
        self.assertEqual(resolution["status"], PASS)
        self.assertEqual(
            resolution["method"],
            "unique_effective_price_axis",
        )
        self.assertEqual(resolution["price"], 1999)

    def test_unverifiable_explicit_price_prevents_gold_purchase(self):
        candidate = product()
        candidate["customization_options"]["颜色分类"][1]["price"] = 2099
        result = evaluate_purchase(
            candidate,
            goal(),
            selected_options={"颜色分类": "白色", "尺码": "XL"},
        )
        self.assertEqual(result.reward_type, "reward_unverifiable")
        self.assertTrue(result.to_dict()["sampling_invalid"])
        self.assertEqual(
            result.to_dict()["evidence"]["price_comparison"]["status"],
            UNVERIFIABLE,
        )

    def test_confirmed_hard_failure_takes_priority_over_other_unverifiable_hard_constraint(self):
        candidate = product()
        candidate["category"] = "家电›清洁电器›吸尘器"
        candidate["customization_options"]["颜色分类"][1]["price"] = 2099
        result = evaluate_purchase(
            candidate,
            goal(),
            selected_options={"颜色分类": "白色", "尺码": "XL"},
        )
        self.assertEqual(result.reward_type, "wrong_purchase")
        self.assertTrue(result.reward_valid)
        self.assertEqual(result.reward, DEFAULT_REWARDS["wrong_purchase"])
        self.assertEqual(
            result.to_dict()["evidence"]["price_comparison"]["status"],
            UNVERIFIABLE,
        )

    def test_candidate_eligibility_uses_score_and_coverage(self):
        result = evaluate_candidate_eligibility(product(), goal())
        self.assertTrue(result["known_acceptable"])
        self.assertTrue(result["known_valid"])
        self.assertEqual(result["status"], PASS)
        self.assertGreaterEqual(result["match_score"], 0.7)

    def test_candidate_eligibility_does_not_require_agent_option_selection(self):
        no_option_instruction = {
            "instruction": "购买支持热洗的洗地机，预算2200元",
            "attributes": ["洗地", "热洗"],
            "instruction_options": [],
        }
        target = product()
        task_goal = {
            "asin": target["asin"],
            "category": target["category"],
            "price_upper": 2200,
            "instruction_text": no_option_instruction["instruction"],
            **compile_reward_features(no_option_instruction, target),
        }
        result = evaluate_candidate_eligibility(target, task_goal)
        self.assertTrue(result["known_acceptable"])
        self.assertEqual(result["price_resolution"]["status"], PASS)
        self.assertTrue(result["option_resolution"]["inferred_options"])

    def test_finish_without_purchase_is_always_early_abstain(self):
        early = evaluate_abstain(
            effective_result_sets=2,
            opened_candidates=1,
            known_acceptable_candidates=0,
        )
        explored = evaluate_abstain(
            effective_result_sets=2,
            opened_candidates=2,
            known_acceptable_candidates=0,
        )
        blocked = evaluate_abstain(
            effective_result_sets=2,
            opened_candidates=2,
            known_acceptable_candidates=1,
        )
        self.assertEqual(early.reward_type, "early_abstain")
        self.assertEqual(explored.reward_type, "early_abstain")
        self.assertEqual(blocked.reward_type, "early_abstain")

    def test_field_comparators_resist_common_substring_attacks(self):
        self.assertEqual(
            compare_model(["A20"], product(model="A200"))["status"],
            FAIL,
        )
        compatible = product()
        compatible["brand"] = "Generic"
        compatible["title"] = "Compatible with Apple 洗地机"
        self.assertEqual(
            compare_brand(["Apple"], compatible)["status"],
            FAIL,
        )
        negated = product(attributes=["不支持防水"])
        self.assertEqual(
            compare_core_functions(["防水"], negated)["status"],
            FAIL,
        )

    def test_numeric_comparator_normalizes_units(self):
        gate = compare_numeric_spec(
            {"value": 2, "unit": "L", "operator": "eq"},
            "容量为2000ml",
        )
        self.assertEqual(gate["status"], PASS)

    def test_loop_and_max_step_values_are_frozen(self):
        self.assertEqual(fixed_termination("max_steps").reward, 0.0)
        self.assertEqual(fixed_termination("max_steps", step_count=45).reward, -1.05)
        self.assertEqual(fixed_termination("repeat_loop").reward, -0.6)
        repeat = fixed_termination(
            "repeat_loop",
            subreason="exact_action_repeat",
        ).to_dict()
        self.assertEqual(
            repeat["termination_subreason"],
            "exact_action_repeat",
        )


if __name__ == "__main__":
    unittest.main()
