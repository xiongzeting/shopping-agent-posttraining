import unittest

from web_agent_site.engine.observation import (
    build_observation_state,
    stable_option_id,
)


class ObservationV2Test(unittest.TestCase):
    def test_search_page_preserves_all_twenty_products_and_ranks(self):
        products = {
            f"{index:012d}": {
                "asin": f"{index:012d}",
                "title": f"商品 {index}",
                "category": "测试",
                "pricing": [index],
            }
            for index in range(1, 41)
        }
        state = build_observation_state(
            page_type="search_results",
            session={
                "keywords": ["商品"],
                "normalized_query": "商品",
                "page": 2,
                "total_pages": 2,
                "total_results": 40,
                "search_result_asins": list(products),
                "current_page_asins": list(products)[20:40],
            },
            product_item_dict=products,
            available_actions={
                "has_search_bar": False,
                "clickables": ["< prev", *list(products)[20:40]],
            },
        )
        self.assertEqual(len(state["products"]), 20)
        self.assertEqual(state["rank_start"], 21)
        self.assertEqual(state["rank_end"], 40)
        self.assertEqual(
            {product["asin"] for product in state["products"]},
            set(state["actions"]) - {"< prev"},
        )
        self.assertTrue(
            all(
                "features" not in product and "attributes" not in product
                for product in state["products"]
            )
        )

    def test_builder_has_no_goal_parameter_or_hidden_answer(self):
        state = build_observation_state(
            page_type="search_home",
            session={},
            product_item_dict={},
            available_actions={"has_search_bar": True, "clickables": []},
        )
        self.assertNotIn("goal", state)
        self.assertNotIn("reward", state)

    def test_option_ids_are_stable_and_axis_specific(self):
        first = stable_option_id("A1", "颜色", "标准")
        self.assertEqual(first, stable_option_id("A1", "颜色", "标准"))
        self.assertNotEqual(first, stable_option_id("A1", "尺码", "标准"))
        self.assertNotEqual(first, stable_option_id("A2", "颜色", "标准"))

    def test_product_state_exposes_option_ids_without_candidate_judgment(self):
        asin = "123456789012"
        black_id = stable_option_id(asin, "颜色", "黑色")
        state = build_observation_state(
            page_type="product_detail",
            session={
                "asin": asin,
                "options": {"颜色": "黑色"},
            },
            product_item_dict={
                asin: {
                    "asin": asin,
                    "title": "测试商品",
                    "BulletPoints": ["", "支持防水", "简约设计"],
                    "Attributes": ["高度66cm", "咖啡色"],
                    "options": {"颜色": ["黑色", "白色"]},
                }
            },
            available_actions={
                "has_search_bar": False,
                "clickables": [
                    "Description",
                    "Features",
                    "Reviews",
                    "Attributes",
                    black_id,
                    stable_option_id(asin, "颜色", "白色"),
                    "buy now",
                ],
            },
        )

        self.assertEqual(
            state["selected_options"]["颜色"],
            {"option_id": black_id, "label": "黑色"},
        )
        self.assertEqual(
            state["available_options"]["颜色"][0],
            {"option_id": black_id, "label": "黑色"},
        )
        self.assertEqual(state["product"]["features"], ["支持防水", "简约设计"])
        self.assertEqual(state["product"]["attributes"], ["高度66cm", "咖啡色"])
        self.assertTrue(
            {"description", "features", "reviews", "attributes"}.isdisjoint(
                {action.casefold() for action in state["actions"]}
            )
        )
        self.assertNotIn("candidate_state", state)

    def test_product_state_omits_empty_features_and_attributes(self):
        asin = "123456789012"
        state = build_observation_state(
            page_type="product_detail",
            session={"asin": asin, "options": {}},
            product_item_dict={
                asin: {
                    "asin": asin,
                    "title": "测试商品",
                    "BulletPoints": ["", "  "],
                    "Attributes": [],
                    "options": {},
                }
            },
            available_actions={
                "has_search_bar": False,
                "clickables": ["Description", "Attributes", "buy now"],
            },
        )

        self.assertNotIn("features", state["product"])
        self.assertNotIn("attributes", state["product"])
        self.assertEqual(state["actions"], ["buy now"])


if __name__ == "__main__":
    unittest.main()
