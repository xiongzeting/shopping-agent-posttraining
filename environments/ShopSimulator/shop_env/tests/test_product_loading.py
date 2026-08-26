from unittest.mock import mock_open, patch

from web_agent_site.engine.engine import load_products


def test_product_archive_is_read_as_utf8():
    reader = mock_open(read_data='[]')

    with patch('builtins.open', reader):
        products, product_items, prices, attributes = load_products('products.json')

    reader.assert_called_once_with('products.json', encoding='utf-8')
    assert products == []
    assert product_items == {}
    assert prices == {}
    assert dict(attributes) == {}
