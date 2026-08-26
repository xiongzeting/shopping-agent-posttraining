import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_grpo_dev_probe import (
    _jsonl_task_ids,
    _parquet_task_ids,
    _parse_quotas,
    _row_task_id,
    _stream_json_array,
)


class BuildGrpoDevProbeTests(unittest.TestCase):
    def test_stream_json_array_handles_objects_across_tiny_chunks(self):
        rows = [{"task_id": 1, "text": "中文"}, {"task_id": 2, "nested": {"x": [1, 2]}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(list(_stream_json_array(path, chunk_size=7)), rows)

    def test_task_id_can_be_read_from_serialized_extra_info(self):
        self.assertEqual(_row_task_id({"extra_info": '{"task_id": 42}'}), 42)

    def test_jsonl_task_ids_supports_direct_and_nested_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(
                '{"task_id": 3}\n{"extra_info": {"task_id": 7}}\n',
                encoding="utf-8",
            )
            self.assertEqual(_jsonl_task_ids(path), {3, 7})

    def test_default_quotas_select_one_hundred(self):
        quotas = _parse_quotas("15,45,30,10")
        self.assertEqual(sum(quotas.values()), 100)

    def test_parquet_task_ids_reads_nested_extra_info(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [{"extra_info": {"task_id": 11}}, {"extra_info": {"task_id": 13}}]
                ),
                path,
            )
            self.assertEqual(_parquet_task_ids(path), {11, 13})


if __name__ == "__main__":
    unittest.main()
