from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

    from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
    from shopping_grpo.training.grpo.adapter.runtime import (
        current_runtime_state,
        make_runtime_state,
    )
except ModuleNotFoundError:  # pragma: no cover - optional local training deps
    AgentState = ToolAgentLoop = ShoppingToolAgentLoop = None


@unittest.skipIf(ShoppingToolAgentLoop is None, "veRL training dependencies unavailable")
class ShoppingAgentLoopSerializationTest(unittest.TestCase):
    def test_parallel_calls_keep_first_and_record_truncation(self):
        loop = object.__new__(ShoppingToolAgentLoop)
        agent_data = SimpleNamespace(tool_calls=["first", "second", "third"])
        state = make_runtime_state(task_id=1, max_steps=45)
        token = current_runtime_state.set(state)

        async def parent_handler(_self, received):
            self.assertIs(received, agent_data)
            self.assertEqual(received.tool_calls, ["first"])
            return AgentState.GENERATING

        try:
            with patch.object(
                ToolAgentLoop,
                "_handle_processing_tools_state",
                new=parent_handler,
            ):
                result = asyncio.run(loop._handle_processing_tools_state(agent_data))
        finally:
            current_runtime_state.reset(token)

        self.assertEqual(result, AgentState.GENERATING)
        self.assertEqual(agent_data.tool_calls, ["first"])
        self.assertEqual(state["tool_call_truncation_count"], 2)
        self.assertFalse(state["terminate"])
        self.assertIsNone(state["error"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
