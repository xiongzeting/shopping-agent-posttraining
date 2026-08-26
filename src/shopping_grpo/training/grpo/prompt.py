"""GRPO-only protocol reminder layered over the shared shopping prompt."""

from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT


GRPO_TOOL_PROTOCOL = """

GRPO 工具协议：环境结束前，每次回复必须且只能包含一个标准工具调用。禁止只输出普通 assistant 文本，禁止在同一回复中生成多个工具调用；需要判断时在内部完成，然后直接调用一个当前合法工具。无参数工具必须传严格的 `{}`。
"""

GRPO_SYSTEM_PROMPT = SYSTEM_PROMPT.rstrip() + GRPO_TOOL_PROTOCOL
