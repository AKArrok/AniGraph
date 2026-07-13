"""LLM Token 成本计算 — 支持多提供商。

当前定价（$/1M tokens）:
    DeepSeek:
        deepseek-v4-pro:   input $0.55  output $2.19
        deepseek-v4-flash: input $0.14  output $0.56
    未知模型: input $0   output $0 (cost = "$0.0000")
"""

from typing import ClassVar


class TokenProvider:
    """LLM 成本计算器。"""

    # 定价表: provider -> model -> {input, output}  ($/1M tokens)
    PROVIDERS: ClassVar[dict] = {
        "deepseek": {
            "deepseek-v4-pro":   {"input": 0.55, "output": 2.19},
            "deepseek-v4-flash": {"input": 0.14, "output": 0.56},
        },
        "openai": {
            "gpt-4o":            {"input": 2.50, "output": 10.00},
            "gpt-4o-mini":       {"input": 0.15, "output": 0.60},
        },
    }

    @classmethod
    def calculate(cls, model: str, input_tokens: int, output_tokens: int) -> str:
        """计算单次 LLM 调用成本，返回 "$0.0012" 格式字符串。

        Args:
            model: 模型名 e.g. "deepseek-v4-pro"
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数

        Returns:
            成本字符串，如 "$0.0012"。未知模型返回 "$0.0000"。
        """
        # 查找模型定价
        pricing = None
        for provider_models in cls.PROVIDERS.values():
            if model in provider_models:
                pricing = provider_models[model]
                break

        if pricing is None:
            return "$0.0000"

        cost = (input_tokens / 1_000_000) * pricing["input"] \
             + (output_tokens / 1_000_000) * pricing["output"]

        return f"${cost:.4f}"
