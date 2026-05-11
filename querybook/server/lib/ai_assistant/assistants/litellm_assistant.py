import os

from langchain_openai import ChatOpenAI

from lib.ai_assistant.base_ai_assistant import BaseAIAssistant
from lib.logger import get_logger

LOG = get_logger(__file__)

# Context window sizes for common Bedrock models accessed via LiteLLM
LITELLM_MODEL_CONTEXT_WINDOW_SIZE = {
    "claude-opus-4-6": 200000,
}

DEFAULT_CONTEXT_WINDOW_SIZE = 200000
DEFAULT_MODEL_NAME = "claude-opus-4-6"


class LiteLLMAssistant(BaseAIAssistant):
    """AI Assistant that uses LiteLLM as a proxy to forward requests to AWS Bedrock.

    Required environment variable:
        LITELLM_KEY: API key for the LiteLLM proxy server

    Configuration example in querybook_config.yaml:
        AI_ASSISTANT_CONFIG:
            default:
                model_args:
                    model_name: "claude-opus-4-6"
                    api_base: "https://litellm-proxy.com"
    """

    @property
    def name(self) -> str:
        return "litellm"

    def _get_context_length_by_model(self, model_name: str) -> int:
        return LITELLM_MODEL_CONTEXT_WINDOW_SIZE.get(
            model_name, DEFAULT_CONTEXT_WINDOW_SIZE
        )

    def _get_default_llm_config(self):
        default_config = super()._get_default_llm_config()
        if not default_config.get("model_name"):
            default_config["model_name"] = DEFAULT_MODEL_NAME

        return default_config

    def _get_token_count(self, ai_command: str, prompt: str) -> int:
        # Use a simple approximation for token counting.
        # LiteLLM proxies to various models (Claude, Llama, etc.) that each
        # have different tokenizers. A character-based approximation of ~4
        # characters per token is a reasonable conservative estimate.
        return len(prompt) // 4

    def _get_error_msg(self, error) -> str:
        return super()._get_error_msg(error)

    def _get_llm(self, ai_command: str, prompt_length: int):
        config = self._get_llm_config(ai_command)

        litellm_api_key = os.environ.get("LITELLM_KEY")
        if not litellm_api_key:
            raise ValueError(
                "LITELLM_KEY environment variable is not set. "
                "Please set it to your LiteLLM proxy API key."
            )

        # LiteLLM proxy exposes an OpenAI-compatible API, so we use
        # ChatOpenAI with the LiteLLM proxy's base URL and API key.
        # The model_name should be in LiteLLM format (e.g., "bedrock/anthropic.claude-opus-4-6-20250515-v1:0")
        return ChatOpenAI(
            openai_api_key=litellm_api_key,
            openai_api_base=config.pop("api_base", None),
            **config,
        )
