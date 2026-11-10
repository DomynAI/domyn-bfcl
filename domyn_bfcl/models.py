# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Registry entries added to upstream's ``MODEL_CONFIG_MAPPING`` (handlers in ``handlers/``).

Every OSS handler here runs on ``DomynOSSMixin`` (see ``domyn_bfcl.oss``). Upstream's
own handlers are used unmodified, composed with the mixin.
"""

from bfcl_eval.constants.model_config import ModelConfig
from bfcl_eval.model_handler.local_inference.nanbeige_fc import NanbeigeFCHandler
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

from domyn_bfcl.handlers.chat_fc import ChatFCHandler
from domyn_bfcl.handlers.domyn import DomynHandler
from domyn_bfcl.handlers.ministral_fc import MinistralFCHandler
from domyn_bfcl.handlers.mock import MockHandler
from domyn_bfcl.handlers.nemotron_fc import NemotronOSSHandler
from domyn_bfcl.handlers.olmo_fc import OlmoFCHandler
from domyn_bfcl.oss import DomynOSSMixin


class DomynQwenFCHandler(DomynOSSMixin, QwenFCHandler):
    """Upstream ``QwenFCHandler`` with Domyn sampling, budget and endpoint handling."""


class DomynNanbeigeFCHandler(DomynOSSMixin, NanbeigeFCHandler):
    """Upstream ``NanbeigeFCHandler`` with Domyn sampling, budget and endpoint handling."""


def _oss(
    model_name: str,
    display_name: str,
    url: str,
    org: str,
    model_license: str,
    handler: type,
    *,
    is_fc_model: bool = True,
    underscore_to_dot: bool = False,
) -> ModelConfig:
    return ModelConfig(
        model_name=model_name,
        display_name=display_name,
        url=url,
        org=org,
        license=model_license,
        model_handler=handler,  # pyright: ignore[reportArgumentType]  # upstream annotates the class as str
        input_price=None,
        output_price=None,
        is_fc_model=is_fc_model,
        underscore_to_dot=underscore_to_dot,
    )


MODEL_CONFIGS = {
    "mistralai/Ministral-3-8B-Reasoning-2512": _oss(
        "mistralai/Ministral-3-8B-Reasoning-2512",
        "Ministral-3-8B-Reasoning-2512 (FC)",
        "https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512",
        "Mistral AI",
        "Mistral AI Research License",
        MinistralFCHandler,
        is_fc_model=False,
    ),
    "Ministral-3-3B-Reasoning-2512-FC": _oss(
        "Ministral-3-3B-Reasoning-2512-FC",
        "Ministral-3-3B-Reasoning-2512 (FC)",
        "https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512",
        "Mistral AI",
        "Mistral AI Research License",
        MinistralFCHandler,
    ),
    "Qwen/Qwen3-4B-FC": _oss(
        "Qwen/Qwen3-4B",
        "Qwen3-4B (FC)",
        "https://huggingface.co/Qwen/Qwen3-4B",
        "Qwen",
        "apache-2.0",
        DomynQwenFCHandler,
    ),
    "VibeThinker-3B-FC": _oss(
        "WeiboAI/VibeThinker-3B",
        "VibeThinker-3B (FC)",
        "https://huggingface.co/WeiboAI/VibeThinker-3B",
        "WeiboAI",
        "apache-2.0",
        DomynQwenFCHandler,
    ),
    "Qwen/Qwen3.5-4B-FC": _oss(
        "Qwen/Qwen3.5-4B",
        "Qwen3.5-4B (FC)",
        "https://huggingface.co/Qwen/Qwen3.5-4B",
        "Qwen",
        "apache-2.0",
        DomynQwenFCHandler,
    ),
    "Qwen/Qwen3.5-4B-Base-FC": _oss(
        "Qwen/Qwen3.5-4B-Base",
        "Qwen3.5-4B-Base (FC)",
        "https://huggingface.co/Qwen/Qwen3.5-4B-Base",
        "Qwen",
        "apache-2.0",
        DomynQwenFCHandler,
    ),
    "Qwen/Qwen3.5-9B-FC": _oss(
        "Qwen/Qwen3.5-9B",
        "Qwen3.5-9B (FC)",
        "https://huggingface.co/Qwen/Qwen3.5-9B",
        "Qwen",
        "apache-2.0",
        DomynQwenFCHandler,
    ),
    "Nanbeige4.1-3B-FC": _oss(
        "Nanbeige4.1-3B-FC",
        "Nanbeige4.1-3B (FC)",
        "https://huggingface.co/Nanbeige/Nanbeige4.1-3B",
        "Nanbeige",
        "apache-2.0",
        DomynNanbeigeFCHandler,
    ),
    "mock-FC": _oss("mock", "mock", "", "mock", "mock", MockHandler, underscore_to_dot=True),
    "allenai/Olmo-3-7B-Think-FC": _oss(
        "allenai/Olmo-3-7B-Think",
        "Olmo-3-7B-Think (FC)",
        "https://huggingface.co/allenai/Olmo-3-7B-Think",
        "Allen Institute for AI",
        "apache-2.0",
        OlmoFCHandler,
    ),
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1-FC": _oss(
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "Llama-3.1-Nemotron-Nano-8B-v1 (FC)",
        "https://huggingface.co/nvidia/Llama-3_1-Nemotron-Nano-8B-v1",
        "NVIDIA",
        "nvidia-open-model-license",
        NemotronOSSHandler,
    ),
    "Domyn/Domyn-Edge-FC": _oss(
        "Domyn/Domyn-Edge-FC",
        "Domyn-Edge (FC)",
        "https://huggingface.co/domyn/Domyn-Edge-SFT",
        "Domyn",
        "apache-2.0",
        DomynHandler,
        underscore_to_dot=True,
    ),
    "Domyn/Domyn-Small-FC": _oss(
        "Domyn/Domyn-Small-FC",
        "Domyn-Small (FC)",
        "https://huggingface.co/domyn/Domyn-Small-v1.0",
        "Domyn",
        "mit",
        DomynHandler,
        underscore_to_dot=True,
    ),
    "HuggingFaceTB/SmolLM3-3B-FC": _oss(
        "HuggingFaceTB/SmolLM3-3B-FC",
        "SmolLM3-3B (FC)",
        "https://huggingface.co/HuggingFaceTB/SmolLM3-3B",
        "HuggingFaceTB",
        "apache-2.0",
        ChatFCHandler,
        underscore_to_dot=True,
    ),
    # No upstream or Domyn handler fits these: the server's chat template and
    # --tool-call-parser do the formatting (see ChatFCHandler).
    "allenai/Olmo-3-7B-Instruct-FC": _oss(
        "allenai/Olmo-3-7B-Instruct-FC",
        "Olmo-3-7B-Instruct (FC)",
        "https://huggingface.co/allenai/Olmo-3-7B-Instruct",
        "Ai2",
        "apache-2.0",
        ChatFCHandler,
        underscore_to_dot=True,
    ),
    "mistralai/Ministral-3-8B-Instruct-2512-FC": _oss(
        "mistralai/Ministral-3-8B-Instruct-2512-FC",
        "Ministral-3-8B-Instruct-2512 (FC)",
        "https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512",
        "Mistral AI",
        "apache-2.0",
        ChatFCHandler,
        underscore_to_dot=True,
    ),
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FC": _oss(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FC",
        "NVIDIA-Nemotron-Nano-9B-v2 (FC)",
        "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "NVIDIA",
        "nvidia-open-model-license",
        ChatFCHandler,
        underscore_to_dot=True,
    ),
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16-FC": _oss(
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16-FC",
        "NVIDIA-Nemotron-3-Nano-4B (FC)",
        "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "NVIDIA",
        "nvidia-nemotron-open-model-license",
        ChatFCHandler,
        underscore_to_dot=True,
    ),
    "tencent/Youtu-LLM-2B-FC": _oss(
        "tencent/Youtu-LLM-2B-FC",
        "Youtu-LLM-2B (FC)",
        "https://huggingface.co/tencent/Youtu-LLM-2B",
        "Tencent",
        "youtu-llm",
        ChatFCHandler,
        underscore_to_dot=True,
    ),
    "tencent/Youtu-LLM-2B-Base-FC": _oss(
        "tencent/Youtu-LLM-2B-Base-FC",
        "Youtu-LLM-2B-Base (FC)",
        "https://huggingface.co/tencent/Youtu-LLM-2B-Base",
        "Tencent",
        "youtu-llm",
        ChatFCHandler,
        underscore_to_dot=True,
    ),
}
