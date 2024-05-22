from transformers import (
    MixtralForCausalLM,
    MixtralConfig,
    AutoModelForCausalLM,
)
import torch
from megatron_lm.megatron.global_vars import get_args
from transformers.integrations import is_deepspeed_zero3_enabled


def get_model(
    model_name: str, use_cache: bool = False
) -> MixtralForCausalLM | AutoModelForCausalLM:
    args = get_args()

    if "Mixtral" in model_name:
        model = MixtralForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            max_position_embeddings=args.seq_length,
            # ref: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json#L19
            output_router_logits=args.output_router_logits,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            use_cache=use_cache,
        )

        return model  # type: ignore
    else:
        raise NotImplementedError("model not implemented")
