import os
import sys

import accelerate
import torch
import torch.distributed as torch_distributed
import torch.optim as optim
import wandb
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, set_seed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import set_z3_leaf_modules  # mixtral
from torch.optim.lr_scheduler import StepLR
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from llama_recipes.arguments import parse_args
from llama_recipes.get_fsdp import get_sharding_strategy
from llama_recipes.get_models import get_model
from llama_recipes.optimizer import WarmupCosineAnnealingLR
from llama_recipes.utils.checkpoint import (
    get_latest_iteration,
    load_model_state_dict,
    load_optimizer_state_dict,
    load_rng_state_dict,
    load_scheduler_state_dict,
)
from llama_recipes.utils.distributed import get_local_rank, get_rank, is_rank_0, print_rank_0, set_mpi_env
from llama_recipes.utils.train_utils import clear_gpu_cache, print_model_size, setup_environ_flags, train
from megatron_lm.megatron.global_vars import set_global_variables

current_path: str = os.getcwd()
sys.path.append(f"{current_path}/llama-recipes/src/")


def main() -> None:
    # initialize
    args = parse_args()
    set_global_variables(args=args)

    # Set the seeds for reproducibility
    set_seed(seed=args.seed)

    # Distributed args.
    if args.use_mpi:
        set_mpi_env()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    args.rank = rank
    args.world_size = world_size
    args.gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * world_size)

    get_accelerator().set_device(get_local_rank())  # type: ignore

    # torch_distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    deepPlugin = DeepSpeedPlugin(
        hf_ds_config=args.zero_config,
        zero3_init_flag=True if args.zero_stage == 3 else False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.grad_clip_norm,
        zero_stage=args.zero_stage,
    )
    accelerator = Accelerator(
        mixed_precision="bf16" if args.bf16 else "fp16",
        deepspeed_plugin=deepPlugin,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        # ref: https://github.com/huggingface/accelerate/issues/2142
        # ref: https://huggingface.co/docs/accelerate/concept_guides/performance#learning-rates
        dataloader_config=accelerate.DataLoaderConfiguration(
            even_batches=False,
        ),
    )

    # wandb setting
    if args.wandb_name is not None and is_rank_0():
        import datetime

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        wandb_setting: dict = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": args.wandb_name,
            "config": vars(args),
        }
        wandb.init(**wandb_setting)

    if torch_distributed.is_initialized():
        torch.cuda.set_device(get_local_rank())  # type: ignore
        clear_gpu_cache(get_local_rank())  # type: ignore
        setup_environ_flags(get_rank())  # type: ignore

    iteration: int = get_latest_iteration(args.load)
    args.iteration = iteration
    torch_distributed.barrier()

    # random seed
    if args.load:
        load_rng_state_dict(args.load)
        torch_distributed.barrier()

    if args.continual_pretraining:
        # dataset
        from llama_recipes.datasets.pretrain_dataset import build_train_valid_test_datasets
        from megatron_lm.megatron.data.data_samplers import build_pretraining_data_loader

        train_dataset, validation_dataset, test_dataset = build_train_valid_test_datasets()

        args.consumed_train_samples = args.global_batch_size * args.iteration
        args.consumed_valid_samples = args.global_batch_size * (args.iteration // args.eval_interval) * args.eval_iters

        train_dataloader = build_pretraining_data_loader(
            dataset=train_dataset,
            consumed_samples=args.consumed_train_samples,
        )
        validation_dataloader = build_pretraining_data_loader(
            dataset=validation_dataset,
            consumed_samples=args.consumed_valid_samples,
        )
        torch_distributed.barrier()
    elif args.instruction_tuning:
        from transformers import AutoTokenizer

        from llama_recipes.utils.instruction_tuning import get_instruction_tuning_dataloader

        hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_transformer_model_dir)

        if args.instruction_tuning:
            train_dataloader = get_instruction_tuning_dataloader(
                tokenizer=hf_tokenizer,  # type: ignore
                data_path=args.instruction_train_data_path,
                train=True,
            )
            validation_dataloader = get_instruction_tuning_dataloader(
                tokenizer=hf_tokenizer,  # type: ignore
                data_path=args.instruction_valid_data_path,
            )

            args.train_iters = args.instruction_dataset_size // args.global_batch_size * args.epoch
            args.lr_decay_iters = args.train_iters
            args.lr_warmup_iters = args.lr_decay_iters // 10
            args.save_sampler_state = True
            if rank == 0:
                from llama_recipes.utils.wandb_utils import update_iter_info

                update_iter_info()
    else:
        raise ValueError("Invalid training mode")

    use_cache = False
    model = get_model(model_name=args.base_model, use_cache=use_cache)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # type: ignore
    model.enable_input_require_grads()  # type: ignore
    print_rank_0("Gradient checkpointing enable")

    print_model_size(model, args.base_model, rank)  # type: ignore

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if args.bf16:
        model.to(torch.bfloat16)  # type: ignore
    elif args.fp16:
        model.to(torch.float16)  # type: ignore

    set_z3_leaf_modules(  # z3_leaf
        model=model, leaf_module_classes=[MixtralSparseMoeBlock]  # type: ignore
    )
    model.train()  # type: ignore

    optimizer = optim.AdamW(
        model.parameters(),  # type: ignore
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    if args.lr_decay_style == "cosine":
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_iterations=args.lr_warmup_iters,
            decay_iterations=args.lr_decay_iters,
            max_iterations=args.train_iters,
            eta_min=args.min_lr,
        )
    else:
        scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

    if args.load:
        load_scheduler_state_dict(scheduler, args.load)  # type: ignore

    # ref: https://github.com/microsoft/DeepSpeed/pull/5008#issuecomment-1910607845
    model, optimizer, _, _, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        validation_dataloader,
        scheduler,
    )
    if args.load:
        load_model_state_dict(model, args.load)  # type: ignore

    # Start the training process
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=validation_dataloader,
        optimizer=optimizer,  # type: ignore
        lr_scheduler=scheduler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        accelerator=accelerator,
        local_rank=get_local_rank(),
        rank=get_rank(),
    )


if __name__ == "__main__":
    main()
