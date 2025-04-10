import logging
from pathlib import Path
import json

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from rag_esm.modules.model import RAGModel, RAGConfig
from rag_esm.modules.dataloaders import RAGDataset, DataCollatorForLanguageModelingWithDiffusion, RAG_data_collator, make_eval_datasets
from rag_esm.utils.trainer import TrainerCustomCallback, WandbCustomCallback, ProfilingTrainer
from rag_esm.utils.metrics import compute_metrics

import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, get_constant_schedule_with_warmup

# Hydra sets up the logger automatically.
logger = logging.getLogger(__name__)

use_one_gpu = input("Do you want to use just one gpu? If yes, type the number of the gpu. If no, type -1.\n")
import os
if int(use_one_gpu) >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = use_one_gpu

def train(config: DictConfig) -> None:
    # Initialize wandb
    wandb.init(
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        project=config.wandb.project,
        tags=config.wandb.tags,
        anonymous=config.wandb.anonymous,
        mode=config.wandb.mode,
        dir=Path(config.wandb.dir).absolute(),
    )
    
    # Basic information
    output_dir = Path(config.outputs_dir) / Path(config.model_name_or_path).name /config.run_name
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Running with config: \n{OmegaConf.to_yaml(config, resolve=True)}")
    nr_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {nr_gpus}")
    # np_gen = np.random.default_rng(config.np_seed)
    
    # Load tokenizer and model
    torch.manual_seed(config.torch_seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model_config = RAGConfig(model_name=config.model_name_or_path,
                             use_cross_attention=config.use_cross_attention,
                             freeze_encoder=config.freeze_encoder,
                             train_only_cross_attention=config.train_only_cross_attention,
                             skip_cross_ratio=config.skip_cross_ratio,
                             layers_with_cross_attention=config.layers_with_cross_attention,
                             use_flash_attention=config.use_flash_attention,
                             gate_selection_function=config.gate_selection_function,
                             dropout=config.dropout if not config.evaluate_only else 0.0,
                             tie_weights_encoder_decoder=config.tie_weights_encoder_decoder,
                             rescale_loss_diffusion=config.rescale_loss_diffusion if not config.evaluate_only else False,
                             take_average_embeddings=config.take_average_embeddings,)
    # model = RAGModel(model_config).to(config.device)
    if (config.evaluate_only or config.load_pretrained) and config.checkpoint_dir is not None:
        print(f"Loading model from {config.checkpoint_dir}")
        # model.load_state_dict(torch.load(Path(config.checkpoint_dir) / "pytorch_model.bin", map_location=config.device))
        model = RAGModel.from_pretrained(Path(config.checkpoint_dir), config=model_config).to(config.device)
        if config.tie_weights_encoder_decoder:
            model.tie_weights_enc_dec()
            model.test_tied_weights()
    else:
        model = RAGModel(model_config).to(config.device)
    
    # Load data
    torch.manual_seed(config.torch_seed)
    data_collator_pad = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    data_collator_mlm = DataCollatorForLanguageModelingWithDiffusion(tokenizer=tokenizer,
                                                                     mlm=config.mlm,
                                                                     mlm_probability=config.mlm_probability,
                                                                     diffusion_mlm=config.diffusion_mlm,
                                                                     return_tensors="pt",
                                                                     random_padding=config.random_padding)
    custom_collate_fn = lambda x: RAG_data_collator(x, data_collator_mlm, data_collator_pad)
    dataset_train = RAGDataset(data_dir=Path(config.data_dir) / config.dataset_name,
                        tokenizer=tokenizer,
                        split="train",
                        seed=config.np_seed,
                        num_seq=config.num_seq,
                        context_sampling=config.context_sampling,
                        pe_scores_threshold=config.pe_scores_threshold,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        random_padding=config.random_padding
                        )
    dataset_eval = RAGDataset(data_dir=Path(config.data_dir) / config.dataset_name,
                        tokenizer=tokenizer,
                        split="val",
                        seed=config.np_seed,
                        num_seq=config.num_seq,
                        context_sampling=config.context_sampling,
                        pe_scores_threshold=config.pe_scores_threshold,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        repeat_validation=config.repeat_validation,
                        random_padding=False,
                        )
    
    if config.evaluate_only:
        all_dataset_eval = make_eval_datasets(Path(config.data_dir) / config.dataset_name,
                                            config,
                                            tokenizer)
        # add dataset_eval to all_dataset_eval
        all_dataset_eval["same-as-train"] = dataset_eval
    else:
        if config.repeat_validation>1:
            raise ValueError("repeat_validation should be 1 if not evaluating only.")
        all_dataset_eval = dataset_eval
    
    # Log training information
    steps_per_epoch = len(dataset_train) // config.per_device_batch_size // config.gradient_accumulation_steps // nr_gpus
    logger.info(f"Number of training steps per epoch: {steps_per_epoch}")
    logging_steps = steps_per_epoch // config.logging_events_per_epoch
    logger.info(f"Logging every {logging_steps} steps, i.e. {config.logging_events_per_epoch} times per epoch")
    eval_steps = steps_per_epoch // config.eval_events_per_epoch
    logger.info(f"Evaluating every {eval_steps} steps, i.e. {config.eval_events_per_epoch} times per epoch")
    save_steps = steps_per_epoch // config.save_events_per_epoch
    logger.info(f"Saving every {save_steps} steps, i.e. {config.save_events_per_epoch} times per epoch")

    # Training arguments
    trainer_args = TrainingArguments(
        eval_strategy="steps",
        output_dir=output_dir,
        logging_dir=output_dir,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=8,
        bf16=config.use_bf16,
        max_grad_norm=config.max_grad_norm,
        run_name=config.run_name,
        # torch_compile=True,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        #do not use safetensors because they do not work when using tied weights
        save_safetensors=False,
        report_to="wandb",
        push_to_hub=False,
    )
    # specify parameters to optimize based on the model configuration
    parameters_to_optimize = [{"params": param, "lr": config.lr_cross_attention}
                                for name, param in model.named_parameters() if "crossattention" in name]
    if not config.train_only_cross_attention:
        if config.freeze_encoder:
            parameters_to_optimize += [{"params": param, "lr": config.learning_rate}
                                        for name, param in model.decoder.named_parameters() if "crossattention" not in name]
        else:
            parameters_to_optimize += [{"params": param, "lr": config.learning_rate}
                                        for name, param in model.named_parameters() if "crossattention" not in name]
        
            
    # optimizer (train only parameters with "crossattention" in name if specified, train only decoder if specified)
    optimizer = torch.optim.AdamW(parameters_to_optimize,
                                  lr=config.learning_rate,
                                  betas=(config.adam_beta1, config.adam_beta2),
                                  weight_decay=config.weight_decay)
    logger.info(f"Number of trainable parameters: {sum(sum(p.numel() for p in params['params']) for params in parameters_to_optimize)}")
    logger.info(f"Number of parameters (total): {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Number of parameters with requires_grad=True: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"Number of parameters (decoder): {sum(p.numel() for p in model.decoder.parameters())}")
    # if the model has an encoder, log the number of parameters
    if model.encoder is not None:
        logger.info(f"Number of parameters (encoder): {sum(p.numel() for p in model.encoder.parameters())}")
    # scheduler warmup+constant learning rate
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps)
    
    trainer = Trainer(
        optimizers=(optimizer, scheduler),
        model=model,
        args=trainer_args,
        data_collator=custom_collate_fn,
        train_dataset=dataset_train,
        eval_dataset=all_dataset_eval,#{"val": dataset_eval, "same-val": dataset_eval_same},
        callbacks=[TrainerCustomCallback()],
        compute_metrics=compute_metrics,
    )

    # add wandb callback
    wandb_callback = WandbCustomCallback(trainer, show_cross_attentions=config.show_cross_attentions)
    trainer.add_callback(wandb_callback)
    
    assert trainer.args._n_gpu == nr_gpus, "Number of gpus used is not the same as the number of gpus available"
    
    #  evaluate model before training
    eval_results = trainer.evaluate()
    if config.evaluate_only:
        for name_dat in ["ham-ord", "ham-ran"]:
            for mlm_prob in [0.25, 0.50, 0.75]:
                data_collator_mlm.mlm_probability = mlm_prob
                data_collator_mlm.diffusion_mlm = False
                trainer.data_collator = lambda x: RAG_data_collator(x, data_collator_mlm, data_collator_pad)
                eval_results = eval_results | trainer.evaluate({f"{name_dat}_mlm-{mlm_prob}": all_dataset_eval[name_dat]})
        data_collator_mlm.mlm_probability = config.mlm_probability
        data_collator_mlm.diffusion_mlm = config.diffusion_mlm
        trainer.data_collator = custom_collate_fn
        json.dump(eval_results, open(output_dir / "eval_results.json", "w"))
        # save the config as yaml file
        with open(output_dir / "config.yaml", "w") as f:
            OmegaConf.save(config, f)
    logger.info(f"Evaluation results: {eval_results}")
    
    if not config.evaluate_only:
        # train model
        trainer.train(resume_from_checkpoint=config.checkpoint_dir if config.resume_from_checkpoint else None)

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    if config.evaluate_only:
        with torch.no_grad():
            train(config)
    else:
        train(config)

if __name__ == "__main__":
    main()
