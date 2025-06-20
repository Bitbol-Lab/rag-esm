from pathlib import Path
from datetime import datetime
import json
import yaml
import argparse
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rag_esm.utils.generate import denoise, score_sequences
from rag_esm.modules.model import RAGModel, RAGConfig
from rag_esm.modules.dataloaders import RAGDataset, RAG_data_collator, RAGDatasetFromFasta, DataCollatorForLanguageModelingWithDiffusion

import torch
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, EsmForMaskedLM
from torch.utils.data import DataLoader

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

use_one_gpu = input("Do you want to use just one gpu? If yes, type the number of the gpu. If no, type -1.\n")
import os
if int(use_one_gpu) >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = use_one_gpu

def worker_init_fn(worker_id):
    np.random.seed(args.np_seed + worker_id)
        
def main(args):
    # Basic information
    # load yaml file from path
    training_config = yaml.safe_load(open(args.training_config_path, "r"))
    training_config = OmegaConf.create(training_config)
    output_dir = Path(args.outputs_dir)
    print(f"Output directory: {output_dir}")
    print(f"Working directory: {Path.cwd()}")
    nr_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {nr_gpus}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name_or_path)
    copy_tokenizer = AutoTokenizer.from_pretrained(training_config.model_name_or_path)
    copy_tokenizer.SPECIAL_TOKENS_ATTRIBUTES = [el for el in copy_tokenizer.SPECIAL_TOKENS_ATTRIBUTES if el != 'mask_token']
    if args.use_random_padding:
        copy_tokenizer.SPECIAL_TOKENS_ATTRIBUTES = [el for el in copy_tokenizer.SPECIAL_TOKENS_ATTRIBUTES if el != 'unk_token']
    model_config = RAGConfig(model_name=training_config.model_name_or_path,
                            use_cross_attention=training_config.use_cross_attention,
                            freeze_encoder=training_config.freeze_encoder,
                            train_only_cross_attention=training_config.train_only_cross_attention,
                            skip_cross_ratio=training_config.skip_cross_ratio,
                            layers_with_cross_attention=training_config.layers_with_cross_attention,
                            use_flash_attention=training_config.use_flash_attention,
                            gate_selection_function=training_config.gate_selection_function,
                            tie_weights_encoder_decoder=False, # no need to tie the weights since we are not training
                            )
    # model = RAGModel(model_config).to(args.device)
    # model.load_state_dict(torch.load(Path(args.checkpoint_dir) / "pytorch_model.bin", map_location=args.device))
    model = RAGModel.from_pretrained(Path(args.checkpoint_dir), config=model_config).to(args.device)
    # Set model to eval mode
    model.eval()        
    print(f"Number of parameters (total): {sum(p.numel() for p in model.parameters())}")

    # Load data
    torch.manual_seed(args.torch_seed)
    data_collator_pad = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    data_collator_mlm = DataCollatorForLanguageModelingWithDiffusion(tokenizer=tokenizer,
                                                                    mlm=True,
                                                                    mlm_probability=args.mlm_probability,
                                                                    diffusion_mlm=False,
                                                                    training=False,
                                                                    return_tensors="pt")
    if not args.use_fasta_files:
        print("Using dataset")
        if args.use_fixed_length_masked_input:
            args.context_sampling = "same-sequence"
        if args.context_sampling == "pe-score" and args.pe_scores_threshold is None:
            raise ValueError("If you set context_sampling to pe-score you must provide a pe_scores_threshold")
        custom_collate_fn = lambda x: RAG_data_collator(x, data_collator_mlm, data_collator_pad)
        dataset = RAGDataset(data_dir=Path(args.dataset_path),
                            tokenizer=tokenizer,
                            split="test",
                            seed=args.np_seed,
                            num_seq=training_config.num_seq,
                            context_sampling=args.context_sampling,
                            pe_scores_threshold=args.pe_scores_threshold,
                            max_length=2048,
                            max_length_ctx=2048,
                            repeat_validation=1,
                            use_fixed_length_masked_input=args.use_fixed_length_masked_input)
    elif args.input_path is not None and args.context_path is not None:
        print("Using fasta files")
        # use data_collator_pad instead of data_collator_mlm for the input_ids because
        # the input_ids are already masked
        custom_collate_fn = lambda x: RAG_data_collator(x, data_collator_pad, data_collator_pad)
        dataset = RAGDatasetFromFasta(input_path=args.input_path,
                                      context_path=args.context_path,
                                      tokenizer=tokenizer,
                                      seed=args.np_seed,
                                      context_sampling="same-position")
    else:
        raise ValueError("If you set use_fasta_files to True then input_path and context_path must be provided")
    # load data into device and create dataloader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=False,
                            collate_fn=custom_collate_fn,
                            worker_init_fn=worker_init_fn,
                            persistent_workers=True,
                            pin_memory=True)

    strategy = {"top_k": args.top_k, "top_p": args.top_p, "min_p": args.min_p,"temperature": args.temperature,
                "error_correction": args.error_correction, "entropy_unmasking": args.entropy_unmasking,
                "start_error_correction": args.start_error_correction}
    print(f"Strategy: {strategy}")
    # Generate sequences
    print("Generating sequences...")
    # while True:
    with torch.no_grad():
        gen_seq = []
        orig_seq = []
        ctx_seq = []
        perp_seq = []
        clus_ids = []
        
        for j in range(args.repeat_validation):
            for i, batch in enumerate(dataloader):
                if args.use_fixed_length_masked_input:
                    batch.pop("labels")
                    args.iterations = min(args.iterations, args.use_fixed_length_masked_input)
                # Shape new_seq: (iterations+1, number_of_sequences, sequence_length) or (2, number_of_sequences, sequence_length)
                # Shape perplexities: (number_of_sequences)
                # Shape context_ids: (number_of_sequences, context_length)
                new_seq, perplexities = denoise(model,
                    tokenizer,
                    batch["input_ids"].to(args.device),
                    batch["attention_mask_input"].to(args.device),
                    batch["context_ids"].to(args.device) if batch["context_ids"] is not None else None,
                    batch["attention_mask_context"].to(args.device) if batch["attention_mask_context"] is not None else None,
                    strategy=strategy,
                    use_random_padding=args.use_random_padding,
                    iterations=args.iterations,
                    save_intermediate_steps=args.save_intermediate_steps,
                    labels_to_debug=None) # batch["labels"])
                # Replace masked tokens with the ground truth
                if "labels" in batch:
                    labels = batch["labels"]
                    # Shape input_ids: (number_of_sequences, sequence_length)
                    input_ids = batch["input_ids"].clone()
                    mask = input_ids == tokenizer.mask_token_id
                    input_ids[mask] = labels[(labels!=-100)]
                # List of cluster_ids
                if args.repeat_validation > 1:
                    clus_ids += [f"{clus_name}_{j}" for clus_name in batch["cluster_ids"]]
                else:
                    clus_ids += batch["cluster_ids"]
                # List of length number_of_sequences, each element is a list (of lenght iterations+1 or 2) of strings
                gen_seq += [[copy_tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "") for seq in new_seq[:,i,:]] for i in range(new_seq.shape[1])]
                # List of length number_of_sequences, each element is a float
                perp_seq += perplexities.cpu().numpy().tolist()
                # Lists of length number_of_sequences, each element is a string
                if "labels" in batch:
                    orig_seq += [copy_tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "") for seq in input_ids]
                ctx_seq += [copy_tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "") for seq in batch["context_ids"]]
                # Clear memory
                del new_seq, perplexities
                torch.cuda.empty_cache()
                # Break if in debug mode
                if args.debug and i > 0:
                    break

    # Get timestamp in the format YY    YY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = output_dir / f"{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # When the effective number of denoising iterations is less than args.iterations, we need to pad the generated sequences
    # so that the array has shape (args.iterations+1, number_of_sequences), to do so we repeat the first element of the sequence
    # to the beginning of the sequence until the length is args.iterations+1
    gen_seq =  np.array([[el[0]]*(args.iterations+1-len(el))+el for el in gen_seq])
    # Save generated sequences
    np.save(output_dir / "cluster_ids.npy", np.array(clus_ids)) 
    np.save(output_dir / "generated_sequences.npy", gen_seq)
    np.save(output_dir / "context_sequences.npy", np.array(ctx_seq))
    np.save(output_dir / "generation_perplexities.npy", np.array(perp_seq))
    # Save generated and context sequences as fasta files
    records , records_ctx = [], []
    for i, (seq_g, seq_c) in enumerate(zip(gen_seq[:,-1], ctx_seq)):
        # remove all <unk> tokens from the generated sequence
        seq_g = seq_g.replace("<unk>", "")
        records.append(SeqRecord(Seq(seq_g), id=clus_ids[i], description=""))
        records_ctx.append(SeqRecord(Seq(seq_c), id=clus_ids[i], description=""))
    # write fasta
    SeqIO.write(records, open(output_dir / "generated_sequences.fasta", "w"), "fasta")
    SeqIO.write(records_ctx, open(output_dir / "context_sequences.fasta", "w"), "fasta")
    if not args.use_fasta_files:
        np.save(output_dir / "original_sequences.npy", np.array(orig_seq))
        records_orig = []
        for i, seq_o in enumerate(orig_seq):
            records_orig.append(SeqRecord(Seq(seq_o), id=clus_ids[i], description=""))
        SeqIO.write(records_orig, open(output_dir / "original_sequences.fasta", "w"), "fasta")
    # Compute perplexities of generated and context sequences
    if args.compute_perplexities:
        esm_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(args.device)
        esm_model.eval()
        perplexity_esm, perplexity_esm_ctx = [], []
        perplexity_rag = []
        for i, (g_seq, c_seq) in tqdm(enumerate(zip(gen_seq[:,-1], ctx_seq)), desc="Scoring sequences with ESM2"):
            # remove all <unk> tokens from the generated sequence
            g_seq = g_seq.replace("<unk>", "")
            tokenized_g_seq = tokenizer(g_seq, return_tensors="pt")["input_ids"].to(args.device)
            tokenized_c_seq = tokenizer(c_seq, return_tensors="pt")["input_ids"].to(args.device)
            loss_rag_esm, loss_esm = score_sequences(model,
                                                     esm_model,
                                                     tokenizer,
                                                     tokenized_g_seq,
                                                     [tokenized_c_seq])
            torch.cuda.empty_cache()
            _, loss_esm_ctx = score_sequences(None,
                                              esm_model,
                                              tokenizer,
                                              tokenized_c_seq,
                                              [])
            perplexity_esm.append(np.exp(loss_esm))
            perplexity_rag.append(np.exp(loss_rag_esm))
            perplexity_esm_ctx.append(np.exp(loss_esm_ctx))
            torch.cuda.empty_cache()
        # Save perplexities
        np.save(output_dir / "rag-esm_perplexities_gen.npy", np.array(perplexity_rag))
        np.save(output_dir / "esm2_perplexities_gen.npy", np.array(perplexity_esm))
        np.save(output_dir / "esm2_perplexities_ctx.npy", np.array(perplexity_esm_ctx))
    
    # Save metadata as dictionary
    metadata = {"timestamp": timestamp,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "min_p": args.min_p,
                "temperature": args.temperature,
                "error_correction": args.error_correction,
                "start_error_correction": args.start_error_correction,
                "entropy_unmasking": args.entropy_unmasking,
                "iterations": args.iterations,
                "save_intermediate_steps": args.save_intermediate_steps,
                "mlm_probability": args.mlm_probability if not args.use_fasta_files else None,
                "use_fasta_files": args.use_fasta_files,
                "context_sampling": args.context_sampling if not args.use_fasta_files else "same-position",
                "pe_scores_threshold": args.pe_scores_threshold,
                "number_of_sequences": len(gen_seq)}
    if args.use_fixed_length_masked_input:
        metadata["use_fixed_length_masked_input"] = args.use_fixed_length_masked_input
        metadata["context_sampling"] = "use-fixed-length-masked-input"
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    # Save also the training config as yaml
    with open(output_dir / "training_config.yaml", "w") as f:
        yaml.dump(training_config, f)
    print(f"Output saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--outputs_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=False, default=None,)
    parser.add_argument("--input_path", type=str, default=None, help="Path to the fasta file with the input sequences, to specify masked positions \
        use the special token `<mask>` in the sequence or the gap symbol: `-`")
    parser.add_argument("--context_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument("--np_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--error_correction", type=str, default=None, choices=["all-masked", "prev-unmasked", "all-residues", None])
    parser.add_argument("--start_error_correction", type=int, default=0)
    parser.add_argument("--entropy_unmasking", action="store_true", default=False)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--use_random_padding", action="store_true", default=False)
    parser.add_argument("--save_intermediate_steps", action="store_true", default=False)
    parser.add_argument("--mlm_probability", type=float, default=1.0)
    parser.add_argument("--use_fasta_files", action="store_true", default=False)
    parser.add_argument("--context_sampling", type=str, default="random", choices=["random", "same-sequence", "closest-hamming-random",
                                                                                   "pe-score", "closest-hamming-order", "top-10", "closest-homologs"])
    parser.add_argument("--use_fixed_length_masked_input", type=int, default=None)
    parser.add_argument("--pe_scores_threshold", type=float, default=None)
    parser.add_argument("--repeat_validation", type=int, default=1)
    parser.add_argument("--compute_perplexities", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(args)

# python src/rag_esm/sample.py --training_config_path=outputs/generation/ESM150M_diffusion/config.yaml --checkpoint_dir=outputs/esm2_t30_150M_UR50D/2025-01-16_13:08:28/checkpoint-21000 --outputs_dir=outputs/generation/ESM150M_diffusion --dataset_path=data/OpenProteinSet_uniclust30-filtered --batch_size=8 --top_k=10 --error_correction=all-masked --entropy_unmasking --save_intermediate_steps --mlm_probability=1.0 --iterations=100 --compute_perplexities
# python src/rag_esm/sample.py --training_config_path=outputs/generation/ESM150M_diffusion/config.yaml --checkpoint_dir=outputs/esm2_t30_150M_UR50D/2025-03-06_19:31:39/checkpoint-10920 --outputs_dir=outputs/generation/ESM150M_diffusion --dataset_path=data/OpenProteinSet_uniclust30-filtered --batch_size=32 --top_k=10 --error_correction=all-masked --use_random_padding --start_error_correction=20 --save_intermediate_steps --mlm_probability=1.0 --iterations=100 --compute_perplexities --use_fasta_files --input_path=/home/sgarboss/Documents/rag-esm/outputs/generation/ESM150M_diffusion/input_sequences_scaffolding.fasta --context_path=/home/sgarboss/Documents/rag-esm/outputs/generation/ESM150M_diffusion/context_sequences_scaffolding.fasta