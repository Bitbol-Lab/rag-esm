# pytorch Dataset for the RAG model
# This is a custom dataset for the RAG model. It is a subclass of the torch.utils.data.Dataset class.
# It is used to load the data from the dataset and pass it to the model for training and testing.
import torch
from torch.utils.data import Dataset
import datasets
import numpy as np
from Levenshtein import distance, hamming
from Bio import SeqIO

from transformers import DataCollatorForLanguageModeling
from typing import Any, Tuple, Optional

def introduce_unks(seq, L):
    """For a sequence of length L1, introduce pads (unknown tokens) at random positions inside the sequence
    until it reaches length L.
    Example: seq = "AFPEED", L = 12 -> "<unk>AF<unk><unk>PEE<unk>D<unk><unk>"
    """
    L1 = len(seq)
    assert L1 <= L, f"Length of the sequence ({L1}) is larger than the desired length ({L}), {seq}"
    padded_seq = ["<unk>"] * L
    positions = sorted(np.random.choice(L, L1, replace=False))
    for pos, token in zip(positions, seq):
        padded_seq[pos] = token
    return "".join(padded_seq)

class RAGDataset(Dataset):
    def __init__(self,
                 split="train",
                 data_dir="../../../data/example_dataset",
                 tokenizer=None,
                 seed=42,
                 num_seq=1,
                 context_sampling="random",
                 pe_scores_threshold=None,
                 max_length=1024,
                 max_length_ctx=2048,
                 repeat_validation=1,
                 mask_different_res=None,
                 use_fixed_length_masked_input=None,
                 random_padding=False):
        
        self.data_dir = data_dir
        self.split = split
        self.data = []
        # load the data, if the validation data should be repeated (to have more precise average scores), concatenate the dataset
        self.load_data(repeat_validation=repeat_validation)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_ctx = max_length_ctx
        self.num_seq = num_seq
        self.context_sampling = context_sampling
        self.mask_different_res = mask_different_res
        # check that context_sampling is a valid option
        assert self.context_sampling in ["same-sequence", "random", "closest-hamming-random", "closest-hamming-order",
                                         "closest-homologs", "top-10", "exponential", "pe-score"], \
            f"Context sampling method {self.context_sampling} not implemented"
        self.pe_scores_threshold = pe_scores_threshold
        self.use_fixed_length_masked_input = use_fixed_length_masked_input
        self.random_padding = random_padding
        # set the seed
        np.random.seed(seed)

    def load_data(self, repeat_validation=1):
        # Load the data from the dataset
        dataset = datasets.load_from_disk(self.data_dir)
        self.data = dataset[self.split]
        if repeat_validation > 1 and self.split == "val":
            self.data = datasets.concatenate_datasets([self.data]*repeat_validation)

    def __len__(self):
        return len(self.data)

    def _sample_ctx_indices(self, msa_len, ind, num_seq, closest_sequences=None):
        """
            Method to sample the context indices for the current sequence. The method can sample
            the context sequences randomly from the MSA, from the closest 10% of sequences in the MSA
            (hamming distance, either randomly or closest), from the closest sequences in the MSA (homology).
        """
        if num_seq == 0:
            return []
        elif self.context_sampling == "same-sequence":
            # return the same sequence as the input
            return []
        elif self.context_sampling == "random":
            # completely random from the MSA
            return np.random.choice(msa_len, num_seq)
        elif self.context_sampling == "closest-hamming-random":
            # from the closest 10% of sequences in the MSA (hamming distance)
            return np.random.choice(closest_sequences, num_seq)
        elif self.context_sampling == "closest-hamming-order":
            return closest_sequences[:num_seq]
        elif self.context_sampling == "top-10":
            return np.random.choice(closest_sequences[:10], num_seq)
        elif self.context_sampling == "exponential":
            ind = sample_indices_exponentially(min(len(closest_sequences), 100), num_seq, lamb_correction=2)
            return closest_sequences[ind]
        elif self.context_sampling == "closest-homologs":
            # from the closest sequences in the MSA (homology), i.e. the indices close to `ind` in the MSA
            close_inds = np.arange(ind-num_seq, ind+num_seq+1)
            # exclude the current index
            close_inds = close_inds[close_inds != ind]
            # exclude out of bounds indices
            close_inds = close_inds[(close_inds >= 0) & (close_inds < msa_len)]
            # take only num_seq indices, the ones that are closest to `ind`
            close_inds = close_inds[np.argsort(np.abs(close_inds - ind))[:num_seq]]
            return close_inds
        elif self.context_sampling == "pe-score":
            if len(closest_sequences) > 0:
                return np.random.choice(closest_sequences, num_seq)
            else:
                print("No sequences with PE score below the threshold")
                return np.random.choice(msa_len, num_seq)
        else:
            raise NotImplementedError(f"Context sampling method {self.context_sampling} not implemented")
    
    def __getitem__(self, idx):
        # randomize max length
        # max_length, max_length_ctx = np.random.randint(50, self.max_length), np.random.randint(50, self.max_length_ctx)
        # keep same max length
        max_length, max_length_ctx = self.max_length, self.max_length_ctx
        item = self.data[idx]
        cluster_id = item["cluster_id"]
        msa_seq = item["msa"]
        # get a random index from the MSA
        ind = np.random.randint(len(msa_seq))
        if self.pe_scores_threshold is not None:
            # get only the sequences with a PE score below or equal to the threshold
            accepted_ids = [i for i, x in enumerate(item["PE_scores"]) if x is not None and x <= self.pe_scores_threshold]
            # get random index from the accepted sequences in the MSA
            ind = np.random.choice(accepted_ids) if len(accepted_ids) > 0 else ind
        # get sequence from the msa associated with the random index
        seq = msa_seq[ind]
        # sample a random number of context sequences from poisson distribution
        num_seq = self.num_seq # np.random.poisson(self.lambda_poisson) if self.lambda_poisson > 0 else 0
        # get context sequences
        if self.context_sampling in ["closest-hamming-random", "closest-hamming-order", "top-10", "exponential"]:
            closest_sequences = item["closest_sequences"][ind] if "closest_sequences" in item else None
        elif self.pe_scores_threshold is not None:
            closest_sequences = [el for el in accepted_ids if el != ind]
        else:
            closest_sequences = None
        ctx_ids = self._sample_ctx_indices(len(msa_seq), ind, num_seq, closest_sequences)
        ctx = [msa_seq[i] for i in ctx_ids]
        # normalized hamming distance between the sequence and the context sequences
        hamming_dist = [0] if self.context_sampling == "same-sequence" else [hamming(seq, el)/len(seq) for el in ctx]
        if (self.mask_different_res is not None) and len(ctx) == 1:
            different_res = degap_and_track_modifications(seq, ctx[0])
            if not self.mask_different_res:
                # invert the list to mask only the same residues
                different_res = [not el for el in different_res]
            # crop `different_res` to the maximum length - 2 (cls and eos tokens are added)
            different_res = different_res[:max_length-2]
            # add a False value to the beginning and end of the list to account for the cls and eos tokens
            different_res = [False] + different_res + [False]
        # remove gaps and make all uppercase, crop the sequences to the maximum length
        seq = seq.replace("-", "").upper()
        # crop the sequences to the maximum length - 2 (cls and eos tokens are added). Take a random crop
        ctx = [el.replace("-", "").upper() for el in ctx]
        for i, el in enumerate(ctx):
            if len(el) > max_length_ctx:
                crop_start = np.random.randint(0, len(el) - max_length_ctx)
                ctx[i] = el[crop_start:crop_start+max_length_ctx]
        # normalized levenshtein distance between the sequence and the context sequences
        levenshtein_dist = [0] if self.context_sampling == "same-sequence" else [distance(seq, el)/len(seq) for el in ctx]
        if self.random_padding:
            # add pads at random positions sampling from an exponential distribution
            # L = min(sample_indices_exponentially(len(seq), 1, lamb_correction=2)[0] + len(seq), max_length)
            # do the same but uniformly sample the length between len(seq) and len(seq)*2
            L = min(np.random.randint(len(seq), len(seq)*2), max_length)
            if len(seq)< max_length:
                # add pads at random positions only if the original sequence is shorter than the maximum length
                seq = introduce_unks(seq, L)        
        # crop the sequences to the maximum length - 2 (cls and eos tokens are added). Take a random crop
        if len(seq) > (max_length-2):
            crop_start = np.random.randint(0, len(seq) - max_length + 2)
            seq = seq[crop_start:crop_start+max_length-2]
        
        # tokenize the sequences
        tokenized = self.tokenizer([seq], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")["input_ids"]
        input_ids = tokenized.squeeze(0)
           
        if len(ctx) >= 1:
            tokenized_ctx = self.tokenizer(ctx, padding=False)["input_ids"]
        else:
            if self.context_sampling == "same-sequence" and num_seq == 1:
                tokenized_ctx = [input_ids]
                if self.use_fixed_length_masked_input:
                    new_seq = "<mask>"*self.use_fixed_length_masked_input
                    input_ids = self.tokenizer([new_seq], padding=True, return_tensors="pt")["input_ids"].squeeze(0)
            else:
                tokenized_ctx = [None]    
        return {"input_ids": input_ids,
                "context_ids": tokenized_ctx,
                "hamming_dist": hamming_dist,
                "levenshtein_dist": levenshtein_dist,
                "cluster_id": cluster_id,
                } | ({"is_res_same": ~torch.tensor(different_res)} if (self.mask_different_res is not None) and len(ctx) == 1 else {})

class DataCollatorForLanguageModelingWithDiffusion(DataCollatorForLanguageModeling):
    """ Subclass the DataCollatorForLanguageModeling to allow for the sequence diffusion
    objective to be used as well. This is done by adding the parameter diffusion_mlm to the
    init method.
    """
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15, pad_to_multiple_of=None,
                 tf_experimental_compile=False, return_tensors="pt", diffusion_mlm=False, training=True, random_padding=False):
        super().__init__(tokenizer, mlm=mlm, mlm_probability=mlm_probability, pad_to_multiple_of=pad_to_multiple_of,
                         tf_experimental_compile=tf_experimental_compile, return_tensors=return_tensors)
        self.diffusion_mlm = diffusion_mlm
        if diffusion_mlm:
            self.mlm = True
        self.training = training
        if random_padding:
            # <unk> should be removed from the special tokens
            self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES = [el for el in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES if el != 'unk_token']
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        If diffusion_mlm is True, sample mlm_probability from a beta distribution 80% of the time and uniformly 20% of the time.
        Otherwise, use the fixed mlm_probability.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # If diffusion_mlm is True, we sample using mlm_probability sampled from a beta distribution in 80% of the cases
        # and uniform probability in the remaining 20% cases. Otherwise we use the fixed mlm_probability.
        if self.diffusion_mlm:
            prob_beta = 0 if self.diffusion_mlm=="uniform" else torch.rand(1).item()
            new_mlm_probability = torch.rand(1).item() if prob_beta < 0.2 else torch.distributions.beta.Beta(3, 9).sample().item()
            probability_matrix = torch.full(labels.shape, new_mlm_probability)
        else:
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        if self.training:
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]
        else:
            # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
def RAG_data_collator(batch, data_collator_mlm, data_collator_pad):
    """
        Make a the batches for the RAG model, with the input_ids and context_ids collated separately.
        The input_ids are collated with the Masked Language Model (MLM) data collator.
        The context_ids are collated with the padding data collator (no masking), also, if there
        are no context_ids for a sample (None), they won't be collated and this is tracked by
        the ids_reorder_ctx that keeps the position of the sample in the input_ids for each context_ids.
    """
    # Prepare the batch in the format expected by the data collator
    input_ids = [item['input_ids'] for item in batch]
    context_ids = [(el, i) for i, item in enumerate(batch) for el in item['context_ids']]
    is_res_same = [item['is_res_same'] for item in batch] if 'is_res_same' in batch[0] else None

    # Make a list of cluster_ids
    cluster_ids = [item['cluster_id'] for item in batch] if 'cluster_id' in batch[0] else None
    
    # Create a list of dictionaries for the data collator
    if is_res_same is None:
        input_batch = [{'input_ids': ids} for ids in input_ids]
    else:
        input_batch = [{'input_ids': ids, 'special_tokens_mask': mask} for ids, mask in zip(input_ids, is_res_same)]
    context_batch = [{'input_ids': torch.Tensor(ids[0]).to(dtype=torch.int64)} for ids in context_ids if ids[0] is not None]
    ids_reorder_ctx = torch.tensor([ids[1] for ids in context_ids if ids[0] is not None], dtype=torch.int64)
    
    # Make tensor with the distances
    hamming_batch = [el for item in batch for el in item['hamming_dist']]
    levenshtein_batch = [el for item in batch for el in item['levenshtein_dist']]
    
    # Apply the data collator to input_ids and context_ids separately
    # input_ids is collated with the MLM data collator
    input_ids_batch = data_collator_mlm(input_batch)
    # context_ids is collated with the padding data collator (no masking)
    if len(context_batch) == 0:
        context_ids_batch = {'input_ids': None,
                             'attention_mask': None}
    else:
        context_ids_batch = data_collator_pad(context_batch)
    return {
        'input_ids': input_ids_batch['input_ids'],
        'attention_mask_input': input_ids_batch['attention_mask'],
        'context_ids': context_ids_batch['input_ids'],
        'attention_mask_context': context_ids_batch['attention_mask'],
        'hamming_dist': torch.tensor(hamming_batch, dtype=torch.float32),
        'levenshtein_dist': torch.tensor(levenshtein_batch, dtype=torch.float32),
        'cluster_ids': cluster_ids,
    } | ({'labels': input_ids_batch['labels']} if 'labels' in input_ids_batch else {})
    

def make_eval_datasets(data_dir, config, tokenizer):
    # Context sequences: same as the input sequence
    dataset_eval_same = RAGDataset(data_dir=data_dir,
                        tokenizer=tokenizer,
                        split="val",
                        seed=config.np_seed,
                        num_seq=config.num_seq,
                        context_sampling="same-sequence",
                        pe_scores_threshold=config.pe_scores_threshold,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        repeat_validation=config.repeat_validation)
    # Context sequences: None
    dataset_eval_none = RAGDataset(data_dir=data_dir,
                        tokenizer=tokenizer,
                        split="val",
                        seed=config.np_seed,
                        num_seq=0,
                        context_sampling="same-sequence",
                        pe_scores_threshold=config.pe_scores_threshold,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        repeat_validation=config.repeat_validation)
    # Context sequences: closest sequences in the MSA (hamming)
    dataset_eval_hord = RAGDataset(data_dir=data_dir,
                        tokenizer=tokenizer,
                        split="val",
                        seed=config.np_seed,
                        num_seq=config.num_seq,
                        context_sampling="closest-hamming-order",
                        pe_scores_threshold=config.pe_scores_threshold,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        repeat_validation=config.repeat_validation)
    # Context sequences: closest sequences in the MSA (homology)
    dataset_eval_homo = RAGDataset(data_dir=data_dir,
                        tokenizer=tokenizer,
                        split="val",
                        seed=config.np_seed,
                        num_seq=config.num_seq,
                        context_sampling="closest-homologs",
                        pe_scores_threshold=config.pe_scores_threshold,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        repeat_validation=config.repeat_validation)
    # Context sequences: random sequence sampled from the 10% closest ones (hamming)
    dataset_eval_hran = RAGDataset(data_dir=data_dir,
                        tokenizer=tokenizer,
                        split="val",
                        seed=config.np_seed,
                        num_seq=config.num_seq,
                        context_sampling="closest-hamming-random",
                        pe_scores_threshold=config.pe_scores_threshold,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        repeat_validation=config.repeat_validation)
    # Context sequences: random sequence from the ones with PE score below or equal to the threshold of 2
    dataset_eval_pe = RAGDataset(data_dir=data_dir,
                        tokenizer=tokenizer,
                        split="val",
                        seed=config.np_seed,
                        num_seq=config.num_seq,
                        context_sampling=config.context_sampling,
                        pe_scores_threshold=2,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        repeat_validation=config.repeat_validation)
    # Context sequences: same as training, Mask only the different residues
    dataset_eval_diff = RAGDataset(data_dir=data_dir,
                        tokenizer=tokenizer,
                        split="val",
                        seed=config.np_seed,
                        num_seq=config.num_seq,
                        context_sampling=config.context_sampling,
                        pe_scores_threshold=config.pe_scores_threshold,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        repeat_validation=config.repeat_validation,
                        mask_different_res=True)
    # Context sequences: same as training, Mask only the conserved residues
    dataset_eval_cons = RAGDataset(data_dir=data_dir,
                        tokenizer=tokenizer,
                        split="val",
                        seed=config.np_seed,
                        num_seq=config.num_seq,
                        context_sampling=config.context_sampling,
                        pe_scores_threshold=config.pe_scores_threshold,
                        max_length=config.max_length,
                        max_length_ctx=config.max_length_context,
                        repeat_validation=config.repeat_validation,
                        mask_different_res=False)
    return {"same-seq": dataset_eval_same,
            "no-seq": dataset_eval_none,
            "ham-ord": dataset_eval_hord,
            "ham-ran": dataset_eval_hran,
            "homologs": dataset_eval_homo,
            "pe-score": dataset_eval_pe,
            "mask-diff": dataset_eval_diff,
            "mask-cons": dataset_eval_cons
            }


def sample_indices_exponentially(N, num_samples, lamb_correction=1):
    """
    Samples indices using an exponential probability distribution efficiently for large N.
    
    Parameters:
    - N: Total number of elements to sample from (0 to N-1).
    - num_samples: Number of samples to draw.
    
    Returns:
    - A NumPy array of sampled indices.
    """
    # Generate uniform random numbers
    uniform_samples = np.random.uniform(0, 1, size=num_samples)
    # Transform uniform samples to exponential distribution via inverse CDF
    lamb = lamb_correction*np.log(N)/N
    sampled_indices = np.floor(-np.log(1 - uniform_samples) * (1 / lamb)).astype(int)
    # Ensure sampled indices are within bounds by setting the ones larger than N-1 to 0
    sampled_indices[sampled_indices >= N] = 0
    return sampled_indices

def degap_and_track_modifications(a,b):
    """
    Input: a,b fasta sequences in a3m format. They are aligned with gaps but have insertions as lowercase characters.
    Example input: a = "AabB--C---D", b = "AB--C-abcd--D". To be aligned, lowercase characters must be removed.
    This function internally aligns a and b by removing lowercase characters and returns the positions in the original a that are different from b
    when a and b are aligned. Then returns the degapped a and b (a and b without gaps and with lowercase characters promoted to uppercase),
    also returns a boolean list of the same length as the new a that is True if the corresponding position in a is different from b when a and b are aligned.
    """
    # Original a and b to be aligned
    a_orig = a
    b_orig = b
    # positions of non-lowercase characters in a and b
    a_pos = [i for i, c in enumerate(a) if not c.islower()]
    # b_pos = [i for i, c in enumerate(b) if not c.islower()]
    
    # align a and b by removing lowercase characters
    a = [c for c in a if not c.islower()]
    b = [c for c in b if not c.islower()]

    # positions in original a that are different from b when a and b are aligned
    res = [i for i,aa,bb in zip(a_pos,a,b) if aa!=bb and aa!="-"]
    res_a = "".join([aa for i,aa,bb in zip(a_pos,a,b) if aa!=bb and aa!="-"])

    # boolean list of all enties in a_orig that are different from b when a and b are aligned
    bol = [True if i in res else False for i, aa in enumerate(a_orig)]
    final_bol = [bo for bo, aa in zip(bol, a_orig) if aa!="-"]
    final_a = a_orig.replace("-", "").upper()
    # final_b = b_orig.replace("-", "").upper()
    
    # check that the entries of final_a True in final_bol are the same as the entries in res_a
    assert len(final_a) == len(final_bol), f"Error in degap_and_track_modifications: {len(final_a)} != {len(final_bol)}"
    tmp = "".join([aa for aa, bo in zip(final_a, final_bol) if bo])
    assert tmp == res_a, f"Error in degap_and_track_modifications: {tmp} != {res_a}" + f"\n{final_a}\n{final_bol}"
    
    return final_bol

class RAGDatasetFromFasta(Dataset):
    """
    Dataset class for the RAG model that loads the input and the context sequences from fasta files.
    """
    def __init__(self,
                 input_path="../../../data/example_dataset_file/input_sequences.fasta",
                 context_path="../../../data/example_dataset_file/context_sequences.fasta",
                 tokenizer=None,
                 seed=42,
                 context_sampling="same-position"):
        # Load the fasta file with the input sequences
        with open(input_path, "r") as f:
            self.input_seqs = [{"seq": str(seq.seq), "id":str(seq.id)}
                                for seq in SeqIO.parse(f, "fasta")]
        # Load the a3m file with the context sequences
        with open(context_path, "r") as f:
            self.context_seqs = [{"seq": str(seq.seq), "id":str(seq.id)}
                                  for seq in SeqIO.parse(f, "fasta")]
        self.tokenizer = tokenizer
        self.context_sampling = context_sampling
        self.seed = seed
        
    def __len__(self):
        return len(self.input_seqs)
    
    def _sample_ctx_indices(self, msa_len, idx):
        """
            Method to sample the context indices for the current sequence. The method can sample
            the context sequences randomly from the MSA, from the closest 10% of sequences in the MSA
            (hamming distance, either randomly or closest), from the closest sequences in the MSA (homology).
        """
        if self.context_sampling == "random":
            # completely random from the MSA
            print("Random context sampling works only if the context sequences are homologs of all the input sequences")
            return np.random.choice(msa_len, 1)
        elif self.context_sampling == "same-position" and len(self.context_seqs) > idx:
            # return the sequence in the same position as the input sequence
            return [idx]
        else:
            raise NotImplementedError(f"Context sampling method {self.context_sampling} not implemented")
    
    def __getitem__(self, idx):
        # get the input sequence
        data = self.input_seqs[idx]
        seq, seq_id = data["seq"], data["id"]
        # get context sequence
        ctx_ids = self._sample_ctx_indices(len(self.context_seqs), idx)
        ctx = self.context_seqs[ctx_ids[0]]["seq"]
        # remove gaps, crop the sequences to the maximum length
        seq = seq.replace("-", "<mask>")
        ctx = ctx.replace("-", "")
        tokenized = self.tokenizer([seq], padding=False, return_tensors="pt")["input_ids"]
        input_ids = tokenized.squeeze(0)
        tokenized_ctx = self.tokenizer([ctx], padding=False)["input_ids"]
        return {"input_ids": input_ids,
                "context_ids": tokenized_ctx,
                "hamming_dist": [0],
                "levenshtein_dist": [0],
                "cluster_id": seq_id}
        