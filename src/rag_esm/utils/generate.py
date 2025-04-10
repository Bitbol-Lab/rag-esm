import torch
from tqdm import tqdm

def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for non-top-k values to -inf."""
    # Get the threshold value for top-k
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    # Mask logits below the threshold
    logits = logits.masked_fill(logits < threshold, float("-Inf"))
    return logits

def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for non-top-p values to -inf."""
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # Compute cumulative probabilities
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Mask logits where cumulative probability exceeds top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    # Scatter back to original logits
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-Inf"))
    return logits

def modify_logits_for_min_p_filtering(logits, min_p):
    """Set logits below min_p to -inf."""
    # Threshold logits
    indices_to_remove = logits.softmax(dim=-1) < min_p
    # Keep always at least one token, i.e., the token with the highest probability
    indices_to_remove[:, logits.argmax(dim=-1)] = False
    # Mask logits below the threshold
    logits = logits.masked_fill(indices_to_remove, float("-Inf"))
    return logits

def sample(
    logits: torch.Tensor, 
    top_k: int = 1.0, 
    top_p: float = 0.0, 
    min_p: float = 0.0, 
    temperature: float = 1.0,
    **kwargs
) -> torch.Tensor:
    """
    Sample from logits with specified constraints.

    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, vocab_size).
        top_k (int): Number of top logits to consider for sampling.
        top_p (float): Cumulative probability threshold for nucleus sampling.
        min_p (float): Minimum probability threshold for filtering logits.
        temperature (float): Scaling factor for logits.
    
    If `top_k`, `top_p`, and `min_p` are all zero, this function samples from the full distribution.
    With the default settings, this function performs greedy decoding.

    Returns:
        torch.Tensor: Indices of sampled tokens, shape (batch_size).
    """
    assert temperature > 0.0, "`temperature` must be positive."
    assert logits.ndim == 2, "Logits must have shape [batch_size, vocab_size]."
    assert top_k >= 0, "`top_k` must be non-negative."
    
    if top_k == 1:  # Greedy decoding
        probabilities = logits.softmax(dim=-1)
        return logits.argmax(dim=-1), probabilities.max(dim=-1).values
    logits = logits.clone()  # Avoid in-place modifications
    if top_k > 0:
        logits = modify_logits_for_top_k_filtering(logits, top_k)
    if 0.0 < top_p < 1.0:
        logits = modify_logits_for_top_p_filtering(logits, top_p)
    if 0.0 < min_p < 1.0:
        logits = modify_logits_for_min_p_filtering(logits, min_p)
    if temperature != 1.0:
        logits /= temperature
    probabilities = logits.softmax(dim=-1)
    predictions = torch.multinomial(probabilities, num_samples=1).squeeze(dim=-1)
    # get the probabilities associated to the predicted tokens
    probabilities = probabilities.gather(1, predictions.unsqueeze(1)).squeeze(1)
    return predictions, probabilities

def sample_batched_logits(logits, strategy):
    """
    Sample from a batch of logits.
    """
    # B=batch size, L=sequence length, C=num classes
    B, L, C = logits.shape
    samples, probabilities = sample(logits.view(-1, C), **strategy)
    # Reshape the result back to (B, L)
    return samples.view(B, L), probabilities.view(B, L)

def get_error_correction_mask(error_correction, current_sequences, masked_positions, all_masked_positions, mask_token_id, attention_mask_input):
    """
    Get a mask for error correction according to the chosen strategy.
    """
    if error_correction == "prev-unmasked":
        # Get only the positions unmasked in the previous iteration
        return (current_sequences != mask_token_id) & masked_positions
    elif error_correction == "all-masked":
         # Get all masked positions
        return (current_sequences != mask_token_id) & all_masked_positions
    elif error_correction == "all-residues":
        # Get all non-padded residues
        return (current_sequences != mask_token_id) & attention_mask_input.bool()
    else:
        raise NotImplementedError("Error correction strategy not implemented.")

def get_mismatched_tokens(error_correction_mask, current_sequences, logits, labels_to_debug):
    """
    Check if there are any mismatched tokens according to the chosen error correction strategy
    and return the indices of the mismatched tokens and the argmax prediction for all the
    positions involved in error correction.
    """
    argmax_error_correction_mask, probabilities_error_correction = sample(logits[error_correction_mask], top_k=1)
    mismatched_tokens = current_sequences[error_correction_mask] != argmax_error_correction_mask
    if labels_to_debug is not None and torch.any(mismatched_tokens):
        print("Mismatched labels (GT):\t", labels_to_debug[error_correction_mask][mismatched_tokens])
        print("Mismatched tokens (P):\t", current_sequences[error_correction_mask][mismatched_tokens])
        print("Mismatched tokens (new-P):\t", argmax_error_correction_mask[mismatched_tokens])
    return mismatched_tokens, argmax_error_correction_mask, probabilities_error_correction

def sort_by_entropy(logits, batch_indices, seq_indices):
    """
    Sort the masked tokens by entropy in ascending order along the sequence dimension.
    Returns the indices of the batch and sequence dimensions sorted accordingly.
    """
    # Compute the entropy of the logits distribution for each masked token
    probabilities = logits[batch_indices, seq_indices].softmax(dim=-1)
    entropy = -torch.sum(probabilities * probabilities.log(), dim=-1)
    # Sort the masked tokens by entropy in ascending order along the sequence dimension
    _, sorted_indices = entropy.sort()
    return batch_indices[sorted_indices], seq_indices[sorted_indices]

def get_unmasking_mask(batch_size, seq_length, seq_indices, batch_indices, tokens_to_unmask_per_iteration):
    """
    Create a mask for unmasking N tokens in the current sequences where N is `tokens_to_unmask_per_iteration`
    and differs for each element in the batch.
    """
    unmasking_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    for i in range(batch_size):
        # Limit unmasking to the calculated number of tokens for this batch
        mask_indices = seq_indices[batch_indices == i][:tokens_to_unmask_per_iteration[i]]
        unmasking_mask[i, mask_indices] = True
    return unmasking_mask

def unmasking_schedule(N, T):
    """
    Generate the schedule for unmasking tokens accumulating the larger unmasking steps at the beginning.
        N (int): Total number of tokens to unmask.
        T (int): Total number of timesteps.
    Returns:
        list[int]: A list of length T where each entry is the number of tokens
                   to unmask at each timestep, balanced across iterations.
    """
    # Distribute steps accumulating larger steps at the beginning
    schedule = [(N // T + 1) if i < (N % T) else (N // T)
                for i in range(T)]
    assert sum(schedule) == N
    return schedule

def denoise(model,
           tokenizer,
           input_ids,
           attention_mask_input,
           context_ids,
           attention_mask_context,
           strategy={"top_k": 1, "top_p": 0.0, "min_p": 0.0, "temperature": 1.0,
                     "error_correction": None,"entropy_unmasking": False, "start_error_correction": 0},
           use_random_padding=False,
           iterations=1,
           save_intermediate_steps=False,
           labels_to_debug=None):
    """
    Denoises a batch of sequences by unmasking tokens per iteration in parallel.
    Args:
        model: The ESM model trained for MLM.
        tokenizer: The tokenizer used for encoding the input.
        input_ids: A tensor of token IDs with some entries masked (shape: [batch_size, seq_length]).
        attention_mask_input: Attention mask for the input sequences.
        context_ids: Context token IDs.
        attention_mask_context: Attention mask for the context sequences.
        strategy: Dictionary containing sampling strategy parameters.
        use_random_padding: Whether to use random padding in the logits distribution (uses <unk> tokens). This
                            is useful to let the model choose the size of generated sequences (by adding <unk> when needed). 
        iterations: Number of denoising iterations.
        labels_to_debug: Labels for debugging purposes.
        
    Returns:
        A tensor of token IDs with masked tokens denoised (shape: [batch_size, seq_length]).
    """
    batch_size, seq_length = input_ids.shape
    error_correction = strategy["error_correction"]
    start_error_correction = strategy["start_error_correction"]
    entropy_unmasking = strategy["entropy_unmasking"]
    mask_token_id = tokenizer.mask_token_id

    current_sequences = input_ids.clone()
    all_probabilities = torch.zeros_like(current_sequences, dtype=torch.float32)
    masked_positions = current_sequences == mask_token_id  # Shape: [batch_size, seq_length]
    num_masked_tokens = masked_positions.sum(dim=1)  # Shape: [batch_size]
    # tokens_to_unmask_per_iteration = torch.clamp(torch.ceil(num_masked_tokens / iterations).int(), min=1)  # Shape: [batch_size]
    tokens_to_unmask_per_iteration = torch.tensor([unmasking_schedule(n, iterations) for n in num_masked_tokens]) # Shape: [batch_size, iterations]
    
    if labels_to_debug is not None:
        print("Number of masked tokens:\t", num_masked_tokens)
        print("Number of unmasked tokens per iteration:\t", tokens_to_unmask_per_iteration.mean(dim=1))
    
    if save_intermediate_steps:
        intermediate_steps = [current_sequences.clone()]
    
    previous_hidden_states = None
    for iteration in tqdm(range(iterations), desc="Denoising iterations"):
        with torch.no_grad():
            if previous_hidden_states is not None:
                # Reuse the hidden states from the previous iteration (this means that the context is fixed)
                context_ids = None
            # Get logits for the masked tokens
            outputs, previous_hidden_states = model(input_ids=current_sequences,
                        attention_mask_input=attention_mask_input,
                        context_ids=context_ids,
                        attention_mask_context=attention_mask_context,
                        previous_hidden_states=previous_hidden_states,
                        return_context_hidden_states=True)
            logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
            # remove logits associated to non-aminoacid tokens (CLS, EOS, PAD, UNK, X, B, U, Z, O, ., -, NULL_1, MASK)
            if use_random_padding:
                logits[:,:,:3] = float("-Inf") # this keeps <unk> in the logits distribution
            else:
                logits[:,:,:4] = float("-Inf") # set to -inf the logits for the tokens <unk>, <cls>, <eos>, <pad>
            logits[:,:,24:] = float("-Inf")            
        
        # Get the predictions according to the sampling strategy
        predicted_tokens, probabilities_predictions = sample_batched_logits(logits, strategy)  # Shape: [batch_size, seq_length]        
        
        # Error correction
        if error_correction and iteration > start_error_correction:
            all_masked_positions = input_ids == mask_token_id
            error_correction_mask = get_error_correction_mask(error_correction, current_sequences, masked_positions, all_masked_positions, mask_token_id, attention_mask_input)
            # Check if there are any mismatched tokens according to the chosen error correction strategy
            if torch.any(error_correction_mask):
                mismatched_tokens, argmax_error_correction_mask, probabilities_error_correction = get_mismatched_tokens(error_correction_mask, current_sequences, logits, labels_to_debug)

        # Find the masked positions in the current sequences (different from the input_ids because the current_sequences are updated)
        masked_positions = current_sequences == mask_token_id
        # Get the indices of all masked tokens in the batch
        batch_indices, seq_indices = masked_positions.nonzero(as_tuple=True)
        
        # Use the entropy of the logits distribution to unmask tokens with the lowest entropy first
        if entropy_unmasking:
            batch_indices, seq_indices = sort_by_entropy(logits, batch_indices, seq_indices)
        else:
            # Randomize the order of the masked tokens
            indices = torch.randperm(batch_indices.size(0))
            batch_indices, seq_indices = batch_indices[indices], seq_indices[indices]
            

        # Determine which tokens to unmask this iteration
        unmasking_mask = get_unmasking_mask(batch_size, seq_length, seq_indices, batch_indices, tokens_to_unmask_per_iteration[:, iteration])
        # Replace the masked tokens selected for unmasking with the predicted tokens
        current_sequences[unmasking_mask] = predicted_tokens[unmasking_mask]
        all_probabilities[unmasking_mask] = probabilities_predictions[unmasking_mask]
    
        if error_correction and iteration > start_error_correction and torch.any(error_correction_mask):
            # Fix the mismatched tokens that have a different argmax prediction now by replacing them with the new prediction
            indices = torch.nonzero(error_correction_mask, as_tuple=False)[mismatched_tokens]
            current_sequences[indices[:, 0], indices[:, 1]] = argmax_error_correction_mask[mismatched_tokens]
            all_probabilities[indices[:, 0], indices[:, 1]] = probabilities_error_correction[mismatched_tokens]
            
        # Print all the unmasked tokens for debugging
        if labels_to_debug is not None:
            print("Masked tokens (GT):\t", labels_to_debug[masked_positions])
            print("Unmasked tokens (GT):\t", labels_to_debug[unmasking_mask])
            print("Unmasked tokens (P):\t", current_sequences[unmasking_mask])
            
        if save_intermediate_steps:
            intermediate_steps.append(current_sequences.clone())
            
        # Stop early if all tokens are unmasked
        if not masked_positions.any():
            if labels_to_debug is not None:
                print("Completed denoising in ", iteration + 1, " iterations.")
            break
    # Check if there are still masked tokens
    assert not (current_sequences == mask_token_id).any()
    
    # Get the per-sequence perplexity of the denoised sequences from the probabilities
    masked_positions = all_probabilities > 0 # all the positions that were replaced by the predicted tokens
    all_probabilities[~masked_positions] = 1 # set the probabilities of the non-masked tokens to 1 so they don't affect the perplexity
    perplexity = torch.exp(-all_probabilities.log().mean(dim=-1))
    
    if save_intermediate_steps:
        # Shape intermediate_steps: [iterations + 1, batch_size, seq_length], Shape perplexity: [batch_size]
        return torch.stack(intermediate_steps), perplexity
     # Shape output ids: [2, batch_size, seq_length], Shape perplexity: [batch_size]
    return torch.stack([input_ids, current_sequences]), perplexity

def score_sequences(model,
           esm_model,
           tokenizer,
           input_ids,
           context_ids_list):
    """
    Scores a sequence by masking each token separately.
    Args:
        model: The ESM model trained for MLM.
        tokenizer: The tokenizer used for encoding the input.
        input_ids: A tensor of token IDs with no masked entry (shape: [1, seq_length]).
        context_ids: List of context sequences used to score the input sequence (shape: List([1, seq_length])).
        
    Returns:
        A tensor of token IDs with masked tokens denoised (shape: [batch_size, seq_length]).
    """
    _, seq_length = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    num_context_sequences = len(context_ids_list)
    
    # make a tensor with the same sequence repeated seq_length times-2 (exclude bos and eos tokens) and for each one mask a different token
    # example input: [bos, 1, 2, 3, 4, eos] 
    # example output: [[bos, mask, 2, 3, 4, eos], [bos, 1, mask, 3, 4, eos], [bos, 1, 2, mask, 4, eos], [bos, 1, 2, 3, mask, eos]]
    masked_sequences = torch.repeat_interleave(input_ids, seq_length-2 , dim=0)
    labels = masked_sequences.clone()
    mask_diagonal = torch.eye(seq_length-2, dtype=torch.bool).to(masked_sequences.device)
    # masked_sequences has shape [seq_length-2, seq_length]
    masked_sequences[:, 1:-1] = masked_sequences[:, 1:-1].masked_fill(mask_diagonal, mask_token_id)
    # set to -100 all the labels that are not masked
    labels[:, 1:-1] = labels[:, 1:-1].masked_fill(~mask_diagonal, -100)
    labels[:, :1] = -100
    labels[:, -1:] = -100
    # attention mask for the masked sequences (it is the same for all the sequences)
    attention_mask_input = torch.ones_like(masked_sequences)
    
    losses = []
    if model is not None:
        for iter in range(num_context_sequences):
            with torch.no_grad():
                B = 128
                # if the length of the sequence is > B, split it in batches of B to avoid memory issues
                tmp_losses = []
                for i in range(0, seq_length-2, B):
                    # get the batch of masked sequences, attention mask and labels
                    masked_sequences_batch = masked_sequences[i:i+B,:]
                    attention_mask_input_batch = attention_mask_input[i:i+B,:]
                    labels_batch = labels[i:i+B,:]
                    # get the context ids and attention mask for the batch
                    context_ids = context_ids_list[iter]
                    if context_ids is not None:
                        context_ids = torch.repeat_interleave(context_ids, masked_sequences_batch.size(0), dim=0)
                        attention_mask_context = torch.ones_like(context_ids)
                    else:
                        context_ids = None
                        attention_mask_context = None
                    # Get logits and loss for the masked tokens
                    outputs = model(input_ids=masked_sequences_batch,
                                attention_mask_input=attention_mask_input_batch,
                                context_ids=context_ids,
                                attention_mask_context=attention_mask_context,
                                labels=labels_batch)
                    tmp_losses.append(outputs.loss.item()*len(masked_sequences_batch))
                losses.append(sum(tmp_losses)/len(masked_sequences))
            
    # Score the sequences with the ESM model
    esm_loss = None
    if esm_model is not None:
        torch.cuda.empty_cache()
        with torch.no_grad():
            tmp_losses = []
            B = 128
            for i in range(0, seq_length-2, B):
                # get the batch of masked sequences, attention mask and labels
                masked_sequences_batch = masked_sequences[i:i+B,:]
                attention_mask_input_batch = attention_mask_input[i:i+B,:]
                labels_batch = labels[i:i+B,:]
                # Get logits and loss for the masked tokens
                esm_outputs = esm_model(input_ids=masked_sequences_batch,
                                        attention_mask=attention_mask_input_batch,
                                        labels=labels_batch)
                tmp_losses.append(esm_outputs.loss.item()*len(masked_sequences_batch))
            esm_loss = sum(tmp_losses)/len(masked_sequences)
    return losses, esm_loss
            
            
    
    