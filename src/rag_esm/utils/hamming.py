import torch

def make_tensor_msa(msa):
    int_list = [[ord(c) for c in string if not c.islower()] for string in msa]
    return torch.tensor(int_list, dtype=torch.int)

def recover_matrix(hamming_distances):
    """
    From the flattened upper triangular part of the matrix, recover the full matrix
    """
    N = int((1 + (1 + 8*len(hamming_distances))**0.5)/2)
    boolean_mask = torch.triu(torch.ones((N, N), dtype=torch.bool), diagonal=1)
    hamming_matrix = torch.zeros((N, N), dtype=hamming_distances.dtype)
    hamming_matrix[boolean_mask] = hamming_distances
    hamming_matrix = hamming_matrix + hamming_matrix.t()
    return hamming_matrix

def hamming_gpu(sequences, return_matrix=True, normalize=True, batch=None, device="gpu"):
    """
        Compute the Hamming distance between sequences using the GPU to speed up the computation.
        `sequences` is a list of strings.
        If `return_matrix` is True, return the full distance matrix, otherwise return the flattened upper triangular part.
        If `normalize` is True, return the normalized Hamming distance, otherwise return the number of differences.
    """
    sequences = make_tensor_msa(sequences)
    sequences = sequences.to(device) if device=="cuda" else sequences
    # Get the shape of the tensor
    N, L = sequences.shape
    # Expand dimensions to compare each sequence with every other sequence
    expanded_sequences_1 = sequences.unsqueeze(1)  # Shape: (N, 1, L)
    
    if batch is not None:
        # Split in batches to avoid memory issues
        # Initialize an empty tensor to store the results
        hamming_distances = torch.zeros((N, N), device=sequences.device, dtype=torch.float if normalize else torch.int)
        for i in range(0, N, batch):
            R = min(i+batch, N)
            expanded_sequences_2 = sequences[i:R].unsqueeze(0)  # Shape: (1, batch, L)
            # # Compute element-wise differences and sum along the last dimension
            # hamming_distances[:,i:R] = (expanded_sequences_1 != expanded_sequences_2).sum(dim=2)
            # to speed up the computation do not compute the lower triangular part
            hamming_distances[i:,i:R] = (expanded_sequences_1[i:] != expanded_sequences_2).sum(dim=2)
    else:
        # compare all sequences at once
        expanded_sequences_2 = sequences.unsqueeze(0)  # Shape: (1, N, L)
        # Compute element-wise differences and sum along the last dimension
        hamming_distances = (expanded_sequences_1 != expanded_sequences_2).sum(dim=2)
        
    # normalize by the length of the sequences
    hamming_distances = hamming_distances.float()/L if normalize else hamming_distances
    
    # Return the flattened upper triangular part of the matrix
    boolean_mask = torch.triu(torch.ones_like(hamming_distances.t()), diagonal=1).bool()
    hamming_distances = torch.triu(hamming_distances.t(), diagonal=1)[boolean_mask].cpu()
    # clear gpu memory
    torch.cuda.empty_cache()
    return recover_matrix(hamming_distances) if return_matrix else hamming_distances