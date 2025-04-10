import math
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel, PreTrainedModel
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from rag_esm.modules.esm_decoder import NewEsmForMaskedLM, NewEsmModel

class RAGConfig(PretrainedConfig):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", use_cross_attention=True,
                 freeze_encoder=True, train_only_cross_attention=False, skip_cross_ratio=0,
                 layers_with_cross_attention="all", use_flash_attention=False, 
                 gate_selection_function=None, dropout=0.1, tie_weights_encoder_decoder=False,
                 rescale_loss_diffusion=False, take_average_embeddings=False, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.use_cross_attention = use_cross_attention
        self.use_flash_attention = use_flash_attention
        self.freeze_encoder = freeze_encoder
        self.train_only_cross_attention = train_only_cross_attention
        self.skip_cross_ratio = skip_cross_ratio
        self.gate_selection_function = gate_selection_function
        self.dropout = dropout
        self.tie_weights_encoder_decoder = tie_weights_encoder_decoder
        self.rescale_loss_diffusion = rescale_loss_diffusion
        self.take_average_embeddings = take_average_embeddings
        if use_cross_attention:
            self.layers_with_cross_attention = layers_with_cross_attention
        else:
            self.layers_with_cross_attention = "none"
    
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "use_cross_attention": self.use_cross_attention,
            "freeze_encoder": self.freeze_encoder,
            "train_only_cross_attention": self.train_only_cross_attention,
            "skip_cross_ratio": self.skip_cross_ratio,
            "layers_with_cross_attention": self.layers_with_cross_attention,
            "use_flash_attention": self.use_flash_attention,
            "gate_selection_function": self.gate_selection_function,
            "dropout": self.dropout,
            "tie_weights_encoder_decoder": self.tie_weights_encoder_decoder,
            "rescale_loss_diffusion": self.rescale_loss_diffusion,
            "take_average_embeddings": self.take_average_embeddings
        }
    
def gate_selection(gate_selection_function):
    """
        Returns a function that selects the gates to be used in the model. The input of the
        function is a dictionary with the indices of the layer of each gate and their values,
        the function returns a list with the indices of the gates that should be used.
    """
    if gate_selection_function == "none":
        # Use all the gates
        return None
    elif gate_selection_function == "all":
        # Use all the gates
        return lambda x: list(range(len(x)))
    elif "random-" in gate_selection_function:
        # the parameter is of the form "random-N" where N is the percentage of gates to be selected
        frac = 1 - int(gate_selection_function.split("-")[1])/100
        # Select a random fraction of the gates (1-frac) to be used
        return lambda x: torch.randperm(len(x))[math.ceil(frac*len(x)):].tolist()
    elif "top-" in gate_selection_function:
        # the parameter is of the form "top-N"
        N = int(gate_selection_function.split("-")[1])
        # Select the top N gates with the highest values
        return lambda x: sorted(x, key=x.get, reverse=True)[:N]

# Wrapper for the RAG model
class RAGModel(PreTrainedModel):
    
    def __init__(self, config):
        super(RAGModel, self).__init__(config)
        self.config = config
        self.gate_selection_function = gate_selection(config.gate_selection_function) if config.gate_selection_function is not None else None
        if config.use_cross_attention:
            self.encoder = NewEsmModel.from_pretrained(config.model_name,
                                                       attention_probs_dropout_prob = self.config.dropout,
                                                       hidden_dropout_prob = self.config.dropout,
                                                       use_cross_attention=False,
                                                       layers_with_cross_attention="none",
                                                       use_flash_attention=config.use_flash_attention)
            if config.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
        else:
            self.encoder = None
        self.decoder = NewEsmForMaskedLM.from_pretrained(config.model_name,
                                                         attention_probs_dropout_prob = self.config.dropout,
                                                         hidden_dropout_prob = self.config.dropout,
                                                         use_cross_attention=config.use_cross_attention,
                                                         layers_with_cross_attention=config.layers_with_cross_attention,
                                                         use_flash_attention=config.use_flash_attention,
                                                         gate_selection_function=self.gate_selection_function)
        # self.hamming_dist = torch.Tensor([1])
        # self.levenshtein_dist = torch.Tensor([1])
        self.rescale_loss_diffusion = config.rescale_loss_diffusion
        self.tie_weights_encoder_decoder = config.tie_weights_encoder_decoder
        self.take_average_embeddings = config.take_average_embeddings
        if self.tie_weights_encoder_decoder and (not config.train_only_cross_attention) and (not config.freeze_encoder):
            self.tie_weights_enc_dec()
            self.test_tied_weights()

    def tie_weights_enc_dec(self):
        """
        Tie the all the weights shared between the encoder and the decoder. All the weights of the encoder are
        called in the same way in the decoder module.
        """
        print(f"Tying weights between encoder and decoder")
        for enc_name, enc_param in self.encoder.named_parameters():
            for dec_name, dec_param in self.decoder.esm.named_parameters():
                if enc_name == dec_name:
                    # Access the parameter by its hierarchical name
                    dec_module = self.decoder.esm
                    enc_module = self.encoder
                    # Split the parameter path to locate the attribute
                    for part in dec_name.split(".")[:-1]:
                        dec_module = getattr(dec_module, part)
                    for part in enc_name.split(".")[:-1]:
                        enc_module = getattr(enc_module, part)
                    # Tie the parameters explicitly (we need to tie encoder to decoder and not the opposite
                    # because the decoder embeddings are also tied to the lm_head module, in this way we are
                    # able to tie the encoder embeddings to both the decoder embeddings and the lm_head module)
                    setattr(enc_module, enc_name.split(".")[-1], getattr(dec_module, dec_name.split(".")[-1]))
    
    def test_tied_weights(self, only_numerical=False):
        error = False
        for enc_name, enc_param in self.encoder.named_parameters():
            for dec_name, dec_param in self.decoder.esm.named_parameters():
                if enc_name == dec_name:
                    try:
                        if not only_numerical:
                            # check that the weights are tied
                            assert dec_param.data_ptr() == enc_param.data_ptr()
                        # check that the weights are the same
                        assert torch.allclose(enc_param, dec_param)
                    except:
                        difference = torch.sum(torch.abs(enc_param - dec_param)).item()
                        print(f"Weights: {enc_name} and {dec_name} are not tied, absolute difference: {difference}")
                        error = True
        if error:
            raise ValueError("Some weights are not tied")
        
    def forward(self,
                input_ids,
                attention_mask_input,
                context_ids,
                attention_mask_context,
                previous_hidden_states=None,
                return_context_hidden_states=False,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                hamming_dist=None,
                levenshtein_dist=None,
                return_dict=True,
                **kwargs):
        # During training, ignore_cross is set to True a fraction (`skip_cross_ratio`) of the times to avoid losing
        # the ability to handle no-context cases. Otherwise, it is set to False. During evaluation, ignore_cross is always set to False.
        ignore_cross = torch.rand(1).item() <= self.config.skip_cross_ratio if self.training else False
        # Set hamming and levenshtein distances so that they can be logged during training
        if hamming_dist is not None and levenshtein_dist is not None:
            # if ignore_cross is True, set the distances to 1 (the model is not using the context)
            self.hamming_dist = hamming_dist if not ignore_cross else torch.ones_like(hamming_dist)
            self.levenshtein_dist = levenshtein_dist if not ignore_cross else torch.ones_like(levenshtein_dist)
            
        # If the model is provided precomputed hidden states, use them instead of encoding the context sequences
        if previous_hidden_states is not None and return_context_hidden_states:
            encoder_hidden_states = previous_hidden_states
            encoder_attention_mask = attention_mask_context
        elif (not ignore_cross) and self.config.use_cross_attention and context_ids is not None:
            # Encode the context sequences if the model is not ignoring the context
            with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
                outputs_encoder = self.encoder(
                    input_ids=context_ids,
                    attention_mask=attention_mask_context,
                    output_hidden_states=self.decoder.esm.encoder.has_cross_attention
                )
                encoder_hidden_states = outputs_encoder.hidden_states
                encoder_attention_mask = attention_mask_context
                if self.take_average_embeddings:
                    encoder_attention_mask = None
                    encoder_hidden_states = tuple(torch.mean(el, dim=1)[:, None, :] for el in encoder_hidden_states )
                
        else:
            # If the model is ignoring the context, set the encoder hidden states to None
            encoder_hidden_states = None
            encoder_attention_mask = None
            
        # decode the input sequences using the encoded context sequences
        if output_hidden_states == "last":
            # get a list with False on each layer except the last one
            output_hidden_states = [False] * (self.decoder.esm.config.num_hidden_layers - 1) + [True]
        outputs = self.decoder(input_ids=input_ids,
                               attention_mask=attention_mask_input,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask,
                               labels=labels,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)
        
        if labels is not None:
            self.masking_fraction = torch.sum(labels != -100).item() / torch.sum(input_ids != 1).item()
            # Rescale the loss by the masking fraction to avoid the loss being dominated by the masked tokens
            if self.rescale_loss_diffusion:
                outputs.loss = outputs.loss / self.masking_fraction
            
        if return_context_hidden_states:
            return outputs, encoder_hidden_states
        return outputs
    
    def predict_contacts(self, tokens, attns, attention_mask):
        """
        Predict contacts from the attention scores.
        """
        return self.decoder.predict_contacts(tokens, attns, attention_mask)
