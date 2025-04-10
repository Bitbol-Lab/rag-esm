import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.esm.modeling_esm import EsmModel, BaseModelOutputWithPoolingAndCrossAttentions, EsmForMaskedLM, EsmAttention, EsmIntermediate, EsmOutput
from transformers.models.esm.modeling_esm import EsmLayer, BaseModelOutputWithPastAndCrossAttentions, EsmEncoder, EsmSelfAttention, EsmSelfOutput

class NewEsmSelfAttention(EsmSelfAttention):
    """
        CHANGES: Add option to use flash attention (doesn't return attention matrices, in that case set use_flash_attention=False).
                 Add option to use cross attention (set use_cross_attention=True), doesn't apply rotary embeddings to cross-attention.
    """
    def __init__(self, config, position_embedding_type=None, use_cross_attention=True, use_flash_attention=False):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.use_cross_attention = use_cross_attention
        self.use_flash_attention = use_flash_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        if output_attentions and self.use_flash_attention:
            raise NotImplementedError("Flash attention can't output attentions matrices.")
        
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Matt: Our BERT model (which this code was derived from) scales attention logits down by sqrt(head_dim).
        # ESM scales the query down by the same factor instead. Modulo numerical stability these are equivalent,
        # but not when rotary embeddings get involved. Therefore, we scale the query here to match the original
        # ESM code and fix rotary embeddings.
        query_layer = query_layer * self.attention_head_size**-0.5

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        if self.position_embedding_type == "rotary" and (not self.use_cross_attention):
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.use_flash_attention:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()
            # Use scaled_dot_product_attention for absolute or rotary positions
            context_layer = nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=torch.tensor(1.0, dtype=query_layer.dtype) # None if self.use_cross_attention else torch.tensor(1.0, dtype=query_layer.dtype)
            )
            if head_mask is not None:
                context_layer = context_layer * head_mask.unsqueeze(1).unsqueeze(-1)    
        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
                seq_length = hidden_states.size()[1]
                position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
                position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
                distance = position_ids_l - position_ids_r
                positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
                positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

                if self.position_embedding_type == "relative_key":
                    relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores
                elif self.position_embedding_type == "relative_key_query":
                    relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in EsmModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)
            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            # Multiply the attention scores by the value vectors.
            context_layer = torch.matmul(attention_probs.to(value_layer.dtype), value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class NewEsmAttention(EsmAttention):
    """
        CHANGES: Use NewEsmSelfAttention instead of EsmSelfAttention and add a learnable parameter to scale the cross-attention output
    """
    def __init__(self, config, use_cross_attention=True, use_flash_attention=False):
        super().__init__(config)
        self.self = NewEsmSelfAttention(config, use_cross_attention=use_cross_attention, use_flash_attention=use_flash_attention)
        # learnable parameter that scales the cross-attention output (initially set to 0)
        if use_cross_attention:
            self.weight_cross_attention = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.output = NewEsmSelfOutput(config)

class NewEsmLayer(nn.Module):
    """
        CHANGES: Totally replaces `EsmLayer` Use NewEsmAttention instead of EsmAttention for both the self-attention
                 (this to allow the use of flash attention) and the cross-attention layers.
                 Add a feed forward layer and LayerNorm to the cross-attention output.
    """
    def __init__(self, config, this_layer_has_cross_attention: bool = True, use_cross_attention=True, use_flash_attention=False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.attention = NewEsmAttention(config, use_cross_attention=False, use_flash_attention=use_flash_attention)
        self.is_decoder = config.is_decoder
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # cross-attention layer
        self.use_cross_attention = use_cross_attention
        self.this_layer_has_cross_attention = this_layer_has_cross_attention
        if self.use_cross_attention and self.this_layer_has_cross_attention:
            # cross-attention layer
            self.crossattention = NewEsmAttention(config, use_cross_attention=True, use_flash_attention=use_flash_attention)
            # feed forward layer for cross-attention
            self.weight_crossattention_ffw = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.intermediate_crossattention = EsmIntermediate(config)
            self.output_crossattention = EsmOutput(config)
            self.LayerNorm_crossattention = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # delete crossattention layer if it is not needed
        if not self.this_layer_has_cross_attention and hasattr(self, "crossattention"):
            raise AttributeError("There shoudn't be a crossattention layer in this layer")
            delattr(self, "crossattention")
                    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        # Feed forward layer
        layer_output = self.feed_forward_chunk(attention_output)
        
        if self.use_cross_attention and encoder_hidden_states is not None and self.this_layer_has_cross_attention:
            # Cross-attention layer
            cross_attention_outputs = self.crossattention(
                layer_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                None,  # cross_attn_past_key_value
                output_attentions,
            )
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights
            
            # Gating: instead of tanh you can multiply by
            # torch.clamp(self.crossattention.weight_cross_attention, min=-1., max=1.)

            # Gated cross-attention output using tanh activation
            layer_output_cross = layer_output + cross_attention_outputs[0] * torch.tanh(self.crossattention.weight_cross_attention)
            
            # Feed forward layer for cross-attention
            layer_output = layer_output_cross + self.feed_forward_chunk_cross(layer_output_cross) * torch.tanh(self.weight_crossattention_ffw)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        attention_output_ln = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(attention_output_ln)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    def feed_forward_chunk_cross(self, cross_attention_output):
        cross_attention_output_ln = self.LayerNorm_crossattention(cross_attention_output)
        intermediate_output = self.intermediate_crossattention(cross_attention_output_ln)
        layer_output = self.output_crossattention(intermediate_output, cross_attention_output)
        return layer_output


class NewEsmEncoder(EsmEncoder):
    """
        CHANGES: Use NewEsmLayer (modified to include cross attention and flash attention) instead of EsmLayer.
                 Give the correct encoder_hidden_states to the right cross attention layer (numbered).
                 Add a function to select the cross-attention layers based on their gating parameters.
    """
    def __init__(self, config, layers_with_cross_attention="all", use_cross_attention=True,
                 use_flash_attention=False, gate_selection_function=None):
        super().__init__(config)
        self.use_cross_attention = use_cross_attention
        self.gate_selection_function = gate_selection_function
        if layers_with_cross_attention == "all":
            self.layers_with_cross_attention = list(range(config.num_hidden_layers))
        elif layers_with_cross_attention == "last":
            self.layers_with_cross_attention = [config.num_hidden_layers - 1]
        elif layers_with_cross_attention == "none":
            self.layers_with_cross_attention = []
        elif "alternate" in layers_with_cross_attention:
            # get the interval of the layers with cross-attention
            step = int(layers_with_cross_attention.split("-")[1])
            # start from last layer and go backwards
            self.layers_with_cross_attention = list(range(config.num_hidden_layers - 1, -1, -step))
        # check if layers_with_cross_attention is a omegaconf ListConfig
        elif hasattr(layers_with_cross_attention, "__iter__"):
            self.layers_with_cross_attention = layers_with_cross_attention
        else:
            raise ValueError("layers_with_cross_attention should be 'all', 'last', 'none' or a list of integers")
        self.has_cross_attention = [True if i in self.layers_with_cross_attention else False
                                    for i in range(config.num_hidden_layers)]
        print("Where cross-attention is applied:", self.has_cross_attention)
        self.layer = nn.ModuleList([NewEsmLayer(config,
                                                this_layer_has_cross_attention=this_layer_has_cross_attention,
                                                use_cross_attention=use_cross_attention,
                                                use_flash_attention=use_flash_attention)
                                    for _, this_layer_has_cross_attention in zip(range(config.num_hidden_layers), self.has_cross_attention)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False, # otherwise list of boos where True means that the layer should output the hidden states
        return_dict=True,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.use_cross_attention else None

        next_decoder_cache = () if use_cache else None
        
        # if a function to select the layers is provided, get the gating parameters for each layer
        # and select the layers based on their gating parameters
        if self.gate_selection_function is not None:
            # get the gating parameters for each layer
            gating_params = {}
            for i, layer_module in enumerate(self.layer):
                if hasattr(layer_module, "crossattention"):
                    gating_params1 = torch.tanh(layer_module.crossattention.weight_cross_attention).item()
                    gating_params2 = torch.tanh(layer_module.weight_crossattention_ffw).item()
                    gating_params[i] = abs(gating_params1) + abs(gating_params2)
            # select which layers to use
            selected_gates = self.gate_selection_function(gating_params)
            # # get the indices of the layers with the highest 10% gating parameters
            # selected_gates = sorted(gating_params, key=gating_params.get, reverse=True)[:math.ceil(0.1*len(gating_params))]
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:# and i>0:
                # raise NotImplementedError("This shifts layers by 1, models should be retrained")
                # exclude layer 0 because it is the embedding layer
                # if output_hidden_states[i-1]:
                if output_hidden_states[i]:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                else:
                    all_hidden_states = all_hidden_states + (None,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            encoder_hidden_states_layer = encoder_hidden_states[i] if encoder_hidden_states is not None else None
            # if a function to select the layers with the highest gating parameters is provided,
            # set the encoder_hidden_states_layer to None when the layer is not selected by the function
            if self.gate_selection_function is not None:
                encoder_hidden_states_layer = encoder_hidden_states_layer if i in selected_gates else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states_layer,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states_layer,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.has_cross_attention[i] and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.emb_layer_norm_after:
            hidden_states = self.emb_layer_norm_after(hidden_states)
            
        if output_hidden_states:
            if output_hidden_states[-1]:
                all_hidden_states = all_hidden_states + (hidden_states,)
            else:
                all_hidden_states = all_hidden_states + (None,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class NewEsmModel(EsmModel):
    """
        CHANGES: Use NewEsmEncoder instead of EsmEncoder
                 Change how is computed the encoder_batch_size and encoder_sequence_length to take into account that
                 the encoder_hidden_states is a tuple of tensors.
                 Modified the predict_contacts method to take as input pre-computed attention matrices.
    """
    def __init__(self, config, add_pooling_layer=True, layers_with_cross_attention="all", use_cross_attention=True,
                 use_flash_attention=False, gate_selection_function=None):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.use_cross_attention = use_cross_attention
        self.encoder = NewEsmEncoder(config,
                                     layers_with_cross_attention=layers_with_cross_attention,
                                     use_cross_attention=use_cross_attention,
                                     use_flash_attention=use_flash_attention,
                                     gate_selection_function=gate_selection_function)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.use_cross_attention and encoder_hidden_states is not None:
            # encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            encoder_batch_size, encoder_sequence_length, _ = next(h for h in encoder_hidden_states if h is not None).size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        
    def predict_contacts(self, tokens, attns, attention_mask):
        attns = torch.stack(attns, dim=1)  # Matches the original model layout
        # In the original model, attentions for padding tokens are completely zeroed out.
        # This makes no difference most of the time because the other tokens won't attend to them,
        # but it does for the contact prediction task, which takes attentions as input,
        # so we have to mimic that here.
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        return self.contact_head(tokens, attns)

class NewEsmForMaskedLM(EsmForMaskedLM):
    """
        CHANGES: Use NewEsmModel instead of EsmModel, and add the predict_contacts method.
    """
    def __init__(self, config, layers_with_cross_attention="all", use_cross_attention=True,
                 use_flash_attention=False, gate_selection_function=None):
        super().__init__(config)
        self.esm = NewEsmModel(config, add_pooling_layer=False,
                               layers_with_cross_attention=layers_with_cross_attention,
                               use_cross_attention=use_cross_attention,
                               use_flash_attention=use_flash_attention,
                               gate_selection_function=gate_selection_function)
    
    def predict_contacts(self, tokens, attns, attention_mask):
        return self.esm.predict_contacts(tokens, attns, attention_mask)
