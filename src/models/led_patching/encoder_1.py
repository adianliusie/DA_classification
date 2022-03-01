from transformers.models.led.modeling_led import *
from transformers.models.led.modeling_led import _expand_mask

# I have edited the model (quite uglily) to include specific positional encoding at the decoder
# All comments from removed from original source code
# https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/led/modeling_led.py

# All lines added have the tag:    'line/block/method added al826'
#^the ordering of above is in the order of execution, not line number

class LEDEncoderUttEncode(LEDEncoder):
    """ patched LED encoder that uses utterance positional encoding at the encoder"""
    
    #>>> Methods added al826 #############################################################
    def set_up_old(self, embeddings, sep_token=None):
        self.utt_positions = embeddings
        self.sep_token = sep_token if sep_token else self.config.eos_token_id
    
    def set_up(self, embeddings, sep_token=None):
        self.utt_positions = LEDLearnedPositionalEmbedding(
            self.config.max_decoder_position_embeddings,
            self.config.d_model,
        )
        self.sep_token = sep_token if sep_token else self.config.eos_token_id
    
    def make_utterance_embeddings(self, input_ids): 
        utt_pos = (input_ids==self.sep_token) #marks utt start positions 
        utt_nums = torch.cumsum(utt_pos, -1) #[bsz, L], marks utt num of each tok
        utt_embeddings = super(LEDLearnedPositionalEmbedding, self.utt_positions).forward(utt_nums)
        return utt_embeddings
    #<<< End of added Methods #############################################################

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        #>>> line added al826 #############################################################
        assert hasattr(self, 'utt_positions'), "need to save decoder utterance positional encoding" 
        #<<M end of added line ############################################################

        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # check input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # create default attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.size()[:-1], device=inputs_embeds.device, dtype=torch.long)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        # pad input if necessary
        padding_len, input_ids, attention_mask, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # retrieve input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        # convert attention_mask to float
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, seq_len]; 1 -> 0.0; 0 -> "-inf"
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)[:, 0, 0, :]

        # get masking tensors
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        embed_pos = self.embed_positions(input_shape)
        
        #>>> block added al826 #############################################################
        utt_pos = self.make_utterance_embeddings(input_ids) 
        hidden_states = inputs_embeds + embed_pos + utt_pos
        #<<< end of added block ############################################################

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_global_attentions = () if (output_attentions and is_global_attn) else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)

            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, is_global_attn, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        head_mask[idx] if head_mask is not None else None,
                        is_index_masked,
                        is_index_global_attn,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        is_index_masked=is_index_masked,
                        is_index_global_attn=is_index_global_attn,
                        is_global_attn=is_global_attn,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)

                if is_global_attn:
                    # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + (layer_outputs[2].transpose(2, 3),)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # undo padding
        if padding_len > 0:
            # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
            hidden_states = hidden_states[:, :-padding_len]

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions, all_global_attentions] if v is not None
            )
        return LEDEncoderBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            global_attentions=all_global_attentions,
        )