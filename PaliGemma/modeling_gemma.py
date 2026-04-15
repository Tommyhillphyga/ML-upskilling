import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math 
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


#create the kvcache class inform of a buffer that store the keys and 
# values of the attention mechanism for each layer.

class KVCache():
     def __init__(self) -> None:
          self.key_cache = List[torch.Tensor] = []
          self.value_cache = List[torch.Tensor] = []

     def num_items(self) -> int:
        if len(self.key_cache) ==0:
          return 0
        else:
             return self.key_cache[0].shape[-2] # the sequence length of the keys is the number of items in the cache
        
     def update(
               self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer_idx: int, ) -> Tuple[torch.Tensor, torch.Tensor]:
          
          if len(self.key_cache) <= layer_idx:
               # if we never added anything to the KV-Cache of this layer, we just create it
               self.key_cache.append(key_states)
               self.value_cache.append(value_states)

          else:
               self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)     
               self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)   


          return self.key_cache[layer_idx], self.value_cache[layer_idx]
     
# repeak kv repeates the number of missing heads for the query and values
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
     if n_rep == 1:
          return hidden_states
     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class GemmaConfig():
     
     def __init__(
               
               self,
               vocab_size,
               hidden_size,
               intermediate_size,
               num_hidden_layers,
               num_attention_heads,
               num_key_value_heads,
               head_dim = 256,
               max_position_embeddings = 8192,
               rms_norm_eps = 2e-6,
               rope_theta = 10000.0, # used to the rotary positional encoding which is also the base frequency of the RoPE.
               attention_bias = False,
               attention_dropout = 0.0,
               pad_token_ids = None,
               **kwargs,
        ):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.head_dim = head_dim
            self.max_position_embeddings = max_position_embeddings
            self.rms_norm_eps = rms_norm_eps
            self.rope_theta = rope_theta
            self.attention_bias = attention_bias
            self.attention_dropout = attention_dropout
            self.pad_token_ids = pad_token_ids

    
class PaliGemmaConfig():

    def __init__(
            self,
            vision_config = None,
            text_config = None,
            ignore_index = -100,
            image_token_index = 256000,
            vocab_size = 257152, 
            projection_dim = 2048,
            hidden_size = 2048,
            pad_token_id = None,
            **kwargs,
        ):
            super().__init__()
            self.ignore_index = ignore_index
            self.image_token_index = image_token_index
            self.vocab_size = vocab_size
            self.projection_dim = projection_dim
            self.hidden_size = hidden_size
            self.vision_config = vision_config
            self.is_encoder_decoder =  False
            self.pad_token_id = pad_token_id

            self.vision_config = SiglipVisionConfig(**vision_config)
            self.text_config = text_config

            self.text_config = GemmaConfig(**text_config, pad_token=pad_token_id)
            self.vocab_size = self.text_config.vocab_size

            self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size)**2
            self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):
     
    def __init__(self, dim: int, eps: float = 1e-6):
          
          super().__init__()
          self.eps = eps
          self.weight = nn.Parameter(torch.zeros(dim))

    def _norm (self, x):
         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward (self, x):
         
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whlist Gemma is (x * w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    

class GemmaMLP(nn.Module):
     
     def __init__(self, config):
          super().__init__()
          self.config = config
          self.hidden_size = config.hidden_size
          self.intermediate_size = config.intermediate_size
          self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
          self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
          self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias = False)

     def forward(self, x):
          y = self.gate_proj(x)
          y = torch.gelu(y, approximate='tanh')
          j = self.up_proj(x)
          z = y * j
          z = self.down_proj(z)
          return z
     

class GemmaAttention(nn.Module):
     def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
          super().__init__()
          self.config = config
          self.layer_idx = layer_idx

          self.attention_dropout = config.attention_dropout
          self.hidden_size = config.hidden_size
          self.num_heads = config.num_attention_heads
          self.head_dim = config.head_dim
          self.num_key_value_heads = config.num_key_value_heads
          self.num_key_value_group = config.num_heads // self.num_key_value_heads
          self.max_position_embeddings = config.max_position_embeddings
          self.rope_theta = config.rope_theta
          self.is_causal = True

          assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_attention_heads"

          # This is using the qrouped-query attention where we have a different 
          # number of heads for the keys and values and for the queries. 
          # The keys and values are grouped together and each group is shared across multiple query heads. 
          # This is the same as the grouped query attention used in Llama 2 and in Gemma. 
          # It allows to reduce the memory usage of the attention mechanism while keeping the same number of query
          #  heads which is important for the performance of the model.

          self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias)
          self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
          self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
          self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias = config.attention_bias)

          self.rotary_embs = GemmaRotaryEmbedding(
               self.head_dim,
               max_position_embeddings = self.max_position_embeddings,
               base = self.rope_theta,
          )

     def forward(
               self, 
               hidden_states: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               position_ids: Optional[torch.LongTensor] = None,
               kv_cache: Optional[KVCache] = None,
               **kwargs, ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
          batch_size, seq_len, _ = hidden_states.size() # batch_size, seq_len, hidden_size
          # [batch_size, seq_len, Num_heads_KV, * Head_Dim]
          query_states = self.q_proj(hidden_states)
          key_states = self.k_proj(hidden_states)
          value_states = self.v_proj(hidden_states)
          query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
          key_states = key_states.view(batch_size, self.num_key_value_heads, self.head_dim).transpose(1,2)
          value_states = value_states.view (batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

          # apply rotary positional embeddings to the query and key states
          # the rotary positional embeddings are applied in place to save memory
          # # [batch_size, seq_len, Head_dim], [batch_size, seq_len, head_dim]
          # 
          cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
          # [batch_size, Num_heads_q, seq_len, Head_dim] -> [batch_size, Num_heads_kV, seq_len, head_dim]

          query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

          if kv_cache is not None:
               key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)


          # Repeat the key and values to match the number of heads of the query
          key_states = repeat_kv(key_states, self.num_key_value_group)
          value_states = repeat_kv(value_states, self.num_key_value_groups)

          #Perform the calculation of the attention scores
          attn_weights = torch.matmul(query_states, key_states.transpoose(2, 3)) / math.sqrt(self.head_dim)

          assert attention_mask is not None
          attn_weights = attn_weights + attention_mask

          #Apply the softmax
          # [batch_size, num_heads_q, seq_len, seq_len_KV]
          attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
          # Apply attention dropout
          attn_weights = nn.functional.dropout(attn_weights, p= self.attention_dropout, training=self.training)

            # Compute the attention output
          attn_output = torch.matmul(attn_weights, value_states)

          if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
               raise ValueError(
                    f"attn_output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                     f" of size {attn_output.size()}"
               )
          
          # make sure the sequence length is the second dimension
          attn_output = attn_output.transpose(1, 2).contiguous()
          attn_output = attn_output.view(batch_size, seq_len, -1)

          # multiply by wo_proj to get the final output of the attention mechanism
          # [batch_size, seq_len, hidden_size]
          attn_output = self.o_proj(attn_output) 

          return attn_output, attn_weights


class GemmaRotaryEmbedding(nn.Module):
     def __init__(self, dim, max_position_embeddings = 2048, base = 10000, device = None):
          super().__init__()

          self.dim = dim
          self.max_position_embeddings = max_position_embeddings
          self.base = base

          # calculate the theta according to tht formular theta_i = base^(2i/dim) where i is the index of the head dimension[0 , 1,2,...dim//2]

          inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
          self.register_buffer("inv_freq", tensor= inv_freq, persistent=False)


     @torch.no_grad()
     def forward(self, x, position_ids, seq_len=None):
          #x: [batch_size, num_attention_heads, seq_len, head_size]
          self.inv_freq.to(x.device)
          # copy the inv_freq tensor for batch in the sequence
          # inv_freq_expanded: [batch_size, head_dim // 2, 1]
          inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
          # position_ids_expanded: [Batch_size, 1, seq_len]
          position_ids_expanded = position_ids[:, None, :].float()
          device_type = x.device.type
          device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

          with torch.autocast(device_type=device_type, enabled=False):   # Autocast for mixed precision 
               # multiply each theta by the position (which is the argment of the sin and cos function)
               # freqs: [batch_size, head_dim, // 2. 1] @ [batch_size, 1, seq_len] --> [batch_size, seq_len, head_dim // 2]

               freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
               # emb: [batch_size, seq_len, head_dim] 
               emb = torch.cat((freqs, freqs), dim=-1)

               #cos, sin: [batch_size, seq_len, head_dim]
               cos = emb.cos()
               sin = emb.sin()

          return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
     

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1): 
     cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
     sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension

     # Apply the formula (34) of the rotary positional encoding paper
     q_embed = (q * cos) * (rotate_half(q) * sin)
     k_embed = (k * cos) * (rotate_half(k) * sin)

     return q_embed, k_embed 

def rotate_half(x):
     # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding

     x1 = x[..., : x.shape[-1] // 2]    # Takes the first half of the last dimension
     x2 = x[..., x.shape[-1] // 2 :]    # TAkes the second half of the last dimension

     return torch.cat((-x2, x1), dim=-1) # Concatenate the two halves in the right order to get the rotated tensor






     


class GemmaDecoderLayer(nn.Module):
     def __init__(self, config: GemmaConfig, layer_idx: int):
          super().__init__()
          self.hidden_size = config.hidden_size
          self.self_attn = GemmaAttention(config = config, layer_idx = layer_idx)

          self.mlp = GemmaMLP(config)
          self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps= config.rms_norm_eps)
          self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps= config.rms_norm_eps)

     def forward(
            self,
            hidden_states:torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None, 
            position_ids: Optional[torch.LongTensor] = None, 
            kv_cache: Optional[KVCache] = None) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
     

             # self attention 
            residual = hidden_states
            # [batch_size, seq_len, hidden_size]
            hidden_states = self.input_layernorm(hidden_states)
            # [batch_size, seq_len, hidden_size]
            hidden_states, attn_weights = self.self_attn(
                hidden_states = hidden_states ,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )

            # [batch_size, seq_len, hidden_size]
            hidden_states = residual + hidden_states
            # [batch_size, seq_len, hidden_size
            residual = hidden_states
            # batch_size, seq_len, hidden_size
            hidden_states = self.post_attention_layernorm(hidden_states)
            # [batch_size, seq_len, hidden_size]
            hidden_states = self.mlp(hidden_states)
            # [batch_size, seq_len, hidden_size]
            hidden_states = residual + hidden_states
            
            return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
          super().__init__()
          self.config = config
          self.padding_idx = config.pad_token_ids
          self.vocab_size = config.vocab_size

          self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
          self.layers = nn.ModuleList(
               [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
          )
          self.norm = GemmaRMSNorm(config.hidden_size, eps= config.rms_norm_eps)

    def get_input_embeddings(self):
         return self.embed_tokens
    
    def forward(
              self,
              attention_mask: Optional[torch.Tensor] = None,
              position_ids: Optional[torch.LongTensor] = None,
              inputs_embeds: Optional[torch.FloatTensor] = None,
              kv_cache: Optional[KVCache] = None,) -> torch.FloatTensor:
            
            
            # [batch_size, seq_len, hidden_size]
            hidden_states = inputs_embeds
            # [batch_size, seq_len, hidden_size]
            normalizer = torch.tensor(self.config.hidden_size**0.5, dtype= hidden_states.dtype)
            hidden_states = hidden_states * normalizer

            for decoder_layer in self.layers:
                # [batch_size, seq_len, hidden_size]
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask = attention_mask,
                    position_ids = position_ids,
                    kv_cache = kv_cache)
                
            # [batch_size, seq_len, hidden_size]   
            hidden_states = self.norm(hidden_states)

            return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head == nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
         return self.model.embed_tokens
    
    def tie_weights(self):
         self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
              self,
              attention_mask: Optional[torch.Tensor] = None,
              position_ids: Optional[torch.LongTensor] = None,
              inputs_embeds: Optional[torch.FloatTensor] = None,
              kv_cache: Optional[KVCache] = None,
              
    ) -> Tuple:
         # input_embeds: [batch_size, seq_len, hidden_size]
         # outputs: [batch_size, seq_len, hidden_size]
        outputs = self.model(
              attention_mask = attention_mask,
              position_ids = position_ids,
              inputs_embeds = inputs_embeds,
              kv_cache = kv_cache,
         )

        hidden_states = outputs
        logits = self.im_head(hidden_states)
        logits = logits.float()

        return_data = {
             
            "logits": logits,
              
         }

        if kv_cache is not None:
            #Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data
         


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias = True)

    def forward(self, image_features):
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states
    

class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    

    def _merge_input_ids_with_image_features(
              
            self,
            image_features: torch.Tensor,
            inputs_embeds: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            kv_cache: Optional[KVCache] = None ):

            _,_, embed_dim = image_features.shape
            batch_size, sequence_length = input_ids.shape
            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            # shape [batch_size, seq_len, hidden_size]
            scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

            # combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens
            final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

            # shape [batch_size, seq_len]. True for text tokens
            text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
            # shape [batch_size, seq_len]. True for image tokens
            image_mask = input_ids == self.config.image_token_index
            # shape [batch_size, seq_len]. True for padding tokens
            pad_mask = input_ids == self.pad_token_id

            # we used to expand the masks to the embedding dimension otherwise we can't use them in torch.where
            text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
            image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
            pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

            # add the text embeddings
            final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
            # insert image embeddings. we can't use torch.where because the sequence length of the scaled_image_features is not equals to the sequence lenght of the final embeddings
            # but it does the same job as torch.where
            final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features) # copy the image features/tokens to the final embedding placeholder where the image mask is True
            # zero out padding tokens

            final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            min_type = torch.finfo.min
            q_len = inputs_embeds.shape[1]

            if kv_cache is None or kv_cache.num_items() == 0:
                # do not mask any token because we are in the prefill phase
                # This only works when we have no padding

                causal_mask = torch.full(
                    (batch_size, q_len, q_len), 
                    fill_value=0,
                    dtype=dtype,
                    device=device
                )
            else:
                 #since we are generating tokens, the query must be one single token
                 assert q_len == 1
                 kv_len = kv_cache.num_items() + q_len
                 # Also in this case we don't need to mask anyting, since each query should be able to attend to all previoue tokens
                 # This only woeks when we have no padding

                 causal_mask = torch.full(
                      (batch_size, q_len, kv_len), fill_value=0, dtype= dtype, device=device
                 )

            # Add the head dimension
            #(batch_size, q_len, kv_len) -> [batch_size, num_heads_q, q_len, kv_len]

            causal_mask = causal_mask.unsqueeze(1)

            if kv_cache is not None and kv_cache.num_items() > 0:
                 # The position of the query is nust the last position
                 position_ids = attention_mask.cumsum(-1)[1,-1]
                 if position_ids.dim() == 1:
                      position_ids = position_ids.unsqueeze(0)

            else:
                 # create a position_ids based on the size of the attention mask
                 # for masked tokens, use the number 1 as position

                 position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

            return final_embedding, causal_mask, position_ids
                 
                 
    def forward (
            self, 
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None, 
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Extract the input embeddings
        # shape [Batch_size, seq_len, hidden_size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Merge text and images 
        #[batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        #   [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # merge the embedding of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features,
            input_ids, 
            attention_mask,
            kv_cache,

        )
        
        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache,

        )

        return outputs
    


