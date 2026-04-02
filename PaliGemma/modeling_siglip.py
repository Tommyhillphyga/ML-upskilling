from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size = 768,
            intermediate_size = 3072,
            num_hidden_layer = 12,
            num_attention_heads = 12,
            num_channels = 3,
            image_size = 244,
            patch_size = 16,
            layer_norm_eps = 1e-6,
            attention_dropout = 0.0,
            num_image_tokens: int = None,
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layer = num_hidden_layer
        self.attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels= config.num_channels,
            out_channels= self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding = 'valid',

        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            'position_ids', 
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )


    def forward (self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _,_, height, width = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        #[batch_size, Embed_dim, num_patches_h, num_patches_w] -> [batch_size, embed_dim, num_patches_h, num_patches_w]
        embeddings = patch_embeds.flatten(2)
        # [batch_size, embed_dim, num_patches] -> [batch_size, num_patches, embed_dim]
        embeddings = embeddings.transpose(1, 2)
        #Add positional encoding to each patch. Each positional encording is a vector of size [num_patches, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [batch_size, num_patches, embed_dim]
        return embeddings

class SigilpMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intemediate_size, config.hidden_size)

    def forward (self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # hidden states: [batch_size, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
    





class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward (
            self,
            hidden_states: torch.Tensor,

    ) -> torch.Tensor:
        # residual [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> batch_size, num_patches, embed_dim
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states,_ = self.self_attn(hidden_states=hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)# the input dimension does not change: [batch_size, num_patchs, embed_dim]
        hidden_states = residual + hidden_states

        return hidden_states
    


        












class SiglipVisionTransformer (nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward (self, pixel_values: torch.Tensor) -> torch.Tensor:
        #pixel values: [batches, channel, height, weight] -> [batches, num_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds = hidden_states)
        last_hidden_states = self.post_layernorm(last_hidden_state)
        return last_hidden_state
    


class SiglipVisionModel (nn.Module):

    def __init__(self, config, SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward (self, pixel_values) -> Tuple:
        # [batch, channels, height, weight] -> [batch, num_patchs, enbed_dim]
        return self.vision_model(pixel_values=pixel_values)