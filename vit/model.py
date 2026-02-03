# Vision Transformer (ViT) 

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)


class Vit(nn.Module):
    def __init__(
            self,
            chw: tuple = (1, 28, 28), 
            n_patches: int = 7,
            n_heads: int = 2,
            n_blocks: int = 2,
            hidden_d: int = 8,
            out_d: int = 10


    ):
        super(Vit, self).__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        self.n_blocks = n_blocks
        self.n_heads = n_heads

        assert chw[1] % n_patches == 0, "input shape should be divisible by n_patches"
        assert chw[2] % n_patches == 0, "input shape should be divisible by n_patches"

        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches) # 4 * 4

        # add linear mapper 
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1]) # 1 * 4 * 4 = 16
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # add learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        # add positional embedding
       #self.pos_embedding = nn.Parameter(torch.tensor(self.n_patches ** 2 + 1, self.hidden_d))
       #self.pos_embedding.requires_grad = False
        self.register_buffer(
            "positional_embeddings",
            get_positional_embedding(n_patches**2+1, hidden_d),
            persistent=False,
        )

        # Encoder block
        self.block = nn.ModuleList(
             EncoderBlock([self.hidden_d, self.n_heads] for _ in range (n_blocks))
         )
        
        #Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )
        
    def forward (self, images):
        #divide images into patches
        n, c, h, w = images
        patches = patch_embedding(images, self.n_patches).to(self.positional_embeddings.device) # n, 49, 16

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches) # n, 49, 8

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        #tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])  # n, 50, 8

        pos_embed = self.positional_embeddings.repeat(n, 1, 1) # n, 50, 8
        out = tokens + pos_embed 

        #transformer block
        for block in self.block:
            out = block(out)

        # Extract classification token
        out = out[:, 0]

        return self.mlp(out) #n, 50 , 8

        

def patch_embedding(images, n_patches):
    """
    Splits the images into patches and embeds them.

    Args:
        images: Tensor of shape (B, C, H, W)
        n_patches: Number of patches to split the image into (assumes square number)
    """
    n, c, h, w = images.shape  # b, 1, 28, 28
    assert h == w, "Image height and width must be the same"
    patches = np.zeros(n, n_patches **2, c * h * w // (n_patches **2))  # b, 49, 16
    patch_size = h // n_patches  # 4

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    
    return patches  # n, 49, 16


def get_positional_embedding(sequence_lenght, d): # 50, 8
    result = torch.ones(sequence_lenght, d)
    for i in range (sequence_lenght):
        for j in range (d):
            result[i][j] = np.sin(i / (10000 ** (j/d))) if j % 2 == 0 else np.cos(i/ (10000 ** ((j-1)/d)))
    return result # 50, 8


class MultiHeadAttention (nn.module):
    def __init__(
            self,
            d,
            n_heads=2):
        super(MultiHeadAttention, self).__init__()

        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"can't divide dmension {d} into : {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.d_head= d_head
        self.softmax = nn.softmax(dim=-1)

    def forward (self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range (self.n_heads): # 0, 1
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                
                seq = sequence[:, head * self.d_head: (head+1) * self.d_head ] # take all rows, then begin from col 0:4, then col 4: 8. the shape: 50, 4
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)

            result.append(torch.hstack(seq_result))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result]) # n, 50, 8
            

# Encoder Block

class EncoderBlock (nn.Module):
    def __init__(
            self,
            hidden_d,
            n_heads,
            mlp_ratio = 4,
    ):
        super(EncoderBlock, self).__init__()

        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
    

        self.mhsa = MultiHeadAttention(hidden_d, n_heads)
        self.norm1 = nn.LayerNorm(hidden_d)
        self.norm2 = nn.LayerNorm(hidden_d)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, hidden_d * mlp_ratio),
            nn.GELU(), 
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
            
        )

    def forward (self, x):
        out = x + self.mhsa(self.norm1(x)) # first residual connection 
        out = out + self.mhsa(self.norm2(out))

        return out 


 


