# ViT - VisionTransformer, a Pytorch implementation

Link to the article: [https://medium.com/artificialis/vit-visiontransformer-a-pytorch-implementation-8d6a1033bdc5](https://medium.com/artificialis/vit-visiontransformer-a-pytorch-implementation-8d6a1033bdc5)

![](https://miro.medium.com/max/700/1*X5jDOEEC_fxmfdk9CdjBHQ.png)

The [Attention is all you need](https://arxiv.org/abs/1706.03762)’s paper revolutionized the world of Natural Language Processing and Transformer-based architecture became the de-facto standard for natural language processing tasks.

It was only a matter of time before someone would actually try to reach the state of the art in Computer Vision, with attention mechanism and transformer architectures.

Despite the fact that convolution-based architectures remain state of the art for image classification tasks, the paper [An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929.pdf) shows that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.
How?

On a very high level, an image is split into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application.

However, natively lacking of CNN’s inherent inductive biases, like locality, Transformers do not generalize well when trained on insufficient amounts of data. It does however reach or beats state of the art on multiple image recognition benchmarks, when trained on large datasets.

Before diving into the implementation, if you never heard of Transformer architectures, I highly recommend to check [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) out.

## Implementation
This is how ViT looks like

![](https://miro.medium.com/max/700/0*65Yla1Sa7IJs-ajU)

You can see the input image being de-composed in 16x16 flattened patches, which are then embedded using a normal fully connected layer, and in front of them, the sum of a special `cls` token and the positional embedding.

This resulting tensor is passed to a standard Transformer encoder, and finally to a MLP head, for classification purposes.

### Table of contents
* Libraries to import
* Patches embeddings: CLS token & Position embedding
* Transformer: Attention, residuals, MLP, Encoder
* Head
* Final ViT

We’ll implement the paper block by block, starting with importing some libraries

```py
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchsummary import summary
from torchvision.transforms import Compose, Resize, ToTensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
```

We’ll also be needing an image to test things out:

```py
img = Image.open('penguin.jpg')

fig = plt.figure()
plt.imshow(img)
plt.show()
```
![](https://miro.medium.com/max/380/1*avKlycJ2HQOGw3k5FF3nNQ.png)

And some small pre-processing:
```py
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])

x = transform(img)
x = x.unsqueeze(0)
print(x.shape)
```

`torch.Size([1,3,224,224])`

Next, we have to break the image into multiple patches, and flatten them.

![](https://miro.medium.com/max/700/0*bWqDPMIOJQuqt03c)
![](https://miro.medium.com/max/700/1*lkErdeAAG8yFA6QQdhyzSw.png)

We can easily use einops to achieve it: 

`patch_size = 16`

`patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)`

The next step would be to project them: this could be easily achieved using a standard linear layer, but the paper implements a convolutional one for performance gains (this is obtained by using a `kernel_size` and `stride` equal to the `patch_size` ). 

Let’s tide things up in the `PatchEmbedding` class.

```py
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        ) # this breaks down the image in s1xs2 patches, and then flat them
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
```

To test it, we can call `PatchEmbedding()(x).shape` :

`torch.Size([1, 196, 768])`

### CLS token & Position embedding
Similarly to BERT’s class token, a learnable embedding is prepended to the sequence of embedded patches. Position embeddings are then added to the patch embeddings to retain positional information. Standard learnable 1D position embeddings are used.

```py
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        ) # this breaks down the image in s1xs2 patches, and then flat them
        
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1) #prepending the cls token
        x += self.positions
        return x
```

The resulting sequence of embedding vectors serves as input to the encoder.

The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multi-headed self-attention and MLP blocks. Layer-norm (LN) is applied before every block, and residual connections after every block.

![](https://miro.medium.com/max/494/0*SHKJDEAq_keTJ6_x)

### Attention
The attention mechanism, as you see, takes three inputs: queries, keys and values. It then computes the attention matrix with queries and keys.

We’ll be implementing a multi-head attention mechanism, hence the computation will be split across multiple heads with a smaller input size.

The main concept consists of using the product between the queries and the keys to get a sort of understanding of how much each element in a sequence is important to the rest.
Such information is later used to scale the values.

We can either use 3 different linear layers for the queries, keys and values matrices, or we can fuse them into a single one.

```py
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3) # queries, keys and values matrix
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values) # sum over the third axis
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        
        return out
```

### Residuals
You can see from the image above that the transformer block has residual connections.

![](https://miro.medium.com/max/310/0*Lt-Pq5Dg4GHUXijG)

We can implement a wrapper to perform the residual addition:

```py
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
```

From the attention block, the output is passed to a fully connected layer. This last one is formed by two layers that upsample by a factor L:

![](https://miro.medium.com/max/331/0*l4nlc9joertFQQQS)

```py
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, L: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, L * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(L * emb_size, emb_size),
        )
```

### Transformer Encoder Block
It’s now time to start tidying things up:

```py
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
                 
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, L=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
```

To test it, you can simply call

```py
patches_embedded = PatchEmbedding()(x)
print(TransformerEncoderBlock()(patches_embedded).shape)
```

Which will return `torch.Size([1,197,768])`

### Transformer
We only need the Encoder from the original transformer, and we can easily use few blocks from our `TransformerEncoderBlock`

```py
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
```

### Classification head
Before tide everything up, we need the last bit — the classification head.
This block, after computing a simple mean over the whole sequence, is a standard fully connected which gives the class probability.

![](https://miro.medium.com/max/396/0*rF-5TxsHpypH8uHM)

```py
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
```

## Tide everything up — VisionTransformer
By composing everything we built so far, we can finally build the ViT architecture.

```py
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
```

We can use `torchsummary` to check out the results:

```py
print(summary(ViT(), (3,224,224), device='cpu'))
```
```
================================================================
Total params: 86,415,592
Trainable params: 86,415,592
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 364.33
Params size (MB): 329.65
Estimated Total Size (MB): 694.56
----------------------------------------------------------------
```

## Conclusion
We learned how to implement the Vision Transformer in Pytorch.
The following is a comparison with the state of the art:

![](https://miro.medium.com/max/700/1*faclyqz1lk_COQL6Syof1w.png)

If you liked the post, consider following me on [Medium](https://alessandroai.medium.com/)

You can join [Artificialis](https://medium.com/artificialis) newsletter, [here](https://sendfox.com/artificialis).

You can also support my work directly and get unlimited access by becoming a Medium member through my referral link [here](https://alessandroai.medium.com/membership)!










