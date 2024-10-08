from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
from .mha import MultiheadAttention
import jittor as jt
from jittor import nn, init

def normal_(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.gauss_(module.weight, mean, std)
    if hasattr(module, 'bias') and isinstance(
            module.bias, jt.Var) and module.bias is not None:
        init.constant_(module.bias, bias)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def execute(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(jt.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def execute(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = jt.concat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = jt.attention.MultiheadAttention(
            # F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=jt.concat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0] # add


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def execute(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def execute(self, x):
        ret = super().execute(x)
        return ret


class QuickGELU(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(1.702 * x)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)

    def execute(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask = None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def execute(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def expand_new(x, *shape): # 重写expand函数
    ''' Expand and broadcast this array, -1 represents this dimension is not changed.
    Example::

    a = jt.zeros((3,1))
    b = a.expand(3, 4)
    assert b.shape == (3,4)
    b = a.expand(-1, 4)
    assert b.shape == (3,4)
    b = a.expand((3, 4))
    assert b.shape == (3,4)
    b = a.expand((-1, 4))
    assert b.shape == (3,4)

    '''
    if len(shape) == 1 and isinstance(shape[0], (tuple,list,jt.NanoVector)):
        shape = shape[0]
    shape = list(shape)
    for i in range(len(shape)):
        if shape[i] == -1:
            shape[i] = x.shape[i-1]
    return x.broadcast(shape)
jt.Var.expand_new = expand_new


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = jt.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = jt.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def execute(self, x): 
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer: 
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand_new(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                x = jt.concat([prefix, visual_context], dim=0) 
            else: 
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand_new(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = jt.concat([prefix, textual_context, suffix], dim=0) 

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers

        # Implements respective encoder blocks for a given design choice
        if design_details and (design_details['trainer'] == 'IVLP' or design_details['trainer'] == 'VPT'):
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif design_details and design_details['trainer'] == 'MaPLe':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
        elif design_details and (design_details['trainer'] == 'CoOp' or design_details['trainer'] == 'CoCoOp'):
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        else:
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def execute(self, x):
        return self.resblocks(x)



class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        if not design_details or design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = jt.empty(n_ctx, width)
            normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * jt.randn(width))
        self.positional_embedding = nn.Parameter(scale * jt.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.transformer = Transformer(width, layers, heads, prompts_needed = design_details["vision_depth"] if design_details and 'vision_depth' in design_details  else 0,
                                       design_details=design_details) # add

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * jt.randn(width, output_dim))

    def execute(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = jt.concat(
            [self.class_embedding.to(x.dtype) + jt.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand_new(x.shape[0], -1, -1).half()
            x = jt.concat([x, visual_ctx], dim=1)

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details = None # add
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            design_details=design_details)

        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=design_details['language_depth'] if design_details and 'language_depth' in design_details else 0,
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(jt.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(jt.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(jt.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        normal_(self.token_embedding.weight, std=0.02)
        normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                normal_(self.visual.attnpool.q_proj.weight, std=std)
                normal_(self.visual.attnpool.k_proj.weight, std=std)
                normal_(self.visual.attnpool.v_proj.weight, std=std)
                normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            normal_(block.attn.in_proj_weight, std=attn_std)
            normal_(block.attn.out_proj.weight, std=proj_std)
            normal_(block.mlp.c_fc.weight, std=fc_std)
            normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = jt.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask = jt.triu(mask, 1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_tuple = jt.argmax(text, dim=-1) # <class 'tuple'> 
        text = jt.array(text_tuple[0]) # Convert the tuple: jt.Var to a Jittor tensor 
        x = x[jt.arange(x.shape[0]), text] @ self.text_projection

        return x

    def execute(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.transpose()
        logits_per_text = logit_scale * text_features @ image_features.transpose()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, jt.attention.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details = None):
    vit = "visual.proj" in state_dict

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)

    return model.eval()
