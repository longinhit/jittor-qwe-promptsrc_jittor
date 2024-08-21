import argparse
import yaml
from tqdm import tqdm
import random
from PIL import Image

import jittor as jt
from jittor import init
from jittor import nn, Module
from jittor import transform
from jittor import optim
from jittor import lr_scheduler

from jclip import clip
from jclip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from datasets import build_dataset
from datasets.utils import build_data_loader, Datum
from utils import *

if jt.has_cuda:
    jt.flags.use_cuda = 1 # Enable CUDA in Jittor
  
_tokenizer = _Tokenizer()

def load_clip_to_cpu(args, zero_shot_model=False):
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 4,
                          "language_depth": 4,
                          "vision_ctx": 9,
                          "language_ctx": 9}
        model, _ = clip.load(args.backbone, design_details = design_details)
        return model
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model, _ = clip.load(args.backbone, design_details = design_details)
        return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def execute(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.cast(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x) # [77,403,512,]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).cast(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        tokenized_prompts_tuple = jt.argmax(tokenized_prompts, dim=1) # <class 'tuple'> 
        tokenized_prompts = jt.array(tokenized_prompts_tuple[0]) # Convert the tuple: jt.Var to a Jittor tensor 
        x = x[jt.arange(x.shape[0]), tokenized_prompts] @ self.text_projection

        return x 


class VLPromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model, fixed_embeddings):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_version = 4
        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with jt.no_grad():
                embedding = clip_model.token_embedding(prompt).cast(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = jt.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {n_ctx_version}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = jt.concat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp_image = load_clip_to_cpu(args, True)
        with jt.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).cast(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual

        self.fixed_embeddings = fixed_embeddings
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = jt.concat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def execute(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.clip_weights_cupl = jt.load(os.path.join('caches', "text_weights_cupl.pkl"))
        print(f"**** clip_weights_cupl shape: {self.clip_weights_cupl.shape}.")
        self.prompt_learner = VLPromptLearner(args, classnames, clip_model, self.clip_weights_cupl)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)


    def execute(self, image, label=None):
        logit_scale = self.logit_scale.exp()

        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = self.image_encoder(image.cast(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.transpose()
        
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.clip_weights_cupl.transpose()
            fixed_embeddings = jt.array(fixed_embeddings)

            with jt.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.cast(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.to(jt.flags.use_cuda) @ fixed_embeddings.to(jt.flags.use_cuda).transpose()
            return logits, text_features, fixed_embeddings, zero_shot_features, image_features, zero_shot_logits
        else:
            return logits


def get_jittor_test_loader(args, clip_model, preprocess):
    test_data_dir = 'data/TestSetA/'
    image_list =[]
    for file_name in os.listdir(test_data_dir):
        impath = os.path.join(test_data_dir, file_name)
        item = Datum(
            impath=impath,
            label=-1,  # 竞赛测试集没有label
            classname='unknown' #  竞赛测试集没有classname
        )
        image_list.append(item)
    test_loader = build_data_loader(data_source=image_list, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    return test_loader,image_list


def generate_jittor_result(args, model):
    image_list =[]
    test_data_dir='data/TestSetB/'
    for file_name in os.listdir(test_data_dir):
        impath = os.path.join(test_data_dir, file_name)
        item = Datum(
            impath=impath,
            label=-1,  # 竞赛测试集没有label
            classname='unknown' #  竞赛测试集没有classname
        )
        image_list.append(item)
    predict_result_loader = build_data_loader(data_source=image_list, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    model.eval()
    preds = []
    with jt.no_grad():
        for images, _ in tqdm(predict_result_loader):
            output = model(images.to(jt.flags.use_cuda))
            preds += jt.tolist(output.topk(5, 1, True, True)[1])
    with open('./data/result.txt', 'w') as save_file:
       for index in range(len(preds)):
            image_file = image_list[index].impath
            image_file = image_file[image_file.rfind('/')+1:]
            top5_idx = preds[index]
            save_file.write(image_file + ' ' + ' '.join(str(idx) for idx in top5_idx) + '\n')


def PromptSRC(args, model, train_loader_F, val_loader, test_loader):
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            # Make sure that VPT prompts are updated
            if "VPT" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            if "ZS_image_encoder" in name:
                param.requires_grad = False

    # Double check
    enabled = set()
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
            params_to_update.append(param)
    print(f"Parameters to be updated: {sorted(list(enabled))}")
    print(f"Parameters count: {len(enabled)}")
    
    optimizer = jt.optim.AdamW([
        {'params': params_to_update, 'lr': args.lr * 3, 'eps': args.eps, 'weight_decay': 1e-1}],
        lr=args.lr,  
        eps=args.eps)
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch * len(train_loader_F))
    Loss = SmoothCrossEntropy()

    for train_idx in range(args.train_epoch):
        # Train
        model.train()
        print('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
        acc_list, loss_list = [], [] 
        for i, (images, labels) in enumerate(tqdm(train_loader_F)):
            images, labels = images.to(jt.flags.use_cuda), labels.to(jt.flags.use_cuda)
            
            logits, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, zero_shot_logits = model(images, labels)
            
            loss_ce = Loss(logits,  labels)

            # Calculate the L_SCL_text loss
            scl_text_loss = nn.l1_loss(normalized_text_features, zs_clip_text_embeddings.to(jt.flags.use_cuda))
            # Calculate the lcl_image_loss_mean
            scl_text_loss_mean = jt.mean(scl_text_loss)
            loss_scl_text = scl_text_loss_mean * 25

            # Calculate the L_SCL_image loss
            scl_image_loss = nn.l1_loss(image_ft, zs_image_embedd.to(jt.flags.use_cuda)) 
            # Calculate the scl_image_loss_mean
            scl_image_loss_mean = jt.mean(scl_image_loss)
            loss_scl_image = scl_image_loss_mean * 10
            
            # Now calculate L_SCL_logits
            kl_loss = nn.KLDivLoss(reduction="sum", log_target=True)
            L_SCL_logits = kl_loss(nn.log_softmax(logits / 1, dim=1), nn.log_softmax(zero_shot_logits / 1, dim=1)) * (1 * 1) / logits.numel()
            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            loss = (loss_ce + L_SCL)
            loss_list.append(loss.item())
            acc_list.append(cls_acc(logits, labels))
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            scheduler.step()
        current_lr = optimizer.lr 
        if train_idx % 10 == 0 or train_idx == args.train_epoch-1:
            model.eval()
            val_acc_list = []
            for images, labels in tqdm(val_loader):
                images, labels = images.to(jt.flags.use_cuda), labels.to(jt.flags.use_cuda)
                logits = model(images)
                acc = cls_acc(logits, labels)
                val_acc_list.append(acc)
            print(f"LR: {current_lr}, Train Acc: {sum(acc_list)/len(acc_list) :.4f}, \
                    Loss: {sum(loss_list)/len(loss_list):.4f}, val acc: {sum(val_acc_list)/len(val_acc_list)}")
        else:
            print(f"LR: {current_lr}, Train Acc: {sum(acc_list)/len(acc_list) :.4f} \
                    Loss: {sum(loss_list)/len(loss_list):.4f}")
    model.eval()
    test_acc_list = []
    for images, labels in tqdm(test_loader):
        images, labels = images.to(jt.flags.use_cuda), labels.to(jt.flags.use_cuda)
        logits = model(images)
        test_acc_list.append(cls_acc(logits, labels))
    print(f"test acc: {sum(test_acc_list)/len(test_acc_list)}")
    
    return model, sum(test_acc_list)/len(test_acc_list)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', dest='root_path', default='./data', help='root path of the data file')
    parser.add_argument('--shots', dest='shots', type=int, default=4, help='shots number')
    parser.add_argument('--seed', dest='seed', type=int, default='3407', help='seed') 
    parser.add_argument('--dataset', dest='dataset', default='jittor_dataset', help='data file')
    parser.add_argument('--backbone', dest='backbone', default='./jclip/ViT-B-32.pkl', help='backbone path')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--eps', dest='eps', type=float, default=0.001, help='eps') 
    parser.add_argument('--train_epoch', dest='train_epoch', type=int, default=50, help='train epoch')
    parser.add_argument('--save_model_dir', dest='save_model_dir', default='./save_model', help='save_model directory path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()

    random.seed(args.seed)
    jt.set_seed(args.seed)

    print(f"Loading CLIP (backbone: {args.backbone})")
    clip_model, preprocess = clip.load(args.backbone)
    clip_model_prompt, preprocess = clip.load(args.backbone, design_details = {"trainer": 'IVLP',
                      "vision_depth": 9,
                      "language_depth": 9, 
                      "vision_ctx": 4,
                      "language_ctx": 4})
    
    print(f"build {args.dataset} dataset ")
    dataset = build_dataset(args.dataset, args.root_path, args.shots)
    train_tranform = transform.Compose([
            transform.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=Image.BICUBIC),
            transform.RandomHorizontalFlip(p=0.5),
            transform.ToTensor(),
            transform.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=64, tfm=train_tranform, is_train=True, shuffle=True)
    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    
    print("Building custom CLIP")
    model = CustomCLIP(args, dataset.classnames, clip_model_prompt)
    
    # 统计模型的参数数量
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()

    print(f"Total number of parameters: {total_params}")
    
    model, test_acc = PromptSRC(args, model, train_loader_F, val_loader, test_loader)
    
    os.makedirs(args.save_model_dir, exist_ok=True)
    save_model(model, args.save_model_dir, test_acc)