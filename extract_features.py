import os
import random
import argparse
import yaml
from tqdm import tqdm

import jittor as jt
from jittor import nn
from jittor import transform

from datasets import build_dataset
from datasets.utils import build_data_loader
import jclip as clip
from utils import *

from PIL import Image

if jt.has_cuda:
    jt.flags.use_cuda = 1 # Enable CUDA in Jittor


# For positive prompts
def extract_text_feature(args, classnames, prompt_path, clip_model, template):
    f = open(prompt_path)
    prompts = json.load(f)
    with jt.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            
            # positive prompts 
            texts = prompts[classname]
        
            texts_token = clip.tokenize(texts, truncate=True).to(jt.flags.use_cuda)
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = jt.stack(clip_weights, dim=1).to(jt.flags.use_cuda)
        print(clip_weights.shape)
    jt.save(clip_weights, args.cache_dir + "/text_weights_cupl.pkl")
    return


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', dest='root_path', default='./data', help='root path of the data file')
    parser.add_argument('--shots', dest='shots', type=int, default=4, help='shots number')
    parser.add_argument('--cache_dir', dest='cache_dir', default='./caches', help='cache directory path')
    parser.add_argument('--dataset', dest='dataset', default='jittor_dataset', help='data file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = get_arguments()
    clip_model, preprocess = clip.load("./jclip/ViT-B-32.pkl") 
    clip_model.eval()

    k_shot = [4]
    os.makedirs(args.cache_dir, exist_ok=True)

    for k in k_shot:
        
        random.seed(1)
        jt.set_seed(1)
        
        args.shots = k
        dataset = build_dataset(args.dataset, args.root_path, k)

        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

        train_tranform = transform.Compose([
            transform.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=Image.BICUBIC),
            transform.RandomHorizontalFlip(p=0.5),
            transform.ToTensor(),
            transform.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=64, tfm=train_tranform, is_train=True, shuffle=False)
     
        extract_text_feature(args, dataset.classnames, dataset.cupl_path, clip_model, dataset.template)