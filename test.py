import os
import argparse
import yaml
import random

import jittor as jt
from tqdm import tqdm
from jclip import clip

from utils import load_model, cls_acc
from datasets import build_dataset
from datasets.utils import build_data_loader, Datum
from PromptSRCCLIP import CustomCLIP


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest='model_path', default='./model-best.pkl', help='model path')
    parser.add_argument('--root_path', dest='root_path', default='./data', help='root path of the data file')
    parser.add_argument('--shots', dest='shots', type=int, default=4, help='shots number')
    parser.add_argument('--seed', dest='seed', type=int, default='3407', help='seed') 
    parser.add_argument('--dataset', dest='dataset', default='jittor_dataset', help='data file')
    parser.add_argument('--backbone', dest='backbone', default='./jclip/ViT-B-32.pkl', help='backbone path')
    args = parser.parse_args()
    return args


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
    test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
    
    print("Building custom CLIP")
    model = CustomCLIP(args, dataset.classnames, clip_model_prompt)
    
    model = load_model(model, args.model_path)

    model.eval()
    test_acc_list = []
    for images, labels in tqdm(test_loader):
        images, labels = images.to(jt.flags.use_cuda), labels.to(jt.flags.use_cuda)
        logits = model(images)
        test_acc_list.append(cls_acc(logits, labels))
    print(f"test acc: {sum(test_acc_list)/len(test_acc_list)}")

    generate_jittor_result(args, model)