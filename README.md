
#### 第四届计图挑战赛

# Jittor 开放域少样本视觉分类赛题--B榜

## 方法

本赛题的核心目标聚焦于如何在有限的数据资源下，增强模型在特定领域内的表现能力。这要求能够在不损害模型原有广泛适用性的前提下，将预训练的基础模型有效迁移至特定的下游任务中。因此我们基于[PromptSRC](https://arxiv.org/abs/2307.06948)实现本方法。

为实现这一目标，我们聚焦于三大关键点：一是深入挖掘预训练模型的特征潜力，二是细致分析并优化训练过程中的轨迹变化，三是积极利用文本多样性来丰富特征提取过程。通过这些措施，我们旨在以少量的样本数据高效训练模型，确保其在跨越多领域数据集的测试过程中，能够展现出卓越的分类准确性和稳定性。

具体来说：首先保留了预训练视觉语言模型特征中的通用zero-shot知识，这些特征在冻结状态下仍具备可迁移性，但缺乏针对特定任务的知识。为了弥补这一不足，引入了增强文本提示机制，旨在通过最大化提示与冻结模型特征之间的一致性，使模型能够更好地适应给定的下游任务。这一过程不仅调节了学习提示，还确保了其在任务中的有效性。

然后，鉴于视觉编码器能够处理每个类别下的多个图像样本，而文本编码器则受限于每个类别仅有一个文本标签的多样性不足问题，直接对多模态特征施加相互协议约束可能会导致性能受限。为了克服这一挑战，采用了针对每个类别设计不同文本标签模板的方法来规范提示。这一举措有效缓解了文本侧标签缺乏多样性的问题，从而提升了模型的整体性能。

![方法](https://img.picui.cn/free/2024/08/19/66c3063c48467.jpg)

上图中，CLIP编码器用于在图像和文本两个维度上生成提示及预训练特征。首先，为了丰富文本侧的多样性，引入了文本增强，以此生成一系列带有差异化的文本特征，这些特征来自冻结的模型。随后，我们通过对这些特征进行平均化处理，来构建出一个综合性的预训练模型文本特征向量。紧接着，我们运用了相互协议最大化约束（SCL Loss）的方法，以精细调节生成的提示，确保这些提示在特征和逻辑层面上，都能与预训练的模型表示紧密对齐。鉴于CLIP模型处于冻结状态，统一使用其内置的模型编码器来提取这两种类型的特征。最后，这些经过集成处理的视觉和文本提示被联合应用于模型的推理过程中。

## 环境配置

操作系统: CentOS Linux release 7.9.2009

语言: python 3.9.19

框架: jittor 1.3.8.5

CPU: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz

GPU: NVIDIA A800 80GB PCIe

Cuda: 12.2

## 训练模型

### 下载预训练模型

下载Jittor版本CLIP ViT-B/32预训练模型[Vit-B-32.pkl](https://github.com/uyzhang/JCLIP/releases/download/%E6%9D%83%E9%87%8D/ViT-B-32.pkl)放在jclip文件夹下。

### 数据预处理

将训练集[TrainSet.zip](https://cloud.tsinghua.edu.cn/f/7c44b138a6344f4b8fd1/?dl=1)和[B榜测试集](https://cloud.tsinghua.edu.cn/f/fb732f6217db4c32a3c0/?dl=1)下载解压到 `<root>/data` 下。 数据目录结构如下

```
data/
    TestSetB
        image_1.jpg
        image_2.jpeg
        ...
    TrainSet
        Animal
            Bear
                1.jpg
                2.jpg
                ...
            Bee
                ...
        Caltech-101
            ...
        Food-101
            ...
        Thu-dog
            ...
```

## 训练模型

```shell
bash tran.sh 
```

## 推理结果

```shell
python test.py --model_path=./model-best.pkl
```

最后结果保存在data/result.txt

*注意*: 通过--model_path参数指定推理使用的模型权重

## 开源仓库链接

gitlink开源仓库地址：https://www.gitlink.org.cn/longinhit/jittor-qwe-promptsrc_jittor

github开源仓库地址：https://github.com/longinhit/jittor-qwe-promptsrc_jittor

## 模型参数量

模型参数总数：229M (228,975,201)
