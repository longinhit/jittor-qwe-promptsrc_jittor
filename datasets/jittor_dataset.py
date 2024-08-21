import sys
import os
import os.path as osp
import json
import numpy as np

from .utils import Datum, DatasetBase, read_json

template = ['a photo of a {}.']
negative_template = ['a photo without {}.']

class JittorDataSet(DatasetBase):
    dataset_dir = ''
    def __init__(self, root, num_shots):
        train = []
        val = []
        test = []
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'TrainSet')
        self.split_path = os.path.join(self.dataset_dir, 'split_jittor-dataset.json')
        
        self.template = template
        self.negative_template = negative_template
        self.cupl_path = './prompts/CuPL_prompts_jittor_dataset.json'
        train, val, test = self.read_split(self.split_path, self.dataset_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
        self._lab2cname, self._classnames = self.all_lab2cname(self.dataset_dir)
        self._num_classes = len(self._classnames)

    def all_lab2cname(self, dataset_dir):
        container = set()
        convert_map={
            "faces": "face",
            "faces_easy": "face",
            "leopards": "leopard",
            "motorbikes":"motorbike",
            "airplanes":"airplane"
        }
        classes_name_path = os.path.join(dataset_dir, 'classes_b.txt')
        with open(classes_name_path, 'r') as fd:
            for line in fd.readlines():
                if line.endswith('\n'):
                    line = line[:-1]
                class_name, label = tuple(line.split(' '))
                if class_name.startswith('Stanford-Cars'):
                    class_name = class_name[class_name.find('_')+1:].lower()
                    class_name = class_name[-4:] + "_" + class_name[:-5]
                else:
                    class_name = class_name[class_name.find('_')+1:].lower()
                if class_name in convert_map:
                    print(f"{class_name} to {convert_map[class_name]}")
                    class_name = convert_map[class_name]
                container.add((int(label), class_name))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def read_split(self, filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
