
import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import tools.dataset_tools
import tools.target_model_tools
import tools.algorithm_tools
from utils.registry import registry

if __name__=="__main__":
    # myfunc_data = registry.get_data('cifar10_tool')
    # data_loader = myfunc_data(32, "E:\\tmp\\cifar10")
    # print(type(data_loader))

    # myfunc_model = registry.get_model('image_classification_modeltool')
    # target_model = myfunc_model('awp')
    # print(type(target_model))

    # myfunc_algorithm = registry.get_attack('fgsm_tool')
    # attacker = myfunc_algorithm('resnet18')
    # print(type(attacker))
    info = {'cifar10_tool': {'batch_size': 32, 'dataset_path': '../dataset/cifar10/'}, 'image_classification_modeltool': {'target_model_name': 'Resnet-18'}, 'fgsm_tool': {'target_model_name': 'Resnet-18', 'epsilon': 0.1}, 'attack_tool': {'target_model_name': 'Resnet-18', 'target_model_type': 'Image Classification', 'dataset': 'CIFAR10', 'attack_algorithm': 'FGSM'}}
    print(type(info))
    if 'cifar10_tool' in info.keys():
        myfunc_data = registry.get_data('cifar10_tool')
        data_loader = myfunc_data(**info['cifar10_tool'])
        print(type(data_loader))

    print('本地修改')
    print('远端修改')
