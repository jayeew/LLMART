
import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import tools.dataset_tools
import tools.target_model_tools
import tools.algorithm_tools
import tools.task_tools
from utils.registry import registry

if __name__=="__main__":
    # 测试
    # myfunc_data = registry.get_data('cifar10_tool')
    # data_loader = myfunc_data(32, "E:\\tmp\\cifar10")
    # print(type(data_loader))

    # myfunc_model = registry.get_model('image_classification_modeltool')
    # target_model = myfunc_model('awp')
    # print(type(target_model))

    # myfunc_algorithm = registry.get_attack('fgsm_tool')
    # attacker = myfunc_algorithm('resnet18')
    # print(type(attacker))
    info = {'cifar10_tool': {'batch_size': 128, 'dataset_path': '../dataset/cifar10/'}, 'image_classification_modeltool': {'model_name': 'awp'}, 'fgsm_tool': {'target_model_name': 'awp', 'epsilon': 0.1}, 'attack_tool': {'target_model_name': 'Resnet-18', 'target_model_type': 'Image Classification', 'dataset': 'CIFAR10', 'attack_algorithm': 'FGSM'}}
    # print(type(info))
    # model = registry.get_model('image_classification_modeltool')(**info['image_classification_modeltool'])
    # dataloader = registry.get_data('cifar10_tool')(**info['cifar10_tool'])
    # attacker = registry.get_attack('fgsm_tool')(model)
    # clean_accuracy, robust_accuracy = registry.get_task('attack_tool')(model, attacker, dataloader)
    # print(clean_accuracy, robust_accuracy)
    print(list(info.keys())[0])