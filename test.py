
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

    myfunc_model = registry.get_model('image_classification_modeltool')
    target_model = myfunc_model('awp')
    print(type(target_model))

    # myfunc_algorithm = registry.get_attack('fgsm_tool')
    # attacker = myfunc_algorithm('resnet18')
    # print(type(attacker))

    