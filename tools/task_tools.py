# coding: utf8
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.registry import registry
from utils.utils import set_seed
from apis.metrics import AverageMeter, accuracy

set_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AttackParamSchema(BaseModel):
    target_model_name: str = Field(None, title="target_model_name", description="目标模型的名称.")
    target_model_type: str = Field(None, title="target_model_type", description="目标模型的类型，比如图像分类或目标检测.")
    dataset: str = Field(None, title="dataset", description="用于对目标模型攻击或生成对抗样本的数据集.")
    attack_algorithm: str = Field(None, title="attack_algorithm", description="攻击目标模型时用到的攻击算法.")

def run_attack(target_model_name:str, dataset:str, attack_algorithm:str) -> str:
    print("正在攻击中...")
    # return "攻击成功！生成的对抗样本已经存储在本地路径: /path/to/adversarial_sample"
    return "攻击成功！"

@registry.register_task('attack_tool')
def execute_attack(target_model:torch.nn.Module, attack_algorithm:object, data_loader:DataLoader) -> str:
    top1_m = AverageMeter()
    adv_top1_m = AverageMeter()
    for idx, (images, labels) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing"):
    # for i, (images, labels) in enumerate(data_loader):
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)

        # clean acc， 每轮攻击后计算正常样本下的模型准确率
        with torch.no_grad():
            logits = target_model(images)
        clean_acc = accuracy(logits, labels)[0]
        top1_m.update(clean_acc.item(), batchsize)

        # robust acc，每轮攻击后计算对抗样本下的模型准确率
        adv_images = attack_algorithm(images = images, labels = labels, target_labels = None)
        # if args.attack_name == 'autoattack':
        #     if adv_images is None:
        #         adv_acc = 0.0
        #     else:
        #         adv_acc = adv_images.size(0) / batchsize * 100
        # else:
        with torch.no_grad():
            adv_logits = target_model(adv_images)
        adv_acc = accuracy(adv_logits, labels)[0]
        adv_acc = adv_acc.item()
        adv_top1_m.update(adv_acc, batchsize)

    clean_accuracy = "Clean accuracy is {}%".format(round(top1_m.avg, 2))
    robust_accuracy = "Robust accuracy is {}%".format(round(adv_top1_m.avg, 2))

    return clean_accuracy, robust_accuracy

attack_tool = StructuredTool(
    name = "attack_tool",
    description = "根据目标模型名称、目标模型类型、数据集和攻击算法对目标模型进行攻击或生成对抗样本, 并返回攻击结果.",
    args_schema = AttackParamSchema,
    func = run_attack
)

model_attack_tools = [
    attack_tool,
]

model_performance_tools = [

]

model_robustness_tools = [
    
]