# coding: utf8
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import timm
import torch
from utils.utils import set_seed
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from utils.registry import registry
from model.cifar_controller import CifarController


set_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ModelParamSchema(BaseModel):
    model_name: str = Field("resnet18", title="model_name", description="目标模型的名称")

@registry.register_model("image_classification_modeltool")
def image_classification_getter(model_name:str) -> str:
    '''The function to get image classification model.'''
    model = CifarController(model_name).to(device)
    return model

def image_classification_loader(model_name:str) -> torch.nn.Module:
    '''The function to load image classification model from local.'''
    model = image_classification_getter(model_name)
    return model


image_classification_modeltool = StructuredTool(
    name = "image_classification_modeltool",
    description = "根据目标模型名称获取待攻击或测评的目标模型对象",
    args_schema = ModelParamSchema,
    func = image_classification_getter
)

image_classification_tools = [
    image_classification_modeltool,
]

object_detection_tools = [
    
]

if __name__ == '__main__':
    print(image_classification_modeltool.func)
    myfunc = registry.get_model('image_classification_modeltool')
    print(myfunc('resnet18'))