# coding: utf8
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from utils.utils import set_seed
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.tools import StructuredTool
from apis.fgsm import FGSM
import torch
import timm
from utils.registry import registry

set_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FGSMParamSchema(BaseModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True)
    target_model: str = Field(None, title="target_model", description="被FGSM攻击的目标模型的名称.")

@registry.register_attack("fgsm_tool")
def fgsm_getter(target_model:str) -> FGSM:
    '''The function to get fgsm tool.'''
    model = timm.create_model(target_model, pretrained=True)
    return FGSM(model, device)

fgsm_tool = StructuredTool(
    name = "fgsm_tool",
    description = "根据目标模型名称创建FGSM攻击工具",
    args_schema = FGSMParamSchema,
    func = fgsm_getter
)

white_box_attack_tools = [
    fgsm_tool,
]

black_box_attack_tools = [
    
]

if __name__ == '__main__':
    print(fgsm_tool.func)
    myfunc = registry.get_attack('fgsm_tool')
    print(myfunc('resnet18'))