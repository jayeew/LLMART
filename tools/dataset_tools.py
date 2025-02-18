# coding: utf8
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from utils.registry import registry

class DatasetParamSchema(BaseModel):
    batchsize: int = Field(32, title="batchsize", description="加载数据集的批次大小")
    cifar10_path: str = Field("E:\\tmp\\cifar10", title="dataset_path", description="数据集的存储路径")

@registry.register_data('cifar10_tool')
def cifar10_loader(batchsize: int = 32, dataset_path: str = "E:\\tmp\\cifar10") -> DataLoader: 
    '''The function to create cifar10 dataloader.'''
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    data_loader = DataLoader(cifar, batch_size=batchsize, shuffle=False, num_workers=1, pin_memory= False, drop_last= False)
    data_loader.name = "cifar10"
    data_loader.batch = batchsize
    return data_loader

cifar10_tool = StructuredTool(
    name = "cifar10_tool",
    description = "批次大小和数据集的存储路径创建cifar10数据集加载器",
    args_schema = DatasetParamSchema,
    func = cifar10_loader
)

dataload_tools = [
    cifar10_tool,
]

if __name__=="__main__":
    myfunc = registry.get_data('cifar10_tool')
    data_loader = myfunc(32, "E:\\tmp\\cifar10")
    print(type(data_loader))