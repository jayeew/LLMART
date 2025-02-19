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
    batch_size: int = Field(32, title="batch_size", description="加载数据集的批次大小")
    dataset_path: str = Field("..\\datasets\\cifar10", title="dataset_path", description="数据集的存储路径")

@registry.register_data('cifar10_tool')
def cifar10_loader(batch_size: int = 32, dataset_path: str = "..\\datasets\\cifar10") -> DataLoader: 
    '''The function to create cifar10 dataloader.'''
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    data_loader = DataLoader(cifar, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory= False, drop_last= False)
    data_loader.name = "cifar10"
    data_loader.batch = batch_size
    return data_loader

cifar10_tool = StructuredTool(
    name = "cifar10_tool",
    description = "根据加载数据集的批次大小和数据集的存储路径创建cifar10数据集加载器",
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