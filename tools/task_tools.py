# coding: utf8
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class AttackParamSchema(BaseModel):
    target_model_name: str = Field(None, title="target_model_name", description="目标模型的名称.")
    target_model_type: str = Field(None, title="target_model_type", description="目标模型的类型，比如图像分类或目标检测.")
    dataset: str = Field(None, title="dataset", description="用于对目标模型攻击或生成对抗样本的数据集.")
    attack_algorithm: str = Field(None, title="attack_algorithm", description="攻击目标模型时用到的攻击算法.")

def run_attack(target_model_name:str, target_model_type:str, dataset:str, attack_algorithm:str) -> str:
    print("正在攻击中...")
    # return "攻击成功！生成的对抗样本已经存储在本地路径: /path/to/adversarial_sample"
    return "攻击失败！"

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