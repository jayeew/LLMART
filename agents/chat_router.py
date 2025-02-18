# coding:utf8
import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.chain import router_chain
from llm.llm_zoo import qwen2, deepseek_r1_7b, deepseek_r1_14b
from llm.prompt import openai_plan_prompt
from agents.creat_agent import create_agent
from tools.algorithm_tools import white_box_attack_tools, black_box_attack_tools
from tools.dataset_tools import dataload_tools
from tools.target_model_tools import image_classification_tools, object_detection_tools
from tools.task_tools import model_attack_tools, model_performance_tools, model_robustness_tools

toolkits_dict = {
    'FGSM': white_box_attack_tools,
    'PGD': white_box_attack_tools,
    'ImageNet': dataload_tools,
    'CIFAR10': dataload_tools,
    'Image Classification': image_classification_tools,
    'Object Detection': object_detection_tools,
    'Attack': model_attack_tools,
    'Model Performance Test': model_performance_tools,
    'Model Robustness Test': model_robustness_tools
}

def route(question:str, agent_type:str, llm:ChatOllama, system_prompt: str, debug:bool) -> str:
    info = router_chain.invoke({
        "question": question
    })
    print(info)
    info = info.split('最终结果:')[-1]
    print('-'*60)
    toolkit = []
    for kind in ['FGSM', 'PGD', 'ImageNet', 'CIFAR10', 'Image Classification', 'Object Detection', 
                 'Attack', 'Model Performance Test', 'Model Robustness Test']:
        if kind in info:
            toolkit.extend(toolkits_dict[kind])

    plan_chain = (
        PromptTemplate.from_template(system_prompt) |
        llm |
        StrOutputParser()
    )
    info = plan_chain.invoke({
        "question": question, "chat_history": info, "tools": toolkit
    })
    print(info)
    print('-'*60)
    info = info.split('json')[-1].split('```')[0]
    info = json.loads(info)
    print(type(info), info)
    print('-'*60)

if __name__ == '__main__':
    llm = deepseek_r1_14b()
    question = '在cifar10上对图片分类模型wresnet进行攻击并生成对抗样本，攻击算法为FGSM。'
    route(question, "openai", llm, openai_plan_prompt, True)