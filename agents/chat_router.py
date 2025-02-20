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
from llm.prompt import openai_plan_prompt, openai_summary_prompt
from agents.creat_agent import create_agent
from tools.algorithm_tools import white_box_attack_tools, black_box_attack_tools
from tools.dataset_tools import dataload_tools
from tools.target_model_tools import image_classification_tools, object_detection_tools
from tools.task_tools import model_attack_tools, model_performance_tools, model_robustness_tools
from utils.registry import registry

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
    reasoning = router_chain.invoke({
        "question": question
    })
    print(reasoning)
    info = reasoning.split('最终结果:')[-1]
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
    planning = plan_chain.invoke({
        "question": question, "chat_history": info, "tools": toolkit
    })
    print(planning)
    print('-'*60)
    info = planning.split('json')[-1].split('```')[0]
    info = json.loads(info)
    print(type(info), info)
    print('-'*60)

    infokeys = list(info.keys())
    # dataset
    dataloader = registry.get_data(infokeys[0])(**info[infokeys[0]])
    # target model
    model = registry.get_model(infokeys[1])(**info[infokeys[1]])
    # attack algorithm
    attacker = registry.get_attack(infokeys[2])(model)
    # task
    clean_accuracy, robust_accuracy = registry.get_task(infokeys[3])(model, attacker, dataloader)

    summary_chain = (
        PromptTemplate.from_template(openai_summary_prompt) |
        llm |
        StrOutputParser()
    )
    summarizing = summary_chain.invoke({
        "question": question, "chat_history": reasoning, "tool_history": planning, "tool_result": clean_accuracy+robust_accuracy
    })
    print('-'*40, '最终答案', '-'*40)
    print(summarizing)
    

if __name__ == '__main__':
    llm = deepseek_r1_14b()
    question = '在cifar10上对图片分类模型wresnet进行攻击并生成对抗样本，攻击算法为FGSM。'
    route(question, "openai", llm, openai_plan_prompt, True)