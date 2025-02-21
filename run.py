# coding:utf8
import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import time
from agents.chat_router import route
from llm.chain import router_chain as chain
from llm.llm_zoo import qwen2, deepseek_r1_7b, deepseek_r1_14b
from langchain_ollama import ChatOllama
from llm.prompt import openai_agent_prompt, react_agent_prompt, test_prompt, openai_plan_prompt

def answer(question:str, agent_type:str, llm:ChatOllama, system_prompt: str, debug:bool) -> str:
    try:
        messages = route(question, agent_type, llm, system_prompt, debug)
        # messages = agent.invoke({"messages":[("human", question)]}) 
        # messages = agent.invoke({"messages": [("human", question)], "chat_history": chat_history}) # openai
        # messages = agent.invoke({"input":question, "chat_history": chat_history}) # react
    except Exception as e:
        messages = f'Error: {e}'

    return messages

if __name__ == '__main__':
    start_time = time.time()
    llm = deepseek_r1_14b() 
    conversations = 0
    while True:
        print(f'--------------------Conversations:{conversations}-----------------------')
        current_start_time = time.time()
        try:
            # question = input()
            question = ''
            print(question)
            response = answer(question, "openai", llm, openai_plan_prompt, True)
            print(response)
            break
        except ValueError as e:
            print(f'非法输入：{e}')
            break
        except (EOFError, KeyboardInterrupt) as e:
            print(f'输入中断：{e}')
            break
        

        current_end_time = time.time()
        print(f'会话{conversations}用时{current_end_time-current_start_time}s')
        conversations += 1

    end_time = time.time()
    print(f'总耗时：{end_time - start_time}s')