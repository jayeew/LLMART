# codinf:utf8
from langchain.agents import (AgentExecutor, 
                            create_openai_tools_agent, 
                            create_structured_chat_agent,
                            create_react_agent)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import hub
from langchain_ollama import ChatOllama
import torch



def create_agent(agent_type: str, toolkit: list, llm: ChatOllama, system_prompt: str, debug:bool = True) :
    if agent_type == "openai":
        print("创建openai_tools_agent......")
        return openai_tools_agent(toolkit, llm, system_prompt, debug)
    elif agent_type == "react":
        print("创建react_chat_agent......")
        return react_chat_agent(toolkit, llm, system_prompt, debug)
    elif agent_type == "plan_and_execute":
        return plan_and_execute_agent(toolkit, llm, system_prompt, debug)
    elif agent_type == "structured_chat":
        return structured_chat_agent(toolkit, llm, system_prompt, debug)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
def openai_tools_agent(toolkit: list, llm: ChatOllama, system_prompt: str, debug:bool) -> AgentExecutor:
    """
    run:
        agent.invoke({"messages":[("human", question)]})
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    except Exception as e:
        print(f"Error: {e}")
    agent = create_openai_tools_agent(llm, toolkit, prompt)
    # agent = create_openai_tools_agent(llm, toolkit, prompt)
    executor = AgentExecutor(agent=agent, tools=toolkit, verbose=debug)
    return executor

def react_chat_agent(toolkit: list, llm: ChatOllama, system_prompt: str, debug:bool) -> AgentExecutor:

    prompt = PromptTemplate.from_template(system_prompt)
    agent = create_react_agent(llm, toolkit, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit,
        verbose=debug, 
        handle_parsing_errors=True,  # 是否处理解析错误，如果解析不了，会重新尝试，但可能会一直重复死循环
        max_iterations=1,  # 如果重新尝试，最大尝试次数，防止无限死循环下去
        return_intermediate_steps=True,  # 如果死循环了，可以加这个参数，返回中间过程步骤，中间步骤可以当结果
        # early_stopping_method="generate",  # 这个参数没有实现有问题：https://github.com/langchain-ai/langchain/issues/16263
        max_execution_time=60,  # 如果死循环了，可以加这个参数，强制停止,通过时间来进行循环的限制
    )
    return agent_executor

def plan_and_execute_agent(question: str, toolkit: list, llm: ChatOllama, system_prompt: str, debug:bool) -> PlanAndExecute:
    """
    run:
        agent.run(question)
    """
    planer = load_chat_planner(llm=llm, system_prompt=system_prompt)
    executor = load_agent_executor(llm=llm, tools=toolkit, verbose=debug)
    agent = PlanAndExecute(planer=planer, executor=executor, verbose=debug)
    return agent

def structured_chat_agent(question: str, toolkit: list, llm: ChatOllama, system_prompt: str, debug:bool) -> AgentExecutor:
    """
    run:
        agent.invoke({"input":question})
    """
    agent = create_structured_chat_agent(llm, toolkit, system_prompt)
    executor = AgentExecutor(agent=agent, 
                             tools=toolkit, 
                             verbose=debug,
                             handle_parsing_errors=True)
    return executor
