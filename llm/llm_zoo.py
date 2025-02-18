# coding:utf8
from langchain_ollama import ChatOllama

def qwen2(temperature:float =0.1):
    llmodel = ChatOllama(model="qwen2",
                       temperature=temperature,)
    return llmodel

def deepseek_r1_7b(temperature:float =0.1):
    llmodel = ChatOllama(model="deepseek-r1:7b",
                       temperature=temperature,)
    return llmodel

def deepseek_r1_8b(temperature:float =0.1):
    llmodel = ChatOllama(model="deepseek-r1:8b",
                       temperature=temperature,)
    return llmodel

def deepseek_r1_14b(temperature:float =0.1):
    llmodel = ChatOllama(model="deepseek-r1:14b",
                       temperature=temperature,)
    return llmodel