# coding:utf8
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.llm_zoo import qwen2, deepseek_r1_7b, deepseek_r1_8b, deepseek_r1_14b
from llm.prompt import primary_route_prompt

llm = deepseek_r1_7b()

router_chain = (
    PromptTemplate.from_template(primary_route_prompt) |
    llm |
    StrOutputParser()
)

plan_chain = (
    PromptTemplate.from_template(primary_route_prompt) |
    llm |
    StrOutputParser()
)