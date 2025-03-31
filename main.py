from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from os import getenv
from dotenv import load_dotenv
import os
import json
import asyncio
import sys
from datetime import datetime
import httpx

os.environ["https_proxy"] = "socks5://127.0.0.1:7897"
load_dotenv()

# 配置API端点
API_BASE_URL = "http://localhost:8000/api"

def create_llm():
    """创建LLM实例"""
    return ChatOpenAI(
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-r1:free",
        temperature=0
    )

def get_prompts():
    """获取预定和查询的提示模板"""
    booking_template = """你是一个智能助手，专门帮助用户预定会议室。
当前日期是：{current_date}

请从用户的请求中提取以下信息，并根据当前日期计算具体日期：
- 会议室名称
- 预定日期（如果用户说"今天"，使用当前日期；"明天"和"周几"等相对日期，请根据当前日期计算）
- 开始时间
- 结束时间

请以 JSON 格式返回，例如：
{{
    "会议室": "宜山厅",
    "日期": "2024-01-19",
    "开始时间": "17:00",
    "结束时间": "18:00"
}}

用户请求：
{user_input}

请输出 JSON;"""

    query_template = """你是一个智能助手，专门帮助用户查询会议室的可用状态。
当前日期是：{current_date}

请从用户的请求中提取以下信息，并根据当前日期计算具体日期：
- 查询日期（如果用户说"今天"，使用当前日期；"明天"和"周几"等相对日期，请根据当前日期计算）
- 查询时间段（可选）
- 需要查询的会议室（可选）

请以 JSON 格式返回，例如：
{{
    "日期": "2024-01-19",
    "时间段": {{
        "开始时间": "14:00",
        "结束时间": "16:00"
    }},
    "会议室列表": ["宜山厅", "徐汇厅"]
}}

如果用户没有指定具体时间段，则返回整天的查询请求：
{{
    "日期": "2024-01-19",
    "时间段": "全天",
    "会议室列表": ["宜山厅", "徐汇厅"]
}}

如果用户没有指定具体会议室，则返回所有会议室：
{{
    "日期": "2024-01-19",
    "时间段": "全天",
    "会议室列表": "所有"
}}

用户请求：
{user_input}

请输出 JSON;"""

    booking_prompt = PromptTemplate(
        template=booking_template, 
        input_variables=["user_input", "current_date"]
    )
    query_prompt = PromptTemplate(
        template=query_template, 
        input_variables=["user_input", "current_date"]
    )
    
    return booking_prompt, query_prompt

def create_chains():
    """创建预定和查询的处理链"""
    llm = create_llm()
    booking_prompt, query_prompt = get_prompts()
    
    # 创建处理链
    booking_chain = (
        RunnablePassthrough() 
        | booking_prompt 
        | llm
    )
    
    query_chain = (
        RunnablePassthrough() 
        | query_prompt 
        | llm
    )
    
    return booking_chain, query_chain

def process_llm_response(response):
    """处理LLM响应"""
    try:
        content = response.content if hasattr(response, 'content') else str(response)
        print("LLM Response content:", content)
        clean_text = content.strip("```json").strip("```").strip()
        return json.loads(clean_text)
    except (json.JSONDecodeError, AttributeError) as e:
        return {"error": f"无法解析响应: {str(e)}"}

async def call_booking_api(booking_data):
    """调用预订API"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_BASE_URL}/book", json=booking_data)
            return response.json()
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return {"成功": False, "信息": f"API调用失败: {str(e)}"}

async def call_query_api(query_data):
    """调用查询API"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_BASE_URL}/query", json=query_data)
            return response.json()
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return {"错误": f"API调用失败: {str(e)}"}

async def process_request(user_input: str, mode: str = "booking"):
    """处理用户请求并发送到API"""
    booking_chain, query_chain = create_chains()
    chain = booking_chain if mode == "booking" else query_chain
    
    try:
        # 获取当前日期
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # 构造输入数据
        input_data = {
            "user_input": user_input,
            "current_date": current_date
        }
        
        # 获取LLM响应并处理
        response = await chain.ainvoke(input_data)
        llm_result = process_llm_response(response)
        
        # 检查处理是否成功
        if "error" in llm_result:
            return {"error": llm_result["error"]}
        
        # 根据模式调用不同API
        if mode == "booking":
            api_result = await call_booking_api(llm_result)
        else:
            api_result = await call_query_api(llm_result)
        print(api_result)
        print(llm_result)
        # 返回完整结果
        return {
            "llm_result": llm_result,
            "api_result": api_result
        }
    except Exception as e:
        return {"error": f"处理请求时出错: {str(e)}"}

# 测试代码
if __name__ == "__main__":
    async def test():
        test_inputs = [
            {"input": "我想预约今天下午三点到四点的宜山厅", "mode": "booking"},
            {"input": "帮我查询明天下午的会议室", "mode": "query"},
            {"input": "预订浦东厅周五上午9点到11点", "mode": "booking"}
        ]
        
        for test in test_inputs:
            print(f"\n测试输入: {test['input']}")
            print(f"请求类型: {test['mode']}")
            
            result = await process_request(test['input'], test['mode'])
            
            print("处理结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("-" * 50)
    
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(test())

