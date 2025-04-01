import langchain
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Any, Annotated, TypedDict, Sequence, Union
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
from langgraph.graph import Graph, END
from langchain_core.tools import Tool

# langchain.debug = True

os.environ["https_proxy"] = "socks5://127.0.0.1:7897"
load_dotenv()

# 定义工具函数
def book_meeting_room(会议室: str, 日期: str, 开始时间: str, 结束时间: str) -> str:
    """预订会议室的工具函数"""
    result = {
        "会议室": 会议室,
        "日期": 日期,
        "开始时间": 开始时间,
        "结束时间": 结束时间,
        "状态": "已预订"
    }
    return json.dumps(result, ensure_ascii=False)

def query_meeting_room(日期: str, 时间段: str, 会议室列表: str) -> str:
    """查询会议室状态的工具函数"""
    result = {
        "日期": 日期,
        "时间段": 时间段,
        "会议室列表": 会议室列表,
        "结果": [
            {"会议室": "宜山厅", "状态": "空闲"},
            {"会议室": "徐汇厅", "状态": "已预订"},
            {"会议室": "浦东厅", "状态": "空闲"}
        ]
    }
    return json.dumps(result, ensure_ascii=False)

def create_llm():
    """创建LLM实例"""
    return ChatOpenAI(
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-r1:free",
        temperature=0
    )

def create_agent():
    """创建Agent"""
    llm = create_llm()
    
    tools = [
        Tool(
            name="book_meeting_room",
            description="""用于预订会议室。
参数:
- 会议室: 会议室名称（宜山厅、徐汇厅、浦东厅）
- 日期: 预订日期，格式：YYYY-MM-DD
- 开始时间: 开始时间，格式：HH:MM
- 结束时间: 结束时间，格式：HH:MM""",
            func=book_meeting_room
        ),
        Tool(
            name="query_meeting_room",
            description="""用于查询会议室状态。
参数:
- 日期: 查询日期，格式：YYYY-MM-DD
- 时间段: 可以是"全天"或者"HH:MM-HH:MM"格式的时间段
- 会议室列表: 可以是"所有"或者用逗号分隔的会议室列表，如"宜山厅,徐汇厅" """,
            func=query_meeting_room
        )
    ]
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

def agent_node(state: Dict) -> Dict:
    """Agent处理节点"""
    messages = state.get("messages", [])
    current_date = state.get("current_date", datetime.now().strftime('%Y-%m-%d'))
    
    try:
        agent = create_agent()
        last_message = messages[-1].content if messages else ""
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        enhanced_input = f"""当前日期是：{current_date}
明天是：{tomorrow_date}

用户请求：{last_message}"""
        
        result = agent.invoke({"input": enhanced_input})
        output = result.get("output", "处理失败，未获得响应")
        
        return {
            "messages": list(messages) + [AIMessage(content=str(output))],
            "current_date": current_date
        }
        
    except Exception as e:
        print(f"Agent处理错误: {str(e)}")
        return {
            "messages": messages + [AIMessage(content=f"处理出错：{str(e)}")],
            "current_date": current_date
        }

def create_graph() -> Graph:
    """创建工作流图"""
    workflow = Graph()
    
    workflow.add_node("agent", agent_node)
    
    # 设置入口节点
    workflow.set_entry_point("agent")
    
    # 使用 END 作为终止节点
    workflow.add_edge("agent", END)
    
    return workflow.compile()

def process_request(user_input: str) -> Dict:
    """处理用户请求"""
    try:
        # 创建工作流
        workflow = create_graph()
        # 准备初始状态
        current_date = datetime.now().strftime('%Y-%m-%d')
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "current_date": current_date
        }
        # 运行工作流
        result = workflow.invoke(initial_state)
        
        # 从结果中提取最后一条消息
        final_messages = result["messages"]
        final_message = final_messages[-1].content if final_messages else "处理失败，未获得响应"
        
        return {"result": final_message}
        
    except Exception as e:
        print(f"处理请求错误: {str(e)}")
        return {"error": f"处理请求时出错: {str(e)}"}

# 测试代码
if __name__ == "__main__":
    test_inputs = [
        "我想预约今天下午三点到四点的宜山厅",
        "帮我查询明天下午的会议室",
        "预订浦东厅周五上午9点到11点",
        "查一下宜山厅和徐汇厅今天的空闲情况"
    ]
    
    for test_input in test_inputs:
        print(f"\n测试输入: {test_input}")
        result = process_request(test_input)
        print("处理结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("-" * 50)

