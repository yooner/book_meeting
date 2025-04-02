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
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory, ConversationBufferWindowMemory,ConversationSummaryMemory

# langchain.debug = True

os.environ["https_proxy"] = "socks5://127.0.0.1:7897"
load_dotenv()


# 定义工具函数
def book_meeting_room(params: str) -> str:
    """
    预订会议室的工具函数
    
    Args:
        params: 包含预订参数的字典
            - 会议室: 会议室名称
            - 日期: 预订日期，格式：YYYY-MM-DD
            - 开始时间: 开始时间，格式：HH:MM
            - 结束时间: 结束时间，格式：HH:MM
    """
    params = json.loads(params)
    会议室 = params.get("会议室")
    日期 = params.get("日期")
    开始时间 = params.get("开始时间")
    结束时间 = params.get("结束时间")
    
    result = {
        "会议室": 会议室,
        "日期": 日期,
        "开始时间": 开始时间,
        "结束时间": 结束时间,
        "结果": "已预订"
    }
    return json.dumps(result, ensure_ascii=False)

def query_meeting_room(params: str) -> str:
    params = json.loads(params)
    """查询会议室状态的工具函数"""
    result = {
        "日期": params.get("日期")  ,
        "时间段": params.get("时间段"),
        "会议室列表": params.get("会议室列表"),
        "结果": [
            {"会议室": "宜山厅", "状态": "空闲"},
            {"会议室": "徐汇厅", "状态": "已预订"},
        ]
    }
    return json.dumps(result, ensure_ascii=False)

def create_llm():
    """创建LLM实例"""
    return ChatOpenAI(
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-chat-v3-0324",
        temperature=0
    )
llm = create_llm()

# 全局变量存储 agent 和 memory
AGENT = None
MEMORY = None

def init_agent():
    """初始化全局 agent 和 memory"""
    global AGENT, MEMORY
    
    if AGENT is None:
        llm = create_llm()
        
        # 创建 memory
        MEMORY = ConversationTokenBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            llm=llm,
            max_token_limit=16000
        )
        
        tools = [
            Tool(
                name="query_meeting_room",
                description="用于查询会议室状态。",
                func=query_meeting_room
            ),
            Tool(
                name="book_meeting_room",
                description="用于预订会议室。",
                func=book_meeting_room
            )
        ]
        
        system_message = SystemMessage(content="""你是智能会议室预订助手。记住用户的选择和偏好。

可用会议室：宜山厅、徐汇厅、浦东厅

工具使用格式：

1. 预订会议室 (book_meeting_room):
{
    "会议室": "宜山厅",
    "日期": "YYYY-MM-DD",
    "开始时间": "HH:MM",
    "结束时间": "HH:MM"
}

2. 查询会议室 (query_meeting_room):
{
    "日期": "YYYY-MM-DD",
    "时间段": "全天" 或 "HH:MM-HH:MM",
    "会议室列表": "所有" 或 ["会议室1", "会议室2"]
}

偏好记忆：
- 记住用户常用的会议室
- 记住用户常用的时间段
- 记住用户表达的喜好""")

        # 创建agent
        AGENT = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=MEMORY,
            handle_parsing_errors=True,
            agent_kwargs={
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
                "system_message": system_message
            }
        )

def agent_node(state: Dict) -> Dict:
    """Agent处理节点"""
    try:
        messages = state.get("messages", [])
        current_date = state.get("current_date", datetime.now().strftime('%Y-%m-%d'))
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 确保有消息并获取最后一条
        if not messages:
            return {
                "messages": messages + [AIMessage(content="没有收到用户消息")],
                "current_date": current_date
            }
            
        last_message = messages[-1].content
        
        # 确保agent已初始化
        if AGENT is None:
            init_agent()
        
        # 准备输入
        agent_input = f"""当前日期是：{current_date}
明天是：{tomorrow_date}

用户请求：{last_message}"""
        
        print(f"\n[DEBUG] 发送给Agent的输入: {agent_input}")
        print(f"\n[DEBUG] 当前记忆内容: {MEMORY.load_memory_variables({})}")
        
        # 调用agent并获取结果
        result = AGENT.invoke({"input": agent_input})
        print(f"\n[DEBUG] Agent返回的结果: {result}")
        
        output = result.get("output", "处理失败，未获得响应")
        
        # 返回更新后的状态
        return {
            "messages": list(messages) + [AIMessage(content=str(output))],
            "current_date": current_date
        }
        
    except Exception as e:
        print(f"\n[DEBUG] Agent处理错误: {str(e)}")
        return {
            "messages": (messages if 'messages' in locals() else []) + 
                       [AIMessage(content=f"处理出错：{str(e)}")],
            "current_date": state.get("current_date", datetime.now().strftime('%Y-%m-%d'))
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
        # "我想预约今天下午三点到四点的宜山厅",
        # "帮我查询明天下午的会议室",
        "预订浦东厅周五上午9点到11点",
        "跟上一次一样定一下周六的会议室",
        # "查一下宜山厅和徐汇厅今天的空闲情况"
    ]
    
    for test_input in test_inputs:
        print(f"\n测试输入: {test_input}")
        result = process_request(test_input)
        print("处理结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("-" * 50)

# 在程序启动时初始化agent
init_agent()

