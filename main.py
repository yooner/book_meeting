import langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from datetime import datetime
import json
import os
import httpx  # 添加httpx导入
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from datetime import date
from langchain_core.runnables import RunnablePassthrough

# langchain.debug = True
os.environ["https_proxy"] = "socks5://127.0.0.1:7897"
from langchain.chat_models import init_chat_model


# llm = init_chat_model(
#     model_provider="ollama",
#     base_url="http://10.17.8.207:11434",
#     model="deepseek-r1:70b",
#     temperature=0
# )

def create_llm():
    """创建LLM实例"""
    return ChatOpenAI(
        openai_api_key="asd",
        openai_api_base="http://10.17.8.207:11434/v1",
        model_name="deepseek-r1:70b",
        temperature=0
    )

def create_summay_llm():
    """创建LLM实例"""
    return ChatOpenAI(
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-chat-v3-0324:free",
        temperature=0
    )
llm = create_llm()
chat_summary = create_summay_llm()


@tool
def book_meeting_room(params: str) -> str:
    """
    预订会议室的工具函数
    
    Args:
        params: 包含预订参数的字典
            - 会议室: 会议室名称
            - 会议名称: 会议名称
            - 日期: 预订日期，格式：YYYY-MM-DD
            - 开始时间: 开始时间，格式：HH:MM
            - 结束时间: 结束时间，格式：HH:MM
    """
    params = json.loads(params)
    会议室 = params.get("会议室")
    会议名称 = params.get("会议名称")
    日期 = params.get("日期")
    开始时间 = params.get("开始时间")
    结束时间 = params.get("结束时间")
    
    # 构造API请求参数
    api_params = {
        "room_name": 会议室,
        "meeting_name": 会议名称,
        "start_datetime": f'{日期} {开始时间}',
        "end_datetime": f'{日期} {结束时间}',
        # "caller_id": "816",  # 默认值
        # "contacter_id": "816",  # 默认值
        # "description": "",  # 可以根据需要添加描述
        # "total_members": 1  # 可以设置参会人数
    }
    print(api_params)
    try:
        # 使用同步方式发送HTTP请求到agent.py的API接口
        with httpx.Client() as client:
            # 设置请求头
            headers = {
                "Content-Type": "multipart/form-data"
            }
            # 发送POST请求，将数据作为表单数据而不是URL参数
            response = client.post(
                "http://localhost:8000/api/book-room",  # 假设agent.py在本地8000端口运行
                params=api_params,  # 使用data参数将数据作为表单数据发送
                # headers=headers,
                timeout=30.0  # 设置超时时间
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                return json.dumps({
                    "状态": "预订成功",
                    "会议ID": result.get("meeting_id"),
                    "会议室": 会议室,
                    "时间": f"{日期} {开始时间}-{结束时间}"
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "状态": "预订失败",
                    "会议室": 会议室,
                    "时间": f"{日期} {开始时间}-{结束时间}",
                    "原因": result.get("message") or "未知错误"
                }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "状态": "预订失败",
            "会议室": 会议室,
            "时间": f"{日期} {开始时间}-{结束时间}",
            "原因": f"请求错误: {str(e)}"
        }, ensure_ascii=False)

@tool
def query_meeting_room(params: str) -> str:
    """
    查询会议室的工具函数
    
    Args:
        params: 包含预订参数的字典
            - 日期: 预订日期，格式：YYYY-MM-DD
            - 开始时间: 开始时间，格式：HH:MM (可选，默认为09:00)
            - 结束时间: 结束时间，格式：HH:MM (可选，默认为20:00)
    """
    params = json.loads(params)
    会议室 = params.get("会议室", "")
    日期 = params.get("日期")
    开始时间 = params.get("开始时间", "09:00")
    结束时间 = params.get("结束时间", "20:00")
    时间段 = f"{开始时间}-{结束时间}"

    # 构造API请求参数
    api_params = {
        "date": 日期,
        "start_time": 开始时间,
        "end_time": 结束时间
    }
    
    try:
        # 使用同步方式发送HTTP请求到agent.py的API接口
        with httpx.Client() as client:
            response = client.get(
                "http://localhost:8000/api/room-availability-simple",
                params=api_params,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            # 从API响应中提取会议室可用情况
            rooms_data = data.get("rooms", {})
            room_status_list = []
            
            # 如果指定了特定会议室，只返回该会议室的信息
            if 会议室:
                if 会议室 in rooms_data:
                    room_info = rooms_data[会议室]
                    # 检查给定时间段内是否有忙碌时间
                    is_busy = len(room_info.get("busy_time", [])) > 0
                    status = "已预订" if is_busy else "空闲"
                    room_status_list.append({"会议室": 会议室, "状态": status})
                else:
                    # 如果找不到指定会议室，可能会议室名称不完全匹配
                    # 尝试部分匹配
                    found = False
                    for room_name in rooms_data:
                        if 会议室 in room_name:
                            room_info = rooms_data[room_name]
                            is_busy = len(room_info.get("busy_time", [])) > 0
                            status = "已预订" if is_busy else "空闲"
                            room_status_list.append({"会议室": room_name, "状态": status})
                            found = True
                    
                    if not found:
                        room_status_list.append({"会议室": 会议室, "状态": "未找到"})
            else:
                # 返回所有会议室的情况
                for room_name, room_info in rooms_data.items():
                    is_busy = len(room_info.get("busy_time", [])) > 0
                    status = "已预订" if is_busy else "空闲"
                    room_status_list.append({"会议室": room_name, "状态": status})
            
            # 构造结果
            result = {
                "日期": 日期,
                "时间段": 时间段,
                "会议室列表": [会议室] if 会议室 else list(rooms_data.keys()),
                "结果": room_status_list
            }
            
            return json.dumps(result, ensure_ascii=False)
            
    except Exception as e:
        # 发生错误时返回错误信息
        error_result = {
            "日期": 日期,
            "时间段": 时间段,
            "会议室列表": [会议室] if 会议室 else [],
            "结果": [],
            "错误": f"查询失败: {str(e)}"
        }
        return json.dumps(error_result, ensure_ascii=False)



# 定义工具列表
tools = [book_meeting_room, query_meeting_room]

# 从 Hub 获取 ReAct 提示模板
prompt = PromptTemplate.from_template('''你是一个会议室预订助手，可以使用以下工具来回答问题：

{tools}

当前对话历史：
{chat_history}
当前日期是：{current_date}


请按照以下格式思考和作答：

Question  
Thought: 你要做什么？应该调用哪个工具？  
Action: 必须执行以下操作之一:[{tool_names}]  
Action Input: 传入工具的参数，字段名和数据类型必须与工具定义完全一致不需要出现# ,如果缺少相关信息可以依据历史来总结用户的喜好提取缺少的参数信息
Observation: 观察工具返回的结果，如果是book_meeting_room 则 结果为被占用 或者 已预定，如果是被占用则列出可用会议室,如果是query_meeting_room 则 结果为会议室使用情况
Thought: 调用上一步工具后
(可选) 其他轮次：重复上述 Thought/Action/Action Input/Observation 的步骤，直到你可以得出最终答案 
Final Answer: 回答问题的最终结果

开始！

Question:{input}
Thought: {agent_scratchpad}
'''
)

# 创建 ReAct 代理
agent = create_react_agent(llm, tools, prompt)

# 创建代理执行器
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
)

# 创建消息历史存储
demo_ephemeral_chat_history = ChatMessageHistory()

# 创建带历史记录的链
chain_with_message_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def summarize_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            (
                "system",
                "将上述聊天消息提炼为一条总结信息，并尽可能包含具体细节。",
            ),
        ]
    )
    summarization_chain = summarization_prompt | chat_summary

    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    demo_ephemeral_chat_history.clear()

    demo_ephemeral_chat_history.add_message(summary_message)

    return True

chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | chain_with_message_history
)
# 测试代码
if __name__ == "__main__":
    test_inputs = [
        # "预订乐山厅下周六上午9点到11点用于沟通",  
        "查询下周一会议室使用情况"
    ]
    
    # 获取当前日期
    
    for test_input in test_inputs:
        print(f"\n测试输入: {test_input}")
        
        # 在调用时传入 current_date
        current_date = date.today().isoformat()
        for chunk in chain_with_summarization.stream(
            {
                "input": test_input,
                "current_date": current_date  # 添加当前日期
            },
            config={"configurable": {"session_id": "1"}}
        ):
            print(chunk)
            print("-" * 50)
