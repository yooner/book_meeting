import langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
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
import pickle
import time
from pathlib import Path
import re

# langchain.debug = True
os.environ["https_proxy"] = "socks5://127.0.0.1:7897"
from langchain.chat_models import init_chat_model

# 定义保存历史记录的文件路径
CHAT_HISTORY_FILE = "chat_history.txt"
HISTORY_BACKUP_DIR = "history_backups"

# 确保备份目录存在
Path(HISTORY_BACKUP_DIR).mkdir(exist_ok=True)

# LLM管理器类 - 处理LLM的初始化、重连和状态跟踪
class LLMManager:
    def __init__(self):
        self.main_llm = None
        self.summary_llm = None
        self.last_init_time = 0
        self.connection_timeout = 600  # 10分钟后重新初始化LLM
        self.init_attempts = 0
        self.max_init_attempts = 3
    
    def create_main_llm(self):
        """创建主要LLM实例"""
        self.init_attempts += 1
        try:
            self.main_llm = ChatOpenAI(
                openai_api_key="asd",
                openai_api_base="http://10.17.8.207:11434/v1",
                model_name="deepseek-r1:70b",
                temperature=0
            )
            self.last_init_time = time.time()
            self.init_attempts = 0
            print("主LLM初始化成功")
            return self.main_llm
        except Exception as e:
            print(f"主LLM初始化失败 (尝试 {self.init_attempts}/{self.max_init_attempts}): {str(e)}")
            if self.init_attempts >= self.max_init_attempts:
                raise Exception(f"主LLM初始化多次失败: {str(e)}")
            time.sleep(2)  # 短暂延迟后重试
            return self.create_main_llm()
    
    def create_summary_llm(self):
        """创建摘要LLM实例"""
        try:
            self.summary_llm = ChatOpenAI(
                openai_api_key="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",
                openai_api_base="https://openrouter.ai/api/v1",
                model_name="deepseek/deepseek-chat-v3-0324:free",
                temperature=0,
            )
            print("摘要LLM初始化成功")
            return self.summary_llm
        except Exception as e:
            print(f"摘要LLM初始化失败: {str(e)}")
            # 返回一个假的LLM，以便程序能继续运行
            from langchain.llms.fake import FakeListLLM
            return FakeListLLM(responses=["历史对话已总结完毕。"])
    
    def get_main_llm(self, force_refresh=False):
        """获取主LLM，如果超时或强制刷新则重新初始化"""
        current_time = time.time()
        if (self.main_llm is None or 
            force_refresh or 
            current_time - self.last_init_time > self.connection_timeout):
            print(f"主LLM需要重新初始化，已经过去 {current_time - self.last_init_time:.1f} 秒")
            return self.create_main_llm()
        return self.main_llm
    
    def get_summary_llm(self, force_refresh=False):
        """获取摘要LLM，如果未初始化或强制刷新则重新初始化"""
        if self.summary_llm is None or force_refresh:
            return self.create_summary_llm()
        return self.summary_llm

# 创建LLM管理器实例
llm_manager = LLMManager()

# 使用管理器获取LLM实例
llm = llm_manager.get_main_llm()
chat_summary = llm_manager.get_summary_llm()

# 创建消息历史存储
demo_ephemeral_chat_history = ChatMessageHistory()

def message_to_text(message):
    """将消息对象转换为文本格式"""
    if isinstance(message, HumanMessage):
        return f"USER: {message.content}"
    elif isinstance(message, AIMessage):
        return f"ASSISTANT: {message.content}"
    elif isinstance(message, SystemMessage):
        return f"SYSTEM: {message.content}"
    else:
        return f"OTHER: {message.content}"

def text_to_message(text):
    """将文本格式转换回消息对象"""
    if text.startswith("USER: "):
        return HumanMessage(content=text[6:])
    elif text.startswith("ASSISTANT: "):
        return AIMessage(content=text[11:])
    elif text.startswith("SYSTEM: "):
        return SystemMessage(content=text[8:])
    else:
        # 默认作为系统消息处理
        return SystemMessage(content=text)

def save_chat_history_as_text():
    """将聊天历史保存为纯文本格式"""
    try:
        # 获取当前的聊天历史
        messages = demo_ephemeral_chat_history.messages
        
        # 转换为文本格式
        text_lines = [message_to_text(msg) for msg in messages]
        history_text = "\n\n".join(text_lines)
        
        # 保存主文件
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            f.write(history_text)
        
        # 同时创建一个带时间戳的备份
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(HISTORY_BACKUP_DIR, f"chat_history_{timestamp}.txt")
        
        # 每小时最多备份一次，避免过多备份文件
        latest_backup = get_latest_backup()
        if latest_backup:
            latest_time = os.path.getmtime(latest_backup)
            current_time = time.time()
            # 如果上次备份距离现在不到1小时，不创建新备份
            if current_time - latest_time < 3600:  # 3600秒 = 1小时
                return True
        
        # 创建新备份
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(history_text)
        
        print(f"聊天历史已保存到 {CHAT_HISTORY_FILE} 和备份 {backup_file}")
        return True
    except Exception as e:
        print(f"保存聊天历史失败: {str(e)}")
        return False

def get_latest_backup():
    """获取最新的备份文件"""
    backup_files = list(Path(HISTORY_BACKUP_DIR).glob("chat_history_*.txt"))
    if not backup_files:
        return None
    
    # 按修改时间排序，返回最新的
    return str(sorted(backup_files, key=lambda x: os.path.getmtime(x), reverse=True)[0])

def load_chat_history_from_text():
    """从文本文件加载聊天历史"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history_text = f.read()
            
            # 将文本分割为消息
            message_texts = re.split(r'\n\n', history_text)
            messages = [text_to_message(text) for text in message_texts if text.strip()]
            
            # 清除当前历史记录并加载保存的历史记录
            demo_ephemeral_chat_history.clear()
            for message in messages:
                demo_ephemeral_chat_history.add_message(message)
            
            print(f"成功加载了 {len(messages)} 条聊天历史")
            return True
        else:
            print("未找到聊天历史文件，使用空历史记录")
            return False
    except Exception as e:
        print(f"加载聊天历史失败: {str(e)}")
        # 如果主文件加载失败，尝试从最新的备份中恢复
        try:
            latest_backup = get_latest_backup()
            if latest_backup:
                print(f"尝试从备份文件恢复: {latest_backup}")
                with open(latest_backup, 'r', encoding='utf-8') as f:
                    history_text = f.read()
                
                # 将文本分割为消息
                message_texts = re.split(r'\n\n', history_text)
                messages = [text_to_message(text) for text in message_texts if text.strip()]
                
                demo_ephemeral_chat_history.clear()
                for message in messages:
                    demo_ephemeral_chat_history.add_message(message)
                
                print(f"成功从备份恢复了 {len(messages)} 条聊天历史")
                return True
        except Exception as backup_error:
            print(f"从备份恢复也失败了: {str(backup_error)}")
        
        return False
# 
# 修改为使用文本格式的保存/加载函数
save_chat_history = save_chat_history_as_text
load_chat_history = load_chat_history_from_text

# 在程序启动时自动加载历史记录
load_chat_history()

# Don't try to summarize right after loading
# This will prevent the connection error at startup
print("Chat history loaded - skipping initial summarization")

def count_tokens_approx(text):
    """近似计算文本中的token数量，汉字算2个token，英文单词或标点算1个"""
    # 汉字数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 非汉字字符数量
    other_chars = len(re.sub(r'[\u4e00-\u9fff]', '', text))
    # 汉字算2个token，其他字符算1个
    return chinese_chars * 2 + other_chars

# Add this function to check if we need to summarize based on chat history length
def should_summarize(messages, max_tokens=4000):
    """Determine if the chat history needs summarization based on length"""
    # 如果消息少于5条，直接返回不需要总结
    if len(messages) < 5:
        return False
    
    # 计算所有消息内容的大致token数
    total_content = "".join([msg.content for msg in messages])
    token_count = count_tokens_approx(total_content)
    
    # 如果token数超过阈值，需要总结
    return token_count > max_tokens

def generate_summary_message(messages):
    """生成一条总结消息，替代原始的多条消息"""
    try:
        # 重新获取summary LLM，确保连接是活跃的
        summary_llm = llm_manager.get_summary_llm()
        
        # 提取对话中的关键信息进行总结
        summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", """总结以下对话历史，提取关键信息：
1. 已预订的会议室和时间
2. 用户的会议室偏好
3. 常用的会议室和时间段
4. 之前的会议预订情况
请将总结控制在300字以内，只保留对未来对话有用的信息。
"""),
            ("placeholder", "\n".join([f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}" for m in messages])),
        ])
        
        # 使用summary LLM进行总结
        summary_result = summarization_prompt | summary_llm
        summary_text = summary_result.invoke({})
        
        # 创建一个系统消息来保存总结
        return SystemMessage(content=f"历史对话总结：{summary_text}")
    except Exception as e:
        print(f"生成总结失败: {str(e)}")
        # 创建一个简单的总结消息作为备选
        return SystemMessage(content="历史对话中包含了关于会议室预订的信息。")

# Then modify the summarize_messages function
def summarize_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) == 0:
        return False
    
    # 检查是否需要总结
    if not should_summarize(stored_messages):
        print("Skipping summarization - chat history is short")
        return False
        
    try:
        # 生成总结消息
        summary_message = generate_summary_message(stored_messages)
        
        # 保留最近的3条消息，加上总结消息
        recent_messages = stored_messages[-3:] if len(stored_messages) > 3 else stored_messages
        
        # 清除历史并添加总结和最近消息
        demo_ephemeral_chat_history.clear()
        demo_ephemeral_chat_history.add_message(summary_message)
        for message in recent_messages:
            demo_ephemeral_chat_history.add_message(message)
        
        # 立即保存总结后的历史记录
        save_chat_history()
        print(f"历史记录已总结和精简，从 {len(stored_messages)} 条减少到 {1 + len(recent_messages)} 条")
        return True
    except Exception as e:
        print(f"总结失败，继续使用完整历史: {str(e)}")
        return False

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

# 定义一个解析和打印函数
def parse_and_print_json(message):
    """解析 AIMessage 中的 JSON 并打印结构化信息"""
    print("\n----- 提取的结构化信息 -----")
    print(message)
    return {"input": "预定下周六宜山厅6点到八点用于测试"}
    return message
    # 检查输入类型
    if hasattr(message, "content"):
        content = message.content
        print(f"原始消息内容: {content[:150]}..." if len(content) > 150 else content)
    else:
        content = str(message)
        print(f"非标准消息类型: {type(message)}")
    
    # 尝试解析 JSON
    try:
        import json
        import re
        
        # 尝试找到 JSON 部分 - 处理可能的多余文本
        json_match = re.search(r'({.*})', content)
        json_str = json_match.group(1) if json_match else content
        
        # 解析 JSON
        data = json.loads(json_str)
        
        # 打印解析后的结构化数据
        print("\n解析后的结构化数据:")
        print(f"- 意图 (Intent): {data.get('intent', '未提取')}")
        print(f"- 会议室 (Room): {data.get('room_name', '未提取')}")
        print(f"- 日期 (Date): {data.get('date', '未提取')}")
        print(f"- 开始时间: {data.get('start_time', '未提取')}")
        print(f"- 结束时间: {data.get('end_time', '未提取')}")
        print(f"- 会议名称: {data.get('meeting_name', '未提取')}")
        
        # 返回解析后的字典，方便后续处理
        return data
    except Exception as e:
        print(f"JSON 解析失败: {str(e)}")
        print("无法解析为结构化数据")
        return {"error": "解析失败", "raw_content": content}
    finally:
        print("--------------------------\n")

# 创建代理和链的函数，支持重新初始化
def create_agent_and_chains():
    """创建或重新创建代理和执行链"""
    # 强制获取全新的LLM实例
    llm_manager.main_llm = None  # 先置空确保重新创建
    current_llm = llm_manager.create_main_llm()
    
    prompt = PromptTemplate.from_template('''你是一个会议室预订助手，可以使用以下工具来回答问题：
    {tools}

    当前日期是：{current_date}
    请按照以下格式思考和作答：

    Question  
    Thought: 你要做什么？应该调用哪个工具？  
    Action: 必须执行以下操作之一:[{tool_names}]  
    Action Input: 传入工具的参数，
    Observation: 观察工具返回的结果，如果是book_meeting_room 则 结果为被占用 或者 已预定，如果是被占用则列出可用会议室,如果是query_meeting_room 则 结果为会议室使用情况
    Thought: 调用上一步工具后
    (可选) 其他轮次：重复上述 Thought/Action/Action Input/Observation 的步骤，直到你可以得出最终答案 
    Final Answer: 回答问题的最终结果,如果最终查询会议室则以json的形式返回会议室的使用情况

    开始！

    Question:{input}
    Thought: {agent_scratchpad}
    ''')

    # 创建用户输入规范化预处理链 (保持定义，暂不整合)
    input_standardization_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个会议室预订系统的预处理助手。你的任务是将用户的非正式输入转换为更结构化的查询格式。
        请根据输入提取以下信息：
        1. 意图：是查询会议室还是预订会议室
        2. 会议室名称（如果有提及）
        3. 日期：明确的日期或相对日期（今天、明天、后天、下周等）
        4. 时间段：开始和结束时间

        当前日期: {current_date}
        历史对话内容: # <- 注意：这个提取链的 prompt 仍然可以使用 history
        {chat_history}

        请将用户输入规范化为清晰的查询语句。不要添加未提及的信息，但可以基于历史对话补充缺失的关键信息。
        输出格式示例：
        - 查询：请查询[日期][时间段][会议室名称]的可用情况
        - 预订：请预订[日期][时间段][会议室名称]用于[会议名称]
        """
        ),
        ("human", "{input}")
    ])
    standardization_chain = input_standardization_prompt | current_llm | parse_and_print_json

    # 创建 ReAct 代理 (使用修改后的 prompt)
    agent = create_react_agent(current_llm, tools, prompt)

    # 创建代理执行器
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
    )

    # 创建带历史记录的链 (保持不变，Agent 仍然能通过它访问历史)
    chain_with_message_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: demo_ephemeral_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history" # 这个 key 仍然告诉包装器去管理历史
    )

    # 保持原有的主处理链结构
    chain_with_summarization = (
        RunnablePassthrough.assign(messages_summarized=summarize_messages)
        | RunnablePassthrough.assign(
            structured_data=lambda x: standardization_chain.invoke({
                "input": x["input"],
                "current_date": x["current_date"],
                "chat_history": demo_ephemeral_chat_history.messages
            })
        )
        | chain_with_message_history
    )
    
    return chain_with_summarization

# 初始创建代理和链
chain_with_summarization = create_agent_and_chains()

# 修改 LLMReinitChain 类，确保每次都重新初始化
class LLMReinitChain:
    """每次调用都强制重新初始化LLM的包装类"""
    
    def __init__(self, create_chain_func):
        self.create_chain_func = create_chain_func
    
    def get_chain(self):
        """始终重新初始化并获取最新的链"""
        print("强制重新初始化LLM和链")
        chain = self.create_chain_func()
        return chain
    
    def invoke(self, input_data, **kwargs):
        """每次调用强制重新创建链"""
        # 每次调用都重新创建链和LLM
        chain = self.get_chain()
        try:
            result = chain.invoke(input_data, **kwargs)
            save_chat_history()
            return result
        except Exception as e:
            print(f"调用出错: {str(e)}")
            save_chat_history()
            raise
    
    async def ainvoke(self, input_data, **kwargs):
        """异步调用强制重新创建链"""
        # 每次调用都重新创建链和LLM
        chain = self.get_chain()
        try:
            result = await chain.ainvoke(input_data, **kwargs)
            save_chat_history()
            return result
        except Exception as e:
            print(f"异步调用出错: {str(e)}")
            save_chat_history()
            raise
    
    def stream(self, input_data, **kwargs):
        """流式调用强制重新创建链"""
        # 每次调用都重新创建链和LLM
        chain = self.get_chain()
        try:
            for chunk in chain.stream(input_data, **kwargs):
                yield chunk
            save_chat_history()
        except Exception as e:
            print(f"流式调用出错: {str(e)}")
            save_chat_history()
            raise
    
    async def astream(self, input_data, **kwargs):
        """异步流式调用强制重新创建链"""
        # 每次调用都重新创建链和LLM
        chain = self.get_chain()
        try:
            async for chunk in chain.astream(input_data, **kwargs):
                yield chunk
            save_chat_history()
        except Exception as e:
            print(f"异步流式调用出错: {str(e)}")
            save_chat_history()
            raise

# 使用新的包装类替换原来的链
chain_with_summarization = LLMReinitChain(create_agent_and_chains)

# 测试代码
if __name__ == "__main__":
    test_inputs = [
        # "后天定跟上次一样的会议室"
        "周六定乐山厅14点到16点"
    ]
    
    for test_input in test_inputs:
        print(f"\n测试输入: {test_input}")
        
        # 在调用时传入 current_date
        current_date = date.today().isoformat()
        try:
            for chunk in chain_with_summarization.stream(
                {
                    "input": test_input,
                    "current_date": current_date
                },
                config={"configurable": {"session_id": "1"}}
            ):
                print(chunk)
                print("-" * 50)
        except Exception as e:
            print(f"处理请求时出错: {str(e)}")
            print("尝试重新连接大模型服务后再次运行...")
