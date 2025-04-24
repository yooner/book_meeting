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
                openai_api_base="http://10.127.21.3:1025/v1",
                model_name="DeepSeek-R1-bf16-w8a8",
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
            # self.summary_llm = ChatOpenAI(
            #     openai_api_key="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",
            #     openai_api_base="https://openrouter.ai/api/v1",
            #     model_name="deepseek/deepseek-chat-v3-0324:free",
            #     temperature=0,
            # )
            self.summary_llm = ChatOpenAI(
                openai_api_key="asd",
                openai_api_base="http://10.127.21.3:1025/v1",
                model_name="DeepSeek-R1-bf16-w8a8",
                temperature=0
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
    """(优化后) 将消息对象转换为文本格式，过滤掉解析过程"""
    if isinstance(message, HumanMessage):
        return f"USER: {message.content}"
    elif isinstance(message, AIMessage):
        content = message.content
        
        # 过滤掉解析信息
        if "解析：" in content:
            # 尝试找到解析部分之后的实际响应
            parts = content.split("A:", 1)
            if len(parts) > 1:
                content = parts[1].strip()
            else:
                # 如果没有明确的分隔符，则尝试移除所有解析信息段落
                lines = content.split('\n')
                filtered_lines = []
                skip_section = False
                
                for line in lines:
                    if line.strip().startswith("解析：") or "解析" in line.strip():
                        skip_section = True
                    elif skip_section and not line.strip():  # 空行结束跳过
                        skip_section = False
                    elif not skip_section:
                        filtered_lines.append(line)
                
                content = '\n'.join(filtered_lines).strip()
        
        return f"ASSISTANT: {content}"
    elif isinstance(message, SystemMessage):
        return f"SYSTEM: {message.content}"
    else:
        # 优雅地处理其他可能的消息类型
        return f"OTHER: {getattr(message, 'content', str(message))}"

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

        # 1. 手动将消息列表格式化为字符串
        formatted_history = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant' if isinstance(m, AIMessage) else 'System'}: {m.content}"
            for m in messages
        ])

        # 2. 修改提示模板，使用普通字符串占位符
        summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", """总结以下对话历史，提取关键信息：
1. 已预订的会议室和时间
2. 用户的会议室偏好
3. 常用的会议室和时间段
4. 之前的会议预订情况
请将总结控制在300字以内，只保留对未来对话有用的信息。

对话历史如下：
{formatted_history}
"""),
        ])

        # 使用summary LLM进行总结
        summary_chain = summarization_prompt | summary_llm

        # 3. 调用 invoke 时传递格式化后的字符串
        # 注意：键名需要匹配模板中的占位符 {formatted_history}
        summary_response = summary_chain.invoke({"formatted_history": formatted_history})

        # 假设返回的是 AIMessage 或类似对象
        summary_text = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
        # --- 结束修改 ---

        # 创建一个系统消息来保存总结
        return SystemMessage(content=f"历史对话总结：{summary_text}")
    except Exception as e:
        # 打印更详细的错误信息，如果可能
        import traceback
        print(f"生成总结失败: {str(e)}")
        # traceback.print_exc() # 取消注释以获取完整堆栈跟踪

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
    print(f"预订请求参数: {api_params}")
    
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
                # 预订失败，直接调用query_meeting_room查询同一时间段可用会议室
                print(f"预订失败，原因: {result.get('message')}。正在查询同一时间段的可用会议室...")
                
                # 构造查询参数
                query_params = {
                    "日期": 日期,
                    "开始时间": 开始时间,
                    "结束时间": 结束时间
                }
                
                # 调用query_meeting_room函数获取所有会议室状态
                query_result_str = query_meeting_room(json.dumps(query_params))
                query_result = json.loads(query_result_str)
                
                # 提取可用会议室
                available_rooms = []
                if "结果" in query_result:
                    for room_status in query_result["结果"]:
                        if room_status.get("状态") == "空闲":
                            available_rooms.append(room_status.get("会议室"))
                
                # 构造包含预订失败信息和可用会议室的响应
                response_data = {
                    "状态": "预订失败",
                    "会议室": 会议室,
                    "时间": f"{日期} {开始时间}-{结束时间}",
                    "原因": result.get("message") or "未知错误",
                    "可用会议室": available_rooms,
                    "详细查询结果": query_result
                }
                
                return json.dumps(response_data, ensure_ascii=False)
    except Exception as e:
        # 发生异常时也尝试查询可用会议室
        print(f"预订请求出错: {str(e)}。正在查询同一时间段的可用会议室...")
        
        try:
            # 构造查询参数
            query_params = {
                "日期": 日期,
                "开始时间": 开始时间,
                "结束时间": 结束时间
            }
            
            # 调用query_meeting_room函数获取所有会议室状态
            query_result_str = query_meeting_room(json.dumps(query_params))
            query_result = json.loads(query_result_str)
            
            # 提取可用会议室
            available_rooms = []
            if "结果" in query_result:
                for room_status in query_result["结果"]:
                    if room_status.get("状态") == "空闲":
                        available_rooms.append(room_status.get("会议室"))
            
            response_data = {
                "状态": "预订失败",
                "会议室": 会议室,
                "时间": f"{日期} {开始时间}-{结束时间}",
                "原因": f"请求错误: {str(e)}",
                "可用会议室": available_rooms,
                "详细查询结果": query_result
            }
        except Exception as query_error:
            # 如果查询也失败了，返回简单的错误信息
            print(f"查询可用会议室也失败了: {str(query_error)}")
            response_data = {
                "状态": "预订失败",
                "会议室": 会议室,
                "时间": f"{日期} {开始时间}-{结束时间}",
                "原因": f"请求错误: {str(e)}",
                "可用会议室": ["查询可用会议室失败"],
                "查询错误": str(query_error)
            }
        
        return json.dumps(response_data, ensure_ascii=False)

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
                    result = rooms_data[会议室]
                    # 检查给定时间段内是否有忙碌时间
            else:
                result = rooms_data
            
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
    """
    (修改后) 打印来自标准化LLM的消息内容，
    并提取 '操作：' 或 '预订：' 或 '查询：' 之后的语句作为返回值。
    """
    print("\n----- 标准化LLM输出 -----")
    full_content = "" # Default value

    if hasattr(message, "content"):
        full_content = message.content
        print(f"LLM 完整输出: {full_content}")
    else:
        # 处理意外的输入类型
        full_content = str(message)
        print(f"收到非标准消息类型，内容: {full_content}")

    extracted_query = "" # Initialize extracted_query
    match = re.search(r'[- ]*(操作|预订|查询)：\s*(.*)', full_content, re.DOTALL | re.MULTILINE)

    if match:
        extracted_query = match.group(2).strip()
        print(f"提取到的查询语句: {extracted_query}")
    else:
        lines = full_content.split('\n')
        non_think_lines = [line for line in lines if not line.strip().startswith('<think>') and not line.strip().endswith('</think>')]
        if non_think_lines:
    
             last_meaningful_line = non_think_lines[-1].strip()
             if ':' in last_meaningful_line:
                 extracted_query = last_meaningful_line.split(':')[-1].strip()
             else:
                 extracted_query = last_meaningful_line
             print(f"未找到明确前缀，回退提取: {extracted_query}")
        else:
             # If all else fails, return the original content minus think blocks
             extracted_query = "\n".join(non_think_lines).strip()
             print(f"无法提取特定查询，返回处理后的内容: {extracted_query}")

    print("--------------------------\n")
    # 返回提取到的查询语句字符串
    return extracted_query

def try_extract_date_time(x):
    """从用户输入中提取日期和时间信息"""
    try:
        # 创建并运行推断链
        date_inference_prompt = ChatPromptTemplate.from_messages([
            ("system", """从以下用户输入中提取日期信息，转换为实际日期。
    
当前日期: {current_date}
用户输入: {user_input}

以JSON格式返回，包含 "日期"。""")
        ])
        
        # 获取当前LLM实例
        inference_llm = llm_manager.get_main_llm()
        
        # 创建并运行推断链
        inference_chain = date_inference_prompt | inference_llm
        
        # 调用推断链
        result = inference_chain.invoke({
            "current_date": x["current_date"],
            "user_input": x["input"]
        })
        
        # 尝试解析返回的JSON
        import json
        import re
        
        # 尝试提取JSON部分
        json_match = re.search(r'({.*?})', result.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            inference_result = json.loads(json_str)
            return {
                "日期": inference_result.get("日期", x["current_date"]),
            }
        else:
            print(f"无法从LLM响应中提取JSON: {result.content[:100]}...")
            
    except Exception as e:
        print(f"日期推断失败: {str(e)}")
    

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


    # 创建用户输入规范化预处理链 
    input_standardization_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个会议室预订系统的智能预处理助手。你的任务是将用户的非正式输入转换为结构化的预订指令。

        请根据输入和会议室状态信息提取以下内容：
        1. 日期：从用户提及的日期或从历史对话推断
        2. 时间段：开始和结束时间
        3. 会议室名称：根据用户提及或会议室可用状态选择合适的会议室
        4. 会议名称：从用户输入或历史推断

        当前日期: {current_date}
        历史对话内容: 
        {chat_history}
        对应日期的会议室使用情况:
        {room_status}

        重要提示：
        - 始终选择一个当前可用（状态为"空闲"）的会议室。如果用户指定的会议室已被占用，请从可用会议室中选择一个替代。
        - 历史记录反映用户的偏好（如常用会议室、时间段等），可用于补充用户未明确指定的信息。
        - 当用户使用模糊表达（如"明天下午"）时，使用合理的默认值（如"14:00-16:00"）。
        - 如果用户没有提供会议名称，从历史中推断。如果历史没有则默认是沟通
        - 如果指定时间段内所有会议室都已被占用，直接返回没有满足条件的会议室

        你必须输出以下格式：
        - 操作：预订[日期][时间段][会议室名称]用于[会议名称]
        
        注意：不要生成查询操作，直接根据提供的会议室使用情况生成预订指令。你的回答应该只包含上述标准格式的指令尤其是操作:，不要添加额外的解释或注释。
        """
        ),
        ("human", "{input}")
    ])
    # The chain now ends with the modified parse_and_print_json which returns a string
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

    # *** 修正链条定义，确保先从用户输入中推断日期和时间 ***
    chain_with_summarization = (
        RunnablePassthrough.assign(messages_summarized=summarize_messages)
        | RunnablePassthrough.assign(
            # 预先推断用户输入中的日期和时间
            date_inference=lambda x: {
                "user_input": x["input"],
                "current_date": x["current_date"]
            }
        )
        | RunnablePassthrough.assign(
            # 使用推断llm从用户输入中提取日期和时间信息
            query_date_time=lambda x: (
                try_extract_date_time(x)
            )
        )
        | RunnablePassthrough.assign(
            # 使用推断出的日期时间查询会议室使用情况
            room_status=lambda x: query_meeting_room.invoke(json.dumps(x["query_date_time"]))
        )
        | RunnablePassthrough.assign(
            # 将推断的日期时间和查询到的会议室状态传递给标准化处理链
            standardized_input=lambda x: standardization_chain.invoke({
                "input": x["input"],
                "current_date": x["current_date"],
                "chat_history": demo_ephemeral_chat_history.messages,
                "room_status": x["room_status"],  # 传递查询到的会议室状态
            })
        )
        # | RunnablePassthrough.assign(
        #     # 从标准化输入中提取查询部分，传递给实际处理链
        #     input=lambda x: x["standardized_input"]["query"] if isinstance(x["standardized_input"], dict) else x["standardized_input"]
        # )
        # # 最后将标准化后的输入传递给实际的处理链
        # | chain_with_message_history
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
    
    def _update_last_ai_message_content(self, result):
        """辅助函数：用 result['output'] 更新历史记录中最后一条 AI 消息"""
        try:
            if isinstance(result, dict) and 'output' in result and demo_ephemeral_chat_history.messages:
                last_message = demo_ephemeral_chat_history.messages[-1]
                if isinstance(last_message, AIMessage):
                    print(f"准备更新历史记录中的最后一条 AI 消息。原始内容长度: {len(last_message.content)}, 新内容 ('output'): {result['output'][:100]}...")
                    last_message.content = result['output'] # 直接替换内容
                    print("最后一条 AI 消息内容已更新。")
                else:
                     print("历史记录最后一条不是 AI 消息，不更新。")
            elif not demo_ephemeral_chat_history.messages:
                 print("历史记录为空，不更新。")
            else:
                print(f"结果格式不符合预期或缺少 'output' 键，不更新历史记录。结果类型: {type(result)}")

        except Exception as e:
            print(f"更新最后一条 AI 消息时出错: {e}")

    def invoke(self, input_data, **kwargs):
        """每次调用强制重新创建链"""
        chain = self.get_chain()
        try:
            result = chain.invoke(input_data, **kwargs)
            # --- 新增：更新历史记录 ---
            self._update_last_ai_message_content(result)
            # ----------------------
            save_chat_history() # 现在保存的是更新后的历史
            return result
        except Exception as e:
            print(f"调用出错: {str(e)}")
            save_chat_history() # 出错时也尝试保存（可能未更新）
            raise
    
    async def ainvoke(self, input_data, **kwargs):
        """异步调用强制重新创建链"""
        chain = self.get_chain()
        try:
            result = await chain.ainvoke(input_data, **kwargs)
            # --- 新增：更新历史记录 ---
            self._update_last_ai_message_content(result)
            # ----------------------
            save_chat_history() # 现在保存的是更新后的历史
            return result
        except Exception as e:
            print(f"异步调用出错: {str(e)}")
            save_chat_history() # 出错时也尝试保存（可能未更新）
            raise
    
    def stream(self, input_data, **kwargs):
        """流式调用强制重新创建链"""
        chain = self.get_chain()
        final_result_for_history = None # 用于存储最终结果
        try:
            # 流式输出块
            for chunk in chain.stream(input_data, **kwargs):
                # 检查块是否是最终结果字典
                if isinstance(chunk, dict) and 'output' in chunk:
                    final_result_for_history = chunk
                yield chunk # 将块传递给调用者

            # --- 流结束后：更新历史记录 ---
            if final_result_for_history:
                 self._update_last_ai_message_content(final_result_for_history)
            else:
                 print("流式调用结束，但未捕获到包含'output'的最终结果字典，无法更新历史记录。")
            # -------------------------
            save_chat_history() # 保存历史
        except Exception as e:
            print(f"流式调用出错: {str(e)}")
            save_chat_history() # 出错时也尝试保存
            raise
    
    async def astream(self, input_data, **kwargs):
        """异步流式调用强制重新创建链"""
        chain = self.get_chain()
        final_result_for_history = None # 用于存储最终结果
        try:
            # 异步流式输出块
            async for chunk in chain.astream(input_data, **kwargs):
                 # 检查块是否是最终结果字典
                if isinstance(chunk, dict) and 'output' in chunk:
                    final_result_for_history = chunk
                yield chunk # 将块传递给调用者

             # --- 流结束后：更新历史记录 ---
            if final_result_for_history:
                 self._update_last_ai_message_content(final_result_for_history)
            else:
                 print("异步流式调用结束，但未捕获到包含'output'的最终结果字典，无法更新历史记录。")
            # -------------------------
            save_chat_history() # 保存历史
        except Exception as e:
            print(f"异步流式调用出错: {str(e)}")
            save_chat_history() # 出错时也尝试保存
            raise

# 使用新的包装类替换原来的链
chain_with_summarization = LLMReinitChain(create_agent_and_chains)

# 测试代码
if __name__ == "__main__":
    test_inputs = [
        # "后天定跟上次一样的会议室"
        "后天定一个会议室"
        # "查询明天宜山厅使用情况"
    ]
    
    for test_input in test_inputs:
        print(f"\n测试输入: {test_input}")
        
        # 在调用时传入 current_date
        current_date = date.today().isoformat()
        try:
            for chunk in chain_with_summarization.stream(
                {
                    "input": test_input,
                    "current_date": current_date,
                },
                config={"configurable": {"session_id": "1"}}
            ):
                print(chunk)
                print("-" * 50)
        except Exception as e:
            print(f"处理请求时出错: {str(e)}")
            print("尝试重新连接大模型服务后再次运行...")
