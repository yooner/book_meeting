import streamlit as st
import json
from datetime import date, datetime, timedelta
import pandas as pd
import asyncio
import re
from main import chain_with_summarization
import time
import traceback
import nest_asyncio

# 在顶部应用 nest_asyncio 以允许嵌套事件循环
nest_asyncio.apply()

# 添加一个提取JSON的辅助函数 - 移到文件顶部
def extract_json_from_text(text):
    """
    从文本中提取 JSON 格式的数据
    """
    # 如果输入为空或不是字符串，返回空字典
    if not text or not isinstance(text, str):
        return {}
        
    try:
        # 尝试直接解析整个文本
        return json.loads(text)
    except json.JSONDecodeError:
        # 如果不是有效的 JSON，尝试从文本中提取 JSON 部分
        try:
            # 查找 ```json ... ``` 格式的代码块
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
                return json.loads(json_content)
            
            # 如果找不到明确的 JSON 块，尝试寻找最长的 {} 包围的内容
            potential_jsons = re.findall(r'{.*}', text, re.DOTALL)
            if potential_jsons:
                for potential_json in sorted(potential_jsons, key=len, reverse=True):
                    try:
                        return json.loads(potential_json)
                    except:
                        continue
            
            # 没有找到有效的 JSON
            return {}
        except Exception as e:
            print(f"JSON 解析错误: {str(e)}")
            return {}

# 页面配置
st.set_page_config(
    page_title="会议室预订助手",
    page_icon="🏢",
    layout="wide"
)

# 定义办公地点选项
OFFICE_LOCATIONS = [
    "上海8号楼",
    "上海17号楼",
    "无锡",
    "武汉",
    "电子"
]

# 定义默认的会议时长选项（小时）
MEETING_DURATIONS = [
    "1小时",
    "1.5小时",
    "2小时",
    "3小时",
    "4小时",
    "全天"
]

# 自定义CSS
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: row;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.assistant {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding-left: 1rem;
}
.booking-result, .query-result {
    background-color: #1e2536;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-top: 1rem;
}
.status-success {
    color: #4CAF50;
    font-weight: bold;
}
.status-error {
    color: #F44336;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 定义处理查询的异步函数
async def process_query(query):
    """处理用户查询并返回结果"""
    current_date = st.session_state.current_date
    
    # 收集所有输出
    full_response = ""
    print('===============================================---------------------------------'    )
    print(f"当前日期123: {current_date}")
    print('===============================================---------------------------------')
    async for chunk in chain_with_summarization.astream(
        {"input": query, "current_date": current_date},
        config={"configurable": {"session_id": "streamlit-session"}}
    ):
        if isinstance(chunk, str):
            full_response = chunk
    return full_response

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_date' not in st.session_state:
    st.session_state.current_date = date.today().isoformat()

# 标题
st.title("🏢 会议室预订助手")

# 创建一个容器来显示所有消息历史
messages_container = st.container()

# 显示历史消息
with messages_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # 检查是否包含JSON结构
                try:
                    content = message["content"]
                    data = message.get("data", extract_json_from_text(content))
                    
                    if data and isinstance(data, dict):
                        if "会议室列表" in data:
                            # 查询结果 - 会议室列表格式
                            st.markdown(f"### 会议室查询结果")
                            
                            st.markdown(f'<div class="query-result">'
                                        f'<p>日期: {data.get("日期")}</p>'
                                        f'<p>时间段: {data.get("时间段")}</p>'
                                        f'</div>', unsafe_allow_html=True)
                            
                            # 使用DataFrame显示会议室状态
                            room_list = []
                            for room in data.get("会议室列表", []):
                                if isinstance(room, dict):
                                    room_list.append(room)
                                else:
                                    room_list.append({"会议室": room, "状态": "未知"})
                            
                            if room_list:
                                df = pd.DataFrame(room_list)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("没有找到符合条件的会议室")
                        elif "状态" in data:
                            # 预订结果
                            st.markdown(f"### 会议室预订结果")
                            if data["状态"] == "预订成功":
                                st.success(f"""
                                预订成功！
                                会议ID: {data.get('会议ID')}
                                会议室: {data.get('会议室')}
                                时间: {data.get('时间')}
                                """)
                            else:
                                st.error(f"""
                                预订失败
                                会议室: {data.get('会议室')}
                                时间: {data.get('时间')}
                                原因: {data.get('原因', '未知原因')}
                                """)
                        else:
                            st.markdown(content)
                    else:
                        st.markdown(content)
                except Exception as e:
                    st.markdown(content)

# 侧边栏：设置
with st.sidebar:
    global meeting_date
    st.header("会议室预订")
    
    # 1. 办公地点选择
    selected_location = st.selectbox(
        "办公地点",
        options=OFFICE_LOCATIONS,
        index=0,  # 默认选择第一个选项
        key="office_location"
    )
    
    # 2. 会议时间选择
    col1, col2 = st.columns(2)
    with col1:
        # 日期选择，默认为当前日期
        meeting_date = st.date_input(
            "会议日期",
            value=date.fromisoformat(st.session_state.current_date),
            key="meeting_date"
        )
    with col2:
        # 会议时长选择
        meeting_duration = st.selectbox(
            "会议时长",
            options=MEETING_DURATIONS,
            index=0,  # 默认选择1小时
            key="meeting_duration"
        )
    
    # 开始时间选择（24小时制）
    start_time = st.time_input(
        "开始时间",
        value=datetime.strptime("09:00", "%H:%M").time(),  # 默认上午9点
        key="start_time"
    )
    
    # 3. 会议名称输入
    meeting_name = st.text_input(
        "会议名称",
        value="沟通会议",  # 默认会议名称
        key="meeting_name"
    )
    
    # 添加一个预订按钮
    if st.button("快速预订", type="primary"):
        # 构造预订请求
        end_time = None
        if meeting_duration == "1小时":
            end_time = (datetime.combine(date.today(), start_time) + timedelta(hours=1)).time()
        elif meeting_duration == "1.5小时":
            end_time = (datetime.combine(date.today(), start_time) + timedelta(minutes=90)).time()
        elif meeting_duration == "2小时":
            end_time = (datetime.combine(date.today(), start_time) + timedelta(hours=2)).time()
        elif meeting_duration == "3小时":
            end_time = (datetime.combine(date.today(), start_time) + timedelta(hours=3)).time()
        elif meeting_duration == "4小时":
            end_time = (datetime.combine(date.today(), start_time) + timedelta(hours=4)).time()
        elif meeting_duration == "全天":
            end_time = datetime.strptime("20:00", "%H:%M").time()
        
        # 构造查询字符串
        query = f"在{selected_location}预订{meeting_date.strftime('%Y-%m-%d')} {start_time.strftime('%H:%M')}到{end_time.strftime('%H:%M')}的会议室用于{meeting_name}"
        # 将查询添加到输入框
        st.session_state.user_input = query
    
    st.divider()  # 添加分隔线
    
    # 测试按钮和其他设置
    st.header("其他设置")
    if st.button("清除聊天历史"):
        st.session_state.messages = []
        st.rerun()
    
    # 更新当前日期（用于模型上下文）
    if meeting_date:
        st.session_state.current_date = meeting_date.isoformat()
    
    # 使用说明
    st.header("使用说明")
    st.markdown("""
    ### 预订会议室
    1. 选择办公地点和会议时间
    2. 输入会议名称
    3. 点击"快速预订"或直接输入需求
    
    ### 查询会议室
    例如: "查询贡湖厅下周一的使用情况"
    
    ### 提示
    - 可以指定具体日期和时间
    - 可以询问特定会议室的可用性
    - 会议室名称可以简写，如"乐山厅"
    """)

# 用户输入
if "user_input" in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input  # 清除已使用的输入
else:
    user_input = st.chat_input("输入您的问题，例如：预订乐山厅明天上午9点到11点")

# 处理新的用户输入
if user_input:
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 创建一个新的消息容器
    with messages_container:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 创建助手消息容器
        with st.chat_message("assistant"):
            try:
                # 使用 spinner 显示处理状态
                with st.spinner('正在处理您的请求...'):
                    # 导入所需模块
                    from main import chain_with_summarization
                    import time
                    
                    # 创建会话ID
                    session_id = f"session_{time.time()}"
                    
                    # 处理请求
                    response_dict = chain_with_summarization.invoke(
                        {"input": user_input, "current_date": meeting_date.strftime("%Y-%m-%d")},
                        config={"configurable": {"session_id": session_id}}
                    )
                    
                    response = response_dict.get('output', '')
                    
                    # 解析响应
                    try:
                        json_data = json.loads(response)
                    except json.JSONDecodeError:
                        json_data = extract_json_from_text(response)
                    
                    # 处理响应数据
                    if json_data and isinstance(json_data, dict):
                        if "会议室列表" in json_data:
                            # 显示会议室查询结果
                            st.markdown("### 会议室查询结果")
                            if isinstance(json_data["会议室列表"], list) and json_data["会议室列表"]:
                                df = pd.DataFrame(json_data["会议室列表"])
                                st.dataframe(df, use_container_width=True)
                                
                                # 显示统计信息
                                total_rooms = len(df)
                                booked_rooms = len(df[df["状态"] == "已预订"])
                                available_rooms = total_rooms - booked_rooms
                                st.write(f"共有 {total_rooms} 个会议室，其中 {available_rooms} 个空闲，{booked_rooms} 个已预订。")
                            
                        elif "状态" in json_data:
                            # 显示预订结果
                            if json_data["状态"] == "预订成功":
                                st.success(f"""
                                预订成功！
                                会议ID: {json_data.get('会议ID')}
                                会议室: {json_data.get('会议室')}
                                时间: {json_data.get('时间')}
                                """)
                            else:
                                st.error(f"""
                                预订失败
                                会议室: {json_data.get('会议室')}
                                时间: {json_data.get('时间')}
                                原因: {json_data.get('原因', '未知原因')}
                                """)
                        
                        # 保存到会话历史
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "data": json_data
                        })
                    else:
                        # 处理纯文本响应
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                    
                    # 强制重新渲染
                    st.rerun()
                    
            except Exception as e:
                error_message = f"处理请求时出错: {str(e)}\n\n详细信息: {traceback.format_exc()}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                print(error_message)
                # 强制重新渲染
                st.rerun()
