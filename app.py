import streamlit as st
import json
from datetime import date
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

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            # 检查是否包含JSON结构
            try:
                content = message["content"]
                
                # 检查消息是否已经包含预处理的数据
                if "data" in message and isinstance(message["data"], dict):
                    # 直接使用预处理的数据
                    data = message["data"]
                else:
                    # 尝试提取响应中的JSON部分
                    data = extract_json_from_text(content)
                
                if data and isinstance(data, dict):
                    if "会议室列表" in data and isinstance(data.get("会议室列表"), list):
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
                    
                    elif "结果" in data and isinstance(data.get("结果"), list):
                        # 查询结果 - 普通结果列表格式
                        st.markdown(f"### 会议室查询结果")
                        
                        st.markdown(f'<div class="query-result">'
                                    f'<p>日期: {data.get("日期")}</p>'
                                    f'<p>时间段: {data.get("时间段")}</p>'
                                    f'</div>', unsafe_allow_html=True)
                        
                        # 使用DataFrame显示会议室状态
                        result_data = data.get("结果", [])
                        if result_data:
                            df = pd.DataFrame(result_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("没有找到符合条件的会议室")
                    
                    elif "状态" in data and ("预订成功" in data.get("状态") or "预订失败" in data.get("状态")):
                        # 这是预订结果
                        st.markdown(f"### 会议室预订结果")
                        
                        if data.get("状态") == "预订成功":
                            st.markdown(f'<div class="booking-result">'
                                        f'<p><span class="status-success">{data.get("状态")}</span></p>'
                                        f'<p>会议ID: {data.get("会议ID")}</p>'
                                        f'<p>会议室: {data.get("会议室")}</p>'
                                        f'<p>时间: {data.get("时间")}</p>'
                                        f'</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="booking-result">'
                                        f'<p><span class="status-error">{data.get("状态")}</span></p>'
                                        f'<p>会议室: {data.get("会议室")}</p>'
                                        f'<p>时间: {data.get("时间")}</p>'
                                        f'<p>原因: {data.get("原因")}</p>'
                                        f'</div>', unsafe_allow_html=True)
                    else:
                        # 普通文本响应
                        st.markdown(content)
                else:
                    # 普通文本响应
                    st.markdown(content)
            except Exception as e:
                # 记录错误但仍然显示原始内容
                print(f"解析响应出错: {str(e)}")
                st.markdown(content)

# 用户输入
user_input = st.chat_input("输入您的问题，例如：预订乐山厅明天上午9点到11点")

# 处理用户输入
if user_input:
    # 在UI上显示用户输入
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # 显示助手思考中的消息
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("思考中...")
        
        try:
            # 使用完全同步的方式处理请求
            from main import chain_with_summarization
            import time
            
            # 创建自定义会话ID，加入时间戳以避免缓存问题
            session_id = f"session_{time.time()}"
            
            # 同步处理请求，避免异步问题
            def process_query_sync(query):
                max_retries = 3
                retry_count = 0
                retry_delay = 1  # 初始重试延迟（秒）
                
                while retry_count < max_retries:
                    try:
                        # 使用同步调用而不是异步
      
                        response = chain_with_summarization.invoke(
                            {"input": query, "current_date": time.strftime("%Y-%m-%d")},
                            config={"configurable": {"session_id": session_id}}
                        )
                        return response
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"连接错误，尝试重试 ({retry_count}/{max_retries}): {str(e)}")
                            time.sleep(retry_delay)  # 使用同步sleep
                            retry_delay *= 2  # 指数退避
                        else:
                            print(f"重试次数用尽，最终错误: {str(e)}")
                            traceback.print_exc()
                            return f"处理请求时出错: {str(e)}\n\n详细信息: {traceback.format_exc()}"
            
            # 运行同步查询
            response_dict = process_query_sync(user_input)
            response = response_dict.get('output')
            
            # 处理响应
            if isinstance(response, str):
                # 首先尝试将整个response作为JSON解析
                try:
                    json_data = json.loads(response)
                    is_json = True
                except json.JSONDecodeError:
                    is_json = False
                
                if not is_json:
                    # 如果不是纯JSON，尝试提取JSON部分
                    json_data = extract_json_from_text(response)
                
                if json_data and isinstance(json_data, dict):
                    # 成功提取到JSON数据
                    if "会议室列表" in json_data:
                        # 处理会议室查询结果
                        import pandas as pd
                        
                        # 准备并显示DataFrame
                        if isinstance(json_data["会议室列表"], list) and json_data["会议室列表"]:
                            # 确保列表中的每个元素都是字典类型
                            if all(isinstance(item, dict) for item in json_data["会议室列表"]):
                                # 创建DataFrame展示会议室状态
                                results_df = pd.DataFrame(json_data["会议室列表"])
                                
                                # 优化显示 - 可以添加排序、筛选等功能
                                # 按状态排序，把已预订的排在前面
                                results_df = results_df.sort_values(by="状态", ascending=False)
                                
                                # 显示表格
                                st.dataframe(results_df)
                                
                                # 显示会议室可用性摘要
                                total_rooms = len(results_df)
                                booked_rooms = len(results_df[results_df["状态"] == "已预订"])
                                available_rooms = total_rooms - booked_rooms
                                
                                st.write(f"共有 {total_rooms} 个会议室，其中 {available_rooms} 个空闲，{booked_rooms} 个已预订。")
                                
                                # 如果有日期和时间段信息，也显示出来
                                if "日期" in json_data and "时间段" in json_data:
                                    st.write(f"查询日期: {json_data['日期']}, 时间段: {json_data['时间段']}")
                                
                                # 更新占位符内容，不让它一直显示"思考中..."
                                message_placeholder.text(f"查询结果：{json_data['日期']}的会议室使用情况")
                            else:
                                # 处理会议室列表不是字典的情况
                                st.write(f"会议室列表: {', '.join(str(room) for room in json_data['会议室列表'])}")
                                message_placeholder.text("已完成会议室信息处理")
                        elif "结果" in json_data and isinstance(json_data["结果"], list):
                            # 兼容另一种可能的格式
                            import pandas as pd
                            results_df = pd.DataFrame(json_data["结果"])
                            st.dataframe(results_df)
                            message_placeholder.text("已完成会议室信息处理")
                        
                        # 保存结构化数据到聊天历史
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "查询结果如下:", 
                            "data": json_data
                        })
                    else:
                        # 其他类型的JSON响应
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "data": json_data
                        })
                        message_placeholder.text(response)
                else:
                    # 没有提取到有效JSON，直接显示原始响应
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    message_placeholder.text(response)
        except Exception as e:
            error_message = f"处理请求时出错: {str(e)}\n\n详细信息: {traceback.format_exc()}"
            message_placeholder.text(error_message)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_message
            })
            print(error_message)

# 侧边栏：设置
with st.sidebar:
    st.header("设置")
    
    # 测试按钮
    if st.button("清除聊天历史"):
        st.session_state.messages = []
        st.rerun()
    
    # 日期设置
    custom_date = st.date_input("选择日期（用于模型上下文）", value=date.fromisoformat(st.session_state.current_date))
    if custom_date:
        st.session_state.current_date = custom_date.isoformat()
    
    # 使用说明
    st.header("使用说明")
    st.markdown("""
    ### 预订会议室
    例如: "帮我预订乐山厅明天上午9点到11点"
    
    ### 查询会议室
    例如: "查询贡湖厅下周一的使用情况"
    
    ### 提示
    - 可以指定具体日期和时间
    - 可以询问特定会议室的可用性
    - 会议室名称可以简写，如"乐山厅"
    """)
