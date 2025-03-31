import streamlit as st
from main import create_chains, process_llm_response
import asyncio
from main import process_request
import json
from datetime import datetime


def setup_page():
    """设置页面"""
    st.set_page_config(page_title="会议室预订助手", page_icon="🗓️")
    st.title("会议室预订助手")
    st.markdown("""
    👋 你好！我是智能会议室预订助手，可以帮你：
    - 预订会议室
    - 查询会议室可用状态
    
    请直接输入您的需求。
    """)

@st.cache_resource
def get_chains():
    """获取并缓存LLM处理链"""
    return create_chains()

async def handle_user_input(user_input: str, mode: str):
    """异步处理用户输入"""
    return await process_request(user_input, mode)

def main():
    setup_page()
    
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是会议室预订助手，有什么可以帮到你的吗？"}
        ]
    
    # 切换模式按钮（预订/查询）
    mode = st.radio("选择模式:", ["预订会议室", "查询会议室"], horizontal=True)
    current_mode = "booking" if mode == "预订会议室" else "query"
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 用户输入
    if user_input := st.chat_input("请输入您的需求..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 处理用户请求
        with st.spinner("思考中..."):
            # 使用 asyncio 运行异步函数
            result = asyncio.run(handle_user_input(user_input, current_mode))
            
            if "error" in result:
                formatted_response = f"⚠️ {result['error']}"
            else:
                formatted_response = format_response(result)
            
            # 添加助手回复到历史
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})
            
            # 显示助手回复
            with st.chat_message("assistant"):
                st.markdown(formatted_response)

def format_response(data):
    """格式化响应数据"""
    # 处理错误情况
    if "error" in data:
        return f"⚠️ {data['error']}"
        
    # 获取LLM解析结果和API调用结果
    llm_result = data.get("llm_result", {})
    api_result = data.get("api_result", {})
    
    # 预订结果
    if "会议室" in llm_result:  # 预订响应
        if api_result.get("成功", False):
            return f"✅ 预订成功！\n"\
                   f"- 会议室: {llm_result['会议室']}\n"\
                   f"- 日期: {llm_result['日期']}\n"\
                   f"- 时间: {llm_result['开始时间']} - {llm_result['结束时间']}\n\n"\
                   f"📝 系统信息: {api_result.get('信息', '')}"
        else:
            return f"❌ 预订失败\n"\
                   f"- 会议室: {llm_result['会议室']}\n"\
                   f"- 日期: {llm_result['日期']}\n"\
                   f"- 时间: {llm_result['开始时间']} - {llm_result['结束时间']}\n\n"\
                   f"📝 系统信息: {api_result.get('信息', '')}"
    
    # 查询结果
    elif "日期" in llm_result:  # 查询响应
        rooms_status = api_result.get("会议室状态", [])
        
        result = f"🔍 查询结果：\n"\
                f"- 日期: {llm_result['日期']}\n"
        
        if isinstance(llm_result.get('时间段'), dict):
            result += f"- 时间段: {llm_result['时间段']['开始时间']} - {llm_result['时间段']['结束时间']}\n"
        else:
            result += f"- 时间段: {llm_result.get('时间段', '全天')}\n"
        
        # 显示会议室状态
        result += "\n会议室状态:\n"
        for room in rooms_status:
            status_icon = "🟢" if room["状态"] == "空闲" else "🔴"
            result += f"{status_icon} {room['会议室']}: {room['状态']}\n"
            
            # 如果有预订详情，显示预订时段
            if room.get("已预订时段"):
                for booking in room["已预订时段"]:
                    result += f"   ⏰ {booking['开始时间']}-{booking['结束时间']} (预订人: {booking.get('预订人', '未知')})\n"
        
        return result
    
    # 未知响应格式
    return f"⚠️ {json.dumps(data, ensure_ascii=False, indent=2)}"

if __name__ == "__main__":
    main()
