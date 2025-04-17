from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import uvicorn
import httpx
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


rooms_id_dict = {
    "长江厅(上海8号楼3楼)": "14",
    "峨山厅(上海8号楼6楼)": "12",
    "关山厅(上海8号楼6楼)": "6",
    "华山厅(上海8号楼6楼)": "24",
    "黄河厅(上海8号楼3楼)": "15",
    "黄埔厅(上海8号楼3楼)": "16",
    "乐山厅(上海8号楼6楼)": "4",
    "宜山厅(上海8号楼6楼)": "5",
    "凌云厅(上海17号楼)": "19",
    "摘星厅（上海17号楼）": "18",
    "纵横厅（上海17号楼）": "17",
    "接待室01（上海17号楼）": "20",
    "接待室02（上海17号楼）": "21",
    "接待室03（上海17号楼）": "22",
    "接待室04（上海17号楼）": "23",
    "综合楼会议室（电子）": "25",
    "东湖厅(武汉办公室）": "9",
    "蠡湖厅(武汉办公室）": "10",
    "南湖厅(武汉办公室）": "8",
    "浦江厅(武汉办公室）": "7",
    "清江厅(武汉办公室）": "11",
    "贡湖厅(无锡小会议室)": "13",
    "灵山厅(无锡大会议室)": "3",
    "太湖厅(无锡中会议室)": "1"
}

rooms_id_list = list(rooms_id_dict.keys())



# 初始化 FastAPI 应用
app = FastAPI(
    title="会议室预订系统 API",
    description="提供会议室查询功能的 API 服务",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 请求头和URL常量
API_HEADERS = {
    'token': '10e9f8f8-edc5-4a6a-b5ef-da8c27df147a',
    'appid': 'FF5CF901-BF89-23EC-E594-CF85C7CE06B3',
    'userid': 'i8CMuQudA7cy54ylqDkrcH7aDfyiGSELhTPi8PPCutLE9nc1P2n3+NPTXT+hMImBftmGjcqQDiR3oAoJVrV/ph5YVMgcFyZl5ay7arVtPHm+LRyYN0HYJcUz106uf98aGUDYt0pR0Vu/3yyYSKfeV9sT4ia9KgPO55FQDKPOgzdj9vPKBnOjipUhY3G953nbkBHaJRSXA9sL74r6ZBKxf5kw9k9doTz5rtTLjowDvx0cKjfsFTQP66YKujFikT0Nm7jGI7fvHp3kg5LhOgw9cTjXAoM2ChHdqFB1lN7SMJ/PIjAoI6xD3WcZsaH48/tNDNbLaZEnq3ufswFBTdBJ1g==',
    'skipsession': '0',
    'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
}
API_URL = "http://10.0.1.153/api/meeting/room/getRoomReportData"
BOOKING_API_URL = "http://10.0.1.153/api/meeting/base/newMeeting"


@app.get("/")
async def read_root():
    return {"message": "欢迎使用会议室预订系统 API"}

@app.get("/api/room-availability-simple")
async def get_room_availability_simple(date: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    """
    获取会议室在指定时间段的可用性，返回精简格式
    
    Args:
        date: 查询日期，格式为YYYY-MM-DD，默认为当天
        start_time: 开始时间，格式为HH:MM，默认为08:00
        end_time: 结束时间，格式为HH:MM，默认为18:00
        
    Returns:
        会议室在指定时间段的可用性信息，简化格式
    """
    # 如果未指定日期，使用当前日期
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # 默认时间段为工作时间
    if start_time is None:
        start_time = "09:00"
    if end_time is None:
        end_time = "20:00"
    
    try:
        async with httpx.AsyncClient() as client:
            # 构造请求参数，使用传入的date作为currentdate
            params = {
                "roomname": "",
                "mrtype": "",
                "equipment": "",
                "currentdate": date,  # 使用传入的date
                "bywhat": "4",
                "curmoment": f"{date} {datetime.now().strftime('%H:%M:%S')} GMT+0800",
                "roomid": ""
            }
            
            # 发送请求获取会议室数据
            response = await client.post(API_URL, headers=API_HEADERS, params=params)
            response.raise_for_status()
            
            # 解析响应
            data = response.json()
            # 返回精简数据
            return {
                "date": date,
                "rooms": parse_room_availability_simplified(data, date, start_time, end_time)
            }
            
    except Exception as e:
        logger.error(f"获取会议室信息失败: {str(e)}")
        return {
            "date": date,
            "rooms": {}
        }

def parse_room_availability_simplified(data: Dict[str, Any], query_date: str, start_time: str, end_time: str) -> Dict[str, Any]:
    """解析会议室可用性数据，以精简格式返回"""
    # 生成查询时间段内的所有半小时时间点
    time_points = []
    current_time = datetime.strptime(start_time, "%H:%M")
    end_datetime = datetime.strptime(end_time, "%H:%M")
    
    while current_time <= end_datetime:
        time_points.append(current_time.strftime("%H:%M"))
        current_time += timedelta(minutes=30)
    
    result = {}
    
    # 先处理API返回的会议室数据
    if "datas" in data:
        room_mapping = {}
        if "rooms" in data:
            room_mapping = {room["id"]: room["name"] for room in data.get("rooms", [])}
        
        for room_data in data.get("datas", []):
            room_id = room_data.get("roomid")
            if not room_id or room_id not in room_mapping:
                continue
                
            room_name = room_mapping[room_id]
            room_result = {
                "available_time": [],
                "busy_time": []
            }
            
            # 获取会议室预订时间表
            room_status = {}
            for info in room_data.get("info", []):
                time_slot = info.get("time", "")
                # 只处理在查询时间范围内的时间点
                if time_slot in time_points:
                    is_available = info.get("content", 0) == 0
                    room_status[time_slot] = is_available
            
            # 将所有未指定状态的时间点默认为可用
            for time_point in time_points:
                if time_point not in room_status:
                    room_status[time_point] = True
            
            # 合并连续的时间段
            current_status = None
            current_start = None
            
            for i, time_point in enumerate(time_points):
                status = room_status.get(time_point, True)
                
                # 如果状态改变或者是最后一个时间点
                if current_status is None or current_status != status or i == len(time_points) - 1:
                    # 保存上一个时间段（如果存在）
                    if current_status is not None and current_start is not None:
                        # 设置结束时间点
                        end_time_slot = time_point
                        if i == len(time_points) - 1 and current_status == status:
                            next_time = (datetime.strptime(time_point, "%H:%M") + timedelta(minutes=30)).strftime("%H:%M")
                            end_time_slot = next_time
                        
                        time_range = {
                            "start_time": f"{current_start}:00",
                            "end_time": f"{end_time_slot}:00"
                        }
                        
                        if current_status:
                            room_result["available_time"].append(time_range)
                        else:
                            room_result["busy_time"].append(time_range)
                    
                    # 开始新的时间段
                    current_status = status
                    current_start = time_point
            
            result[room_name] = room_result
    
    # 添加rooms_list中的所有会议室，如果已存在则跳过
    for room_name in rooms_id_list:
        if room_name not in result:
            # 对于未在API返回中的会议室，设置为全天可用
            result[room_name] = {
                "available_time": [
                    {
                        "start_time": f"{start_time}:00",
                        "end_time": f"{end_time}:00"
                    }
                ],
                "busy_time": []
            }
    
    return result

def get_room_id_by_simple_name(simple_name: str) -> str:
    """
    通过简化的会议室名称获取会议室ID
    
    Args:
        simple_name: 简化的会议室名称，如"乐山厅"
        
    Returns:
        会议室ID，如果找不到则抛出异常
    """
    for full_name, room_id in rooms_id_dict.items():
        if simple_name in full_name:
            return room_id
    raise HTTPException(status_code=400, detail=f"找不到会议室: {simple_name}")

def parse_datetime(datetime_str: str) -> tuple:
    """
    解析日期时间字符串为日期和时间部分
    
    Args:
        datetime_str: 日期时间字符串，格式为"YYYY-MM-DD HH:MM"
        
    Returns:
        包含日期和时间的元组 (date_str, time_str)
    """
    try:
        # 尝试解析日期时间字符串
        date_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        return date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"无效的日期时间格式: {datetime_str}，请使用 YYYY-MM-DD HH:MM 格式")

@app.post("/api/book-room")
async def book_room(
    room_name: str,
    meeting_name: str,
    start_datetime: str,
    end_datetime: str,
    caller_id: str = "816",
    contacter_id: str = "816",
    description: str = "",
    total_members: int = 1
):
    """
    预订会议室
    
    Args:
        room_name: 会议室名称（可以是简化名称，如"乐山厅"）
        meeting_name: 会议名称
        start_datetime: 开始日期时间 (YYYY-MM-DD HH:MM)
        end_datetime: 结束日期时间 (YYYY-MM-DD HH:MM)
        caller_id: 发起人ID，默认为816
        contacter_id: 联系人ID，默认为816
        description: 会议描述，默认为空
        total_members: 参会人数，默认为1
        
    Returns:
        预订结果，包含会议ID和状态
    """
    try:
        # 解析开始和结束日期时间
        start_date, start_time = parse_datetime(start_datetime)
        end_date, end_time = parse_datetime(end_datetime)
        
        # 验证会议室ID是否有效
        room_id = get_room_id_by_simple_name(room_name)

        # 构造请求数据
        data = {
            "meetingtype": "",
            "name": meeting_name,
            "caller": caller_id,
            "contacter": contacter_id,
            "desc_n": description,
            "project": "",
            "begindate": start_date,
            "begintime": start_time,
            "enddate": end_date,
            "endtime": end_time,
            "address": room_id,
            "zztxhyh": "",
            "hrmmembers": caller_id,
            "othermembers": "",
            "totalmember": str(total_members),
            "servicerows": "0",
            "topicrows": "0",
            "maxRepeatDate": "",
            "method": "",
            "isInterval": "",
            "serviceitems": "",
            "fontsize": "16",
            "bgcolor": "#006699",
            "fontcolor": "#ffffff",
            "showhead": "1",
            "showdep": "1",
            "showsub": "1",
            "shownum": "1",
            "showtitle": "1",
            "bgimg": "",
            "bgimgTemp": "",
            "bgimgLoadlink": "",
            "cfg": "{}",
            "allowSignBack": "1",
            "afterSignCanBack": "5",
            "defaultAllowSignTime": "5",
            "defaultAllowSignBackTime": "30",
            "meetingid": ""
        }
        # print(data)
        async with httpx.AsyncClient() as client:
            print('===============1')
            response = await client.post(
                BOOKING_API_URL,
                headers=API_HEADERS,
                data=data
            )
            print('===============')
            print(response)
            print('===============')
            response.raise_for_status()
            
            
            result = response.json()
            if result.get("status") and result.get("meetingid"):
                return {
                    "status": "success",
                    "meeting_id": result["meetingid"],
                    "message": "会议室预订成功"
                }
            else:
                return {
                    "status": "error",
                    "message": "会议室预订失败",
                    "details": result
                }
                
    except Exception as e:
        import traceback
        error_stack = traceback.format_exc()
        logger.error(f"预订会议室失败: {str(e)}")
        logger.error(f"错误堆栈: {error_stack}")
        raise HTTPException(status_code=500, detail=f"预订会议室失败: {str(e)}")

# 主函数，用于启动API服务器
if __name__ == "__main__":
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True) 