from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import httpx

# 初始化 FastAPI 应用
app = FastAPI(
    title="会议室预订系统 API",
    description="提供会议室预订和查询功能的 API 服务",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头信息
)

# 模型定义
class BookingRequest(BaseModel):
    会议室: str
    日期: str
    开始时间: str
    结束时间: str
    预订人: Optional[str] = "默认用户"

class BookingResponse(BaseModel):
    成功: bool
    信息: str
    预订详情: Optional[dict] = None

class QueryRequest(BaseModel):
    日期: str
    时间段: str
    会议室列表: str

class QueryResponse(BaseModel):
    日期: str
    会议室列表: List[str]  # 修改为必填字段
    会议室状态: List[dict]  # 修改为必填字段


# 模拟数据 - 会议室列表
ROOMS = ["宜山厅", "徐汇厅", "浦东厅", "静安厅", "黄浦厅"]
ROOMS_STATUS = [
    {"会议室": "宜山厅", "状态": "空闲"},
    {"会议室": "徐汇厅", "状态": "空闲"},
    {"会议室": "浦东厅", "状态": "空闲"},
    {"会议室": "静安厅", "状态": "空闲"},
    {"会议室": "黄浦厅", "状态": "空闲"}
]

# API 路由
@app.get("/")
async def read_root():
    return {"message": "欢迎使用会议室预订系统 API"}

@app.post("/api/book", response_model=BookingResponse)
async def book_room(request: BookingRequest):
    """预订会议室"""
    room = request.会议室
    date = request.日期
    start_time = request.开始时间
    end_time = request.结束时间
    
    # 检查会议室是否存在
    if room not in ROOMS:
        raise HTTPException(status_code=404, detail=f"会议室 '{room}' 不存在")
    
    return BookingResponse(
        成功=True,
        信息=f"预订成功：{room} 在 {date} {start_time}-{end_time}",
        预订详情={
            "会议室": room,
            "日期": date,
            "开始时间": start_time,
            "结束时间": end_time,
            "预订人": request.预订人
        }
    )

@app.post("/api/query", response_model=QueryResponse)
async def get_rooms(request: QueryRequest):
    """获取所有会议室列表"""
    return QueryResponse(
        日期=request.日期,
        会议室列表=ROOMS,
        会议室状态=ROOMS_STATUS
    )

# 主函数，用于启动API服务器
if __name__ == "__main__":
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True) 