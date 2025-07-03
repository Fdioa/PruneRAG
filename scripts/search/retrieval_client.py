import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator, model_validator , Field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    queries: List[str] = Field(default_factory=list)
    topk: Optional[int] = 10
    return_scores: bool = True

    @field_validator('return_scores')
    def validate_return_scores(cls, v, info):
        if v and 'topk' not in info.data:
            raise ValueError('开启score返回时必须指定topk参数')
        return v

    @field_validator('queries')
    def validate_queries(cls, v):
        if not v:
            raise ValueError('查询列表不能为空')
        # for idx, query in enumerate(v):
            # if not query.strip():
            #     raise ValueError(f'第{idx+1}个查询内容为空')
            # if len(query) > 1000:
            #     raise ValueError(f'第{idx+1}个查询超过1000字符限制')
        return v

    @field_validator('topk')
    def validate_topk(cls, v):
        if v is not None:
            if v <= 0:
                raise ValueError('topk必须为正整数')
            if v > 100:
                raise ValueError('topk不能超过100')
        return v
class Document(BaseModel):
    id: str
    contents: str

class QueryResult(BaseModel):
    document: Document  # 嵌套Document对象
    score: Optional[float] = None  # 分数可能不存在于某些场景

class QueryResponse(BaseModel):
    results: List[List[QueryResult]]

class RetrievalClient:
    def __init__(self, base_url: str = 'http://localhost:8000', max_retries: int = 3):
        self.base_url = base_url.rstrip('/')

    def query(self, request: QueryRequest, timeout: float = 5.0) -> QueryResponse:
        """
        执行检索查询
        
        :param request: 查询请求参数
        :param timeout: 请求超时时间（秒）
        :return: 解析后的响应结果
        """
        endpoint = f"{self.base_url}/retrieve"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                url=endpoint,
                json=request.model_dump(),
                # timeout=timeout,
                headers=headers
            )

            print(response.raise_for_status())
            print(response.status_code)
            res_json = self._parse_response(response.json())
            # print(res_json)

            
            return res_json
            
        except requests.exceptions.RequestException as e:
            logger.error(f"检索请求失败: {str(e)}")
            logger.debug(f"请求详情: URL={endpoint} Payload={request.model_dump}")
            raise RuntimeError(f"检索服务调用失败: {str(e)}") from e
        except Exception as e:
            logger.error(f"响应解析异常: {str(e)}")
            raise RuntimeError(f"响应数据解析失败: {str(e)}") from e

    def _parse_response(self, raw_data: dict) -> QueryResponse:
        """解析原始响应数据"""
        # print(raw_data)
        try:
            return QueryResponse(**raw_data)
        except Exception as e:
            logger.error(f"响应数据校验失败: {str(e)}")
            raise ValueError(f"无效的响应格式: {str(e)}") from e