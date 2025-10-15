from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Union


#Companies
class CompanyRequest(BaseModel):
    name: str
    industry: str
    location: str
    capital_amount: int
    revenue: int
    Company_certification_documents: str
    patent: bool
    delivery_time: int


#Products
class ProductRequest(BaseModel):
    Company_id: str
    product_name: str
    main_raw_materials: str
    product_standard: str
    technical_advantages: str
    product_certifications: List[str]


#VectorDB
class VectorDBRequest(BaseModel):
    Product_id: str
    Company_id: str
    embedding: List[float]
    metadata_json: dict


#Search
class SearchRequest(BaseModel):
    # Required primary inputs
    query_text: str
    industry: Optional[Union[str, List[str]]] = None
    country: Optional[Union[str, List[str]]] = None
    top_k: int = 5


# class NumericGap(BaseModel):
#     lead_time: Optional[str] = None
#     quality: Optional[str] = None
#     capacity: Optional[str] = None


class SearchResult(BaseModel):
    company: str
    product: Optional[str] = None
    completeness_score: int
    semantic_score: float
    # numeric_gap: NumericGap
    doc_status: str
    total_score: int


#Feedback
class FeedbackRequest(BaseModel):
    query_id: str
    result_id: str
    action_type: str  # "keep", "reject", "compare"


class FeedbackResponse(BaseModel):
    status: str  # "success" or "failure"
    message: str
