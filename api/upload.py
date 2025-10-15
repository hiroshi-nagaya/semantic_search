import pandas as pd
import uuid
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from database.database import Companies, Products, VectorDB, async_session, get_session
from database.schemas import CompanyRequest, ProductRequest, VectorDBRequest
from sqlalchemy import select
from typing import List, Dict, Any
import io
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import date, datetime
import numpy as np
import math
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

load_dotenv()

router = APIRouter()


def calculate_score(row: pd.Series) -> float:
    """
    Calculate score for data.
    Start with 1.0, minus 0.05 for each empty column.
    Handles scalars and list-like values safely.
    """
    score = 1.0
    empty_penalty = 0.05

    def is_empty(value: Any) -> bool:
        # None or NaN
        if value is None:
            return True
        try:
            if pd.isna(value):
                # pd.isna(list) raises, caught below
                return True
        except Exception:
            pass

        # List-like: empty or all items empty/whitespace
        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                return True
            for item in value:
                if item is None:
                    continue
                s = str(item).strip()
                if s != "":
                    return False
            return True

        # Numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return True
            # Consider empty if all stringified items are empty
            return all(str(v).strip() == "" for v in value.flatten())

        # Scalar string/number
        try:
            return str(value).strip() == ""
        except Exception:
            return False

    # Check each column for empty values
    for column in row.index:
        value = row[column]
        if is_empty(value):
            score -= empty_penalty
    return max(0.0, score)  # Ensure score doesn't go below 0


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload Excel or CSV file containing company or product data.
    Automatically updates VectorDB for product data.
    Product Data: File must contain an industry category column (e.g., '產業別', 'industry_category').
    """
    file_id = str(uuid.uuid4())

    # Check file extension
    filename = (file.filename or "").lower()
    file_extension = filename.split('.')[-1] if '.' in filename else ''

    # Read bytes once
    file_bytes = await file.read()

    # Decide parser by extension or content-type
    is_json = file_extension in ["json"] or (file.content_type
                                             and "json" in file.content_type)
    is_tabular = file_extension in ["csv", "xlsx", "xls"]

    if not (is_json or is_tabular):
        raise HTTPException(
            status_code=400,
            detail="Only CSV, Excel, or JSON files are supported")

    try:
        if is_json:
            records = _parse_json_payload(file_bytes)
        else:
            records = _parse_tabular_payload(file_bytes, file_extension)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Failed to parse file: {e}")

    if not records:
        return {
            "status": "ok",
            "message": "No records found",
            "created": 0,
            "updated": 0
        }

    # Upsert into DB
    created_counts = {"companies": 0, "products": 0, "vectors": 0}
    updated_counts = {"companies": 0, "products": 0, "vectors": 0}

    # Ensure OpenAI API key present if embeddings required
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise HTTPException(status_code=500,
                            detail="OPENAI_API_KEY is not set")

    # OpenAI embeddings model
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    async with async_session() as session:
        try:
            for rec in records:
                company_res, company_created = await _upsert_company(
                    session, rec)
                if company_created:
                    created_counts["companies"] += 1
                else:
                    updated_counts["companies"] += 1

                product_res, product_created = await _upsert_product(
                    session, company_res.id, rec)
                if product_created:
                    created_counts["products"] += 1
                else:
                    updated_counts["products"] += 1

                # Build metadata and score
                metadata, score = _build_metadata(rec)

                # Create embedding text per item
                embedding_text = _build_embedding_text(metadata)

                # Call OpenAI for embedding (supports both old and new clients defensively)
                embedding = await _create_embedding_async(
                    embedding_text, embedding_model, openai_api_key)

                vec_res, vec_created = await _upsert_vector(
                    session,
                    company_id=company_res.id,
                    product_id=product_res.id,
                    embedding=embedding,
                    metadata=metadata)
                if vec_created:
                    created_counts["vectors"] += 1
                else:
                    updated_counts["vectors"] += 1

            await session.commit()
        except Exception:
            await session.rollback()
            raise

    # Update todos via return payload
    total_created = sum(created_counts.values())
    total_updated = sum(updated_counts.values())
    return {
        "status": "ok",
        "file_id": file_id,
        "created": created_counts,
        "updated": updated_counts,
        "total_created": total_created,
        "total_updated": total_updated,
    }


def _parse_tabular_payload(file_bytes: bytes,
                           file_extension: str) -> List[Dict[str, Any]]:
    buffer = io.BytesIO(file_bytes)
    if file_extension == "csv":
        df = pd.read_csv(buffer)
    else:
        df = pd.read_excel(buffer)

    # Normalize column names to lower snake for mapping
    df.columns = [str(c).strip() for c in df.columns]

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        normalized = _normalize_row_from_tabular(row)
        if normalized:
            records.append(normalized)
    return records


def _parse_json_payload(file_bytes: bytes) -> List[Dict[str, Any]]:
    payload = json.loads(file_bytes.decode("utf-8"))
    items = payload if isinstance(payload, list) else [payload]
    records: List[Dict[str, Any]] = []
    for item in items:
        normalized = _normalize_row_from_json(item)
        if normalized:
            records.append(normalized)
    return records


def _normalize_row_from_tabular(row: pd.Series) -> Dict[str, Any]:
    get = lambda *keys: next((row.get(k) for k in keys
                              if k in row and not (pd.isna(row.get(k)) or str(
                                  row.get(k)).strip() == "")), None)

    # Handle common variants from the screenshot/example
    company_name = get("Company_Name", "company_name", "Company", "name")
    industry = get("Industry_category", "Industry", "industry_category",
                   "industry")
    location = get("Location", "location")
    capital_amount = get("Capital_Amour", "capital_amount", "Capital_amount")
    revenue = get("Revenue", "revenue")
    cert_docs = get("Company_Certification_Documents",
                    "Company_certification_documents", "cert_docs")
    product_name = get("Product_Name", "product_name")
    main_raw_materials = get("Main_Raw_Materials", "main_raw_materials")
    product_standard = get("Product_Standard", "product_standard")
    technical_advantages = get("Technical_advantages", "technical_advantages")
    product_certs = get("product_certifications",
                        "Product_Certification_Materials")
    patent = get("Patent", "patent")
    delivery_time = get("Delivery_time", "delivery_time")

    # Convert types
    def to_int(v):
        try:
            return int(v) if v is not None and str(v).strip() != "" else None
        except Exception:
            return None

    def to_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        s = str(v).strip().lower()
        return s in ["true", "1", "yes", "y"]

    def to_list(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            # split by commas or semicolons
            parts = [
                p.strip() for p in v.replace(";", ",").split(",")
                if p.strip() != ""
            ]
            return parts
        return [str(v)]

    normalized = {
        "company_name":
        str(company_name).strip() if company_name is not None else None,
        "industry_category":
        str(industry).strip() if industry is not None else None,
        "location":
        str(location).strip() if location is not None else None,
        "capital_amount":
        to_int(capital_amount),
        "revenue":
        to_int(revenue),
        "company_certification_documents":
        str(cert_docs).strip() if cert_docs is not None else None,
        "product_name":
        str(product_name).strip() if product_name is not None else None,
        "main_raw_materials":
        str(main_raw_materials).strip()
        if main_raw_materials is not None else None,
        "product_standard":
        to_list(product_standard),
        "technical_advantages":
        str(technical_advantages).strip()
        if technical_advantages is not None else None,
        "product_certifications":
        to_list(product_certs),
        "patent":
        to_bool(patent),
        "delivery_time":
        to_int(delivery_time),
    }
    return normalized


def _normalize_row_from_json(item: Dict[str, Any]) -> Dict[str, Any]:
    # example.yaml structure
    product = item.get("Product") or {}

    def to_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [
                p.strip() for p in v.replace(";", ",").split(",") if p.strip()
            ]
        return [str(v)]

    normalized = {
        "company_name":
        item.get("Company_Name") or item.get("company_name"),
        "industry_category":
        item.get("Industry_category") or item.get("industry")
        or item.get("industry_category"),
        "location":
        item.get("Location") or item.get("location"),
        "capital_amount":
        item.get("capital_amount"),
        "revenue":
        item.get("Revenue") or item.get("revenue"),
        "company_certification_documents":
        item.get("Company_certification_documents")
        or item.get("Company_Certification_Documents"),
        "product_name":
        product.get("Product_Name") or product.get("product_name"),
        "main_raw_materials":
        product.get("Main_Raw_Materials") or product.get("main_raw_materials"),
        "product_standard":
        to_list(
            product.get("Product_Standard")
            or product.get("product_standard")),
        "technical_advantages":
        product.get("Technical_Advantages")
        or product.get("technical_advantages"),
        "product_certifications":
        to_list(
            product.get("Product_Certification_Materials")
            or product.get("product_certifications")),
        "patent":
        item.get("Patent") if isinstance(item.get("Patent"), bool) else str(
            item.get("Patent")).lower() in ["true", "1", "yes", "y"],
        "delivery_time":
        item.get("Delivery_time") or item.get("delivery_time"),
    }
    return normalized


async def _upsert_company(session: AsyncSession, rec: Dict[str, Any]):
    now = datetime.utcnow().isoformat()
    company_name = rec.get("company_name")
    if not company_name:
        raise HTTPException(status_code=400,
                            detail="Missing company_name in a record")

    result = await session.execute(
        select(Companies).where(Companies.name == company_name))
    existing = result.scalars().first()
    if existing:
        existing.industry = rec.get("industry_category") or existing.industry
        existing.location = rec.get("location") or existing.location
        if rec.get("capital_amount") is not None:
            existing.capital_amount = rec["capital_amount"]
        if rec.get("revenue") is not None:
            existing.revenue = rec["revenue"]
        if rec.get("company_certification_documents") is not None:
            existing.Company_certification_documents = rec[
                "company_certification_documents"]
        if rec.get("patent") is not None:
            existing.patent = bool(rec["patent"])
        if rec.get("delivery_time") is not None:
            existing.delivery_time = rec["delivery_time"]
        existing.updated_at = now
        return existing, False
    else:
        company = Companies(
            id=str(uuid.uuid4()),
            name=company_name,
            industry=rec.get("industry_category") or "",
            location=rec.get("location") or "",
            capital_amount=rec.get("capital_amount") or 0,
            revenue=rec.get("revenue") or 0,
            Company_certification_documents=rec.get(
                "company_certification_documents") or "",
            patent=bool(rec.get("patent"))
            if rec.get("patent") is not None else False,
            delivery_time=rec.get("delivery_time") or 0,
            created_at=now,
            updated_at=now,
        )
        session.add(company)
        return company, True


async def _upsert_product(session: AsyncSession, company_id: str,
                          rec: Dict[str, Any]):
    now = datetime.utcnow().isoformat()
    product_name = rec.get("product_name")
    if not product_name:
        # Create a shell product if no product fields? Skip instead.
        raise HTTPException(status_code=400,
                            detail="Missing product_name in a record")

    result = await session.execute(
        select(Products).where(Products.Company_id == company_id,
                               Products.product_name == product_name))
    existing = result.scalars().first()

    standards = rec.get("product_standard") or []
    certs = rec.get("product_certifications") or []
    if isinstance(standards, str):
        standards = [s for s in standards.split(",") if s.strip()]
    if isinstance(certs, str):
        certs = [s for s in certs.split(",") if s.strip()]

    if existing:
        existing.main_raw_materials = rec.get(
            "main_raw_materials") or existing.main_raw_materials
        existing.product_standard = standards or existing.product_standard
        existing.technical_advantages = rec.get(
            "technical_advantages") or existing.technical_advantages
        existing.product_certifications = certs or existing.product_certifications
        existing.updated_at = now
        return existing, False
    else:
        product = Products(
            id=str(uuid.uuid4()),
            Company_id=company_id,
            product_name=product_name,
            main_raw_materials=rec.get("main_raw_materials") or "",
            product_standard=standards,
            technical_advantages=rec.get("technical_advantages") or "",
            product_certifications=certs,
            created_at=now,
            updated_at=now,
        )
        session.add(product)
        return product, True


def _build_metadata(rec: Dict[str, Any]):
    # Prepare pandas series for scoring
    score_series = pd.Series({
        k: rec.get(k)
        for k in [
            "company_name", "industry_category", "location", "product_name",
            "main_raw_materials", "product_standard", "technical_advantages",
            "product_certifications", "delivery_time"
        ]
    })
    data_score = calculate_score(score_series)

    metadata = {
        "company_name": rec.get("company_name"),
        "industry_category": rec.get("industry_category"),
        "location": rec.get("location"),
        "product_name": rec.get("product_name"),
        "main_raw_materials": rec.get("main_raw_materials"),
        "product_standard": rec.get("product_standard") or [],
        "technical_advantages": rec.get("technical_advantages"),
        "certifications": rec.get("product_certifications") or [],
        "delivery_time": rec.get("delivery_time"),
        "data_score": data_score,
    }
    return metadata, data_score


def _build_embedding_text(metadata: Dict[str, Any]) -> str:
    parts = [
        metadata.get("company_name") or "",
        metadata.get("industry_category") or "",
        metadata.get("location") or "",
        metadata.get("product_name") or "",
        metadata.get("main_raw_materials") or "",
        ", ".join(metadata.get("product_standard") or []),
        metadata.get("technical_advantages") or "",
        ", ".join(metadata.get("certifications") or []),
    ]
    return " | ".join([str(p) for p in parts if str(p).strip() != ""])


async def _create_embedding_async(text: str, model: str,
                                  api_key: str) -> List[float]:
    """Create embeddings using OpenAI client in a background thread.
    The OpenAI Python client methods are synchronous; we offload to a thread.
    """
    client = OpenAI(api_key=api_key)

    def _sync_create() -> List[float]:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return await asyncio.to_thread(_sync_create)


async def _upsert_vector(session: AsyncSession, company_id: str,
                         product_id: str, embedding: List[float],
                         metadata: Dict[str, Any]):
    now = datetime.utcnow().isoformat()
    # Check if there is an existing vector for this product
    result = await session.execute(
        select(VectorDB).where(VectorDB.Product_id == product_id))
    existing = result.scalars().first()
    if existing:
        existing.embedding = embedding
        existing.metadata_json = metadata
        existing.updated_at = now
        return existing, False
    else:
        vector = VectorDB(
            id=str(uuid.uuid4()),
            Product_id=product_id,
            Company_id=company_id,
            embedding=embedding,
            metadata_json=metadata,
            created_at=now,
            updated_at=now,
        )
        session.add(vector)
        return vector, True
