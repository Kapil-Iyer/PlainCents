"""Transaction schemas (TRD §6, §5.4)."""
import re
from datetime import datetime

from pydantic import BaseModel, field_validator

from backend.config import CATEGORIES

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date(value: str) -> str:
    if not _DATE_RE.match(value):
        raise ValueError("date must be in YYYY-MM-DD format")
    return value


def _validate_category(value: str | None) -> str | None:
    if value is not None and value not in CATEGORIES:
        raise ValueError(f"confirmed_category must be one of {CATEGORIES}")
    return value


class TransactionCreate(BaseModel):
    date: str
    merchant: str
    amount: float
    confirmed_category: str | None = None

    _validate_date = field_validator("date")(_validate_date)
    _validate_confirmed_category = field_validator("confirmed_category")(_validate_category)

    @field_validator("merchant")
    @classmethod
    def _merchant_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("merchant must not be blank")
        return value


class TransactionUpdate(BaseModel):
    date: str | None = None
    merchant: str | None = None
    amount: float | None = None
    confirmed_category: str | None = None

    _validate_date = field_validator("date")(_validate_date)
    _validate_confirmed_category = field_validator("confirmed_category")(_validate_category)

    @field_validator("merchant")
    @classmethod
    def _merchant_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("merchant must not be blank")
        return value


class TransactionResponse(BaseModel):
    id: int
    date: str
    merchant: str
    raw_description: str | None
    amount: float
    bank_source: str | None
    predicted_category: str
    confirmed_category: str | None
    effective_category: str
    is_manual_override: bool
    created_at: datetime
    updated_at: datetime

    @field_validator("is_manual_override", mode="before")
    @classmethod
    def _coerce_bool(cls, value):
        # SQLite stores this VIEW-computed boolean as 0/1.
        return bool(value)


class TransactionListResponse(BaseModel):
    items: list[TransactionResponse]
    total: int
    page: int
    page_size: int
