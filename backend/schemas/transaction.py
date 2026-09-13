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
    # Additive (migration 005): WHY predicted_category is what it is --
    # 'model' | 'structural_other' | 'low_confidence_other' | 'gazetteer' |
    # 'ambiguous_e_transfer', or None for a pre-migration row or a manual
    # entry (TransactionService.create_manual() never runs the decide()/
    # decide_batch() path, so there is no decision-path reason to record).
    # This is a SYSTEM fact about how predicted_category was originally
    # reached; it is never touched by a later human correction, so
    # confirmed_category/effective_category remain the sole source of truth
    # for what the transaction actually is.
    decision_source: str | None = None
    # Additive (migration 006): the classifier's raw opinion, even when a
    # low-confidence abstention overrode it to "Other" (predicted_category).
    # Advisory model metadata ONLY -- never affects predicted_category/
    # confirmed_category/effective_category, and it is never touched by a
    # later human correction (same "frozen at decide-time" rule as
    # decision_source). None on structural/ambiguous-e-transfer/manual rows,
    # or a pre-migration row. The frontend uses this to show a "Suggested:
    # {model_category}" advisory chip with a one-click accept for a
    # low_confidence_other row -- see CategoryBadge.tsx.
    model_category: str | None = None
    # Additive (migration 008): spending eligibility, ORTHOGONAL to category.
    # 'spending' | 'internal_transfer' -- backend.services.transfer_eligibility.
    # An 'internal_transfer' row's predicted_category still reads "Other" (a
    # display fallback only, never a real category guess), but this field is
    # what the frontend uses to render it as "Internal transfer" instead of
    # a category badge, and it is what every spend/forecast/category-summary
    # aggregate already excludes it by. None only for a pre-migration row.
    transaction_type: str | None = None
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
