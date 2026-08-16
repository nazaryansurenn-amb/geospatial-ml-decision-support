from __future__ import annotations

from datetime import date
from io import BytesIO
import json
import re
from typing import Any

import pandas as pd

CANONICAL_FIELDS = {
    "customer_code": {
        "label": "Customer code",
        "required": False,
        "aliases": ["customer code", "customer_code", "code", "customer id", "customer_id", "id"],
    },
    "full_name": {
        "label": "Full name",
        "required": True,
        "aliases": ["full name", "full_name", "name", "customer", "debtor", "abonent"],
    },
    "address": {
        "label": "Address",
        "required": True,
        "aliases": ["address", "recipient address", "postal address", "location"],
    },
    "community": {
        "label": "Community / WUA",
        "required": False,
        "aliases": ["community", "wua", "village", "settlement", "region", "branch"],
    },
    "phone": {
        "label": "Phone",
        "required": False,
        "aliases": ["phone", "mobile", "telephone", "tel", "contact"],
    },
    "debt_amount": {
        "label": "Debt amount",
        "required": True,
        "aliases": [
            "debt",
            "debt amount",
            "amount due",
            "balance",
            "final balance",
            "final debt",
            "outstanding",
            "arrears",
        ],
    },
    "debt_period": {
        "label": "Debt period",
        "required": False,
        "aliases": ["period", "debt period", "billing period", "month", "season", "year"],
    },
    "account_status": {
        "label": "Account status",
        "required": False,
        "aliases": ["status", "account status", "debtor status", "state"],
    },
}

DEFAULT_NOTICE_TEMPLATE = """Debt notice

Dear {full_name},

According to our records, your water-use debt is {debt_amount_text} AMD{period_sentence}.
Please visit the payment point or office of {organization_name} and settle the amount{deadline_sentence}.

Address on file: {address}
Reference code: {customer_code}

For questions please contact {contact_phone}.
"""


def load_debt_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix == "csv":
        return pd.read_csv(BytesIO(file_bytes))
    if suffix in {"xlsx", "xlsm", "xltx", "xltm", "xls"}:
        return pd.read_excel(BytesIO(file_bytes))
    raise ValueError(f"Unsupported file type: {filename}")


def normalize_header(name: Any) -> str:
    text = "" if name is None else str(name).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def suggest_column_map(columns: list[str]) -> dict[str, str]:
    normalized = {col: normalize_header(col) for col in columns}
    mapping: dict[str, str] = {}
    for field, meta in CANONICAL_FIELDS.items():
        aliases = {normalize_header(alias) for alias in meta["aliases"] + [meta["label"], field]}
        selected = ""
        for col, norm in normalized.items():
            if norm in aliases:
                selected = col
                break
        mapping[field] = selected
    return mapping


def validate_required_mapping(column_map: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for field, meta in CANONICAL_FIELDS.items():
        if meta["required"] and not column_map.get(field):
            missing.append(meta["label"])
    return missing


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def clean_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    for token in ["<<", ">>", "<<", ">>", '"']:
        text = text.replace(token, "")
    return re.sub(r"\s+", " ", text).strip()


def clean_amount(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    text = text.replace("AMD", "").replace("amd", "")
    text = text.replace(",", "")
    text = text.replace(" ", "")
    text = re.sub(r"[^\d\.\-]", "", text)
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def normalize_phone(value: Any) -> str:
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return ""
    if digits.startswith("374") and len(digits) == 11:
        return f"+{digits}"
    if digits.startswith("0") and len(digits) == 9:
        return f"+374{digits[1:]}"
    if len(digits) == 8:
        return f"+374{digits}"
    if str(value).strip().startswith("+"):
        return str(value).strip()
    return digits


def normalize_debt_table(
    df: pd.DataFrame,
    column_map: dict[str, str],
    only_positive_debt: bool = True,
) -> pd.DataFrame:
    normalized = pd.DataFrame(index=df.index)
    for field in CANONICAL_FIELDS:
        source = column_map.get(field, "")
        if source and source in df.columns:
            normalized[field] = df[source]
        else:
            normalized[field] = ""

    normalized["customer_code"] = normalized["customer_code"].apply(clean_text)
    normalized["full_name"] = normalized["full_name"].apply(clean_name)
    normalized["address"] = normalized["address"].apply(clean_text)
    normalized["community"] = normalized["community"].apply(clean_text)
    normalized["phone"] = normalized["phone"].apply(normalize_phone)
    normalized["debt_amount"] = normalized["debt_amount"].apply(clean_amount)
    normalized["debt_period"] = normalized["debt_period"].apply(clean_text)
    normalized["account_status"] = normalized["account_status"].apply(clean_text)

    normalized = normalized.loc[
        ~(normalized["full_name"].eq("") & normalized["address"].eq("") & normalized["debt_amount"].eq(0))
    ].copy()

    if only_positive_debt:
        normalized = normalized[normalized["debt_amount"] > 0].copy()

    normalized["has_address"] = normalized["address"].ne("")
    normalized["has_phone"] = normalized["phone"].ne("")
    normalized["ready_for_haypost"] = normalized["debt_amount"].gt(0) & normalized["has_address"]
    normalized["notification_status"] = "new"
    normalized["haypost_request_id"] = ""
    normalized["tracking_number"] = ""
    normalized["dispatch_channel"] = "haypost"
    normalized["last_updated"] = pd.Timestamp.utcnow().tz_localize(None)

    return normalized.reset_index(drop=True)


def summarize_debt_table(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "records": 0,
            "debtors": 0,
            "total_debt": 0.0,
            "ready_for_haypost": 0,
            "missing_address": 0,
        }
    return {
        "records": int(len(df)),
        "debtors": int(df["debt_amount"].gt(0).sum()),
        "total_debt": float(df["debt_amount"].sum()),
        "ready_for_haypost": int(df["ready_for_haypost"].sum()),
        "missing_address": int(df["address"].eq("").sum()),
    }


def build_notice_text(
    row: pd.Series,
    organization_name: str,
    payment_deadline: date | None,
    contact_phone: str,
    template: str,
) -> str:
    deadline_sentence = ""
    if payment_deadline is not None:
        deadline_sentence = f" by {payment_deadline.isoformat()}"
    period_sentence = ""
    if row.get("debt_period"):
        period_sentence = f" for period {row['debt_period']}"

    values = {
        "full_name": row.get("full_name", ""),
        "debt_amount": row.get("debt_amount", 0),
        "debt_amount_text": f"{float(row.get('debt_amount', 0)):,.0f}",
        "debt_period": row.get("debt_period", ""),
        "period_sentence": period_sentence,
        "deadline_sentence": deadline_sentence,
        "organization_name": organization_name,
        "contact_phone": contact_phone or "your local office",
        "address": row.get("address", ""),
        "customer_code": row.get("customer_code", ""),
        "community": row.get("community", ""),
    }
    return template.format(**values).strip()


def attach_notice_texts(
    df: pd.DataFrame,
    organization_name: str,
    payment_deadline: date | None,
    contact_phone: str,
    template: str = DEFAULT_NOTICE_TEMPLATE,
) -> pd.DataFrame:
    prepared = df.copy()
    prepared["notice_text"] = prepared.apply(
        lambda row: build_notice_text(
            row=row,
            organization_name=organization_name,
            payment_deadline=payment_deadline,
            contact_phone=contact_phone,
            template=template,
        ),
        axis=1,
    )
    return prepared


def filter_debt_table(
    df: pd.DataFrame,
    query: str,
    ready_only: bool,
    status_filter: str,
) -> pd.DataFrame:
    filtered = df.copy()
    if ready_only:
        filtered = filtered[filtered["ready_for_haypost"]].copy()
    if status_filter != "All":
        filtered = filtered[filtered["notification_status"] == status_filter].copy()
    if query.strip():
        pattern = re.escape(query.strip())
        haystack = (
            filtered["customer_code"].fillna("")
            + " "
            + filtered["full_name"].fillna("")
            + " "
            + filtered["address"].fillna("")
            + " "
            + filtered["community"].fillna("")
        )
        filtered = filtered[haystack.str.contains(pattern, case=False, na=False)].copy()
    return filtered.reset_index(drop=True)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_json_bytes(df: pd.DataFrame) -> bytes:
    records = json.loads(df.to_json(orient="records", date_format="iso"))
    return json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
