from __future__ import annotations

from datetime import datetime

import pandas as pd


def build_haypost_export(df: pd.DataFrame) -> pd.DataFrame:
    export_df = df.copy()
    export_df["recipient_name"] = export_df["full_name"]
    export_df["recipient_address"] = export_df["address"]
    export_df["service_type"] = "registered_letter"
    ordered_cols = [
        "customer_code",
        "recipient_name",
        "recipient_address",
        "community",
        "phone",
        "debt_amount",
        "debt_period",
        "notice_text",
        "service_type",
        "notification_status",
        "haypost_request_id",
        "tracking_number",
        "last_updated",
    ]
    return export_df[[col for col in ordered_cols if col in export_df.columns]].copy()


def build_haypost_payloads(df: pd.DataFrame) -> list[dict]:
    rows = build_haypost_export(df).to_dict(orient="records")
    return rows


def mock_dispatch(df: pd.DataFrame) -> pd.DataFrame:
    dispatched = df.copy()
    now = datetime.utcnow().replace(microsecond=0)
    request_ids = []
    tracking_numbers = []
    statuses = []
    for idx, row in dispatched.reset_index(drop=True).iterrows():
        suffix = f"{now:%Y%m%d}{idx + 1:05d}"
        if bool(row.get("ready_for_haypost", False)):
            request_ids.append(f"HP-REQ-{suffix}")
            tracking_numbers.append(f"HPTRK{suffix}")
            statuses.append("prepared_for_dispatch")
        else:
            request_ids.append("")
            tracking_numbers.append("")
            statuses.append("missing_address")
    dispatched["haypost_request_id"] = request_ids
    dispatched["tracking_number"] = tracking_numbers
    dispatched["notification_status"] = statuses
    dispatched["last_updated"] = now
    return dispatched


def send_live_batch(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError(
        "HayPost live API is not configured yet. Use the export/mock workflow until you have their API schema."
    )
