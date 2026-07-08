from typing import Optional
from pydantic import BaseModel, Field


class FlowFeatures(BaseModel):
    """Single network flow record, matching NSL-KDD 41-feature schema."""
    duration: float = 0
    protocol_type: str = "tcp"
    service: str = "http"
    flag: str = "SF"
    src_bytes: float = 0
    dst_bytes: float = 0
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    num_outbound_cmds: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    count: int = 0
    srv_count: int = 0
    serror_rate: float = 0
    srv_serror_rate: float = 0
    rerror_rate: float = 0
    srv_rerror_rate: float = 0
    same_srv_rate: float = 0
    diff_srv_rate: float = 0
    srv_diff_host_rate: float = 0
    dst_host_count: int = 0
    dst_host_srv_count: int = 0
    dst_host_same_srv_rate: float = 0
    dst_host_diff_srv_rate: float = 0
    dst_host_same_src_port_rate: float = 0
    dst_host_srv_diff_host_rate: float = 0
    dst_host_serror_rate: float = 0
    dst_host_srv_serror_rate: float = 0
    dst_host_rerror_rate: float = 0
    dst_host_srv_rerror_rate: float = 0

    # optional metadata not used by the model, but useful for alert context
    src_ip: Optional[str] = Field(default="0.0.0.0")
    dst_ip: Optional[str] = Field(default="0.0.0.0")


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    severity: str
    class_probabilities: dict


class BatchSummary(BaseModel):
    total_rows: int
    predictions: dict
    alerts_generated: int
    report_available: bool


class AlertRecord(BaseModel):
    id: int
    timestamp: str
    src_ip: str
    dst_ip: str
    predicted_category: str
    confidence: float
    severity: str
