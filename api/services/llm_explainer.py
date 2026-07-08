"""
Groq LLM service for generating plain-English explanations of NIDS alerts.

Usage:
    from api.services.llm_explainer import explain_alert
    text = explain_alert(alert_dict)   # raises RuntimeError if key not configured

Environment variables (loaded from .env automatically):
    GROQ_API_KEY   – required; obtain at https://console.groq.com
    GROQ_MODEL     – optional; defaults to llama-3.3-70b-versatile
"""
import os

from dotenv import load_dotenv

# Load .env from project root (safe no-op if .env doesn't exist)
load_dotenv()

_API_KEY: str | None = os.environ.get("GROQ_API_KEY", "").strip() or None
_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

_SYSTEM_PROMPT = (
    "You are a senior network security analyst. "
    "When given details about a network flow that was flagged by an intrusion "
    "detection system, you explain clearly and concisely (3-5 sentences) why "
    "the traffic is suspicious or malicious. "
    "Avoid technical jargon where possible; write for a security-aware but "
    "non-expert audience. Do NOT recommend actions — only explain the threat."
)


def _build_user_prompt(alert: dict) -> str:
    return (
        f"A network flow was detected and classified by our NIDS model.\n\n"
        f"  Source IP      : {alert.get('src_ip', 'unknown')}\n"
        f"  Destination IP : {alert.get('dst_ip', 'unknown')}\n"
        f"  Attack category: {alert.get('predicted_category', 'unknown')}\n"
        f"  Confidence     : {float(alert.get('confidence', 0)) * 100:.1f}%\n"
        f"  Severity       : {alert.get('severity', 'unknown')}\n\n"
        f"Please explain why this flow looks malicious."
    )


def explain_alert(alert: dict) -> str:
    """
    Call the Groq chat completions API and return a plain-English explanation.

    Args:
        alert: dict with keys src_ip, dst_ip, predicted_category, confidence, severity.

    Returns:
        Plain-English explanation string.

    Raises:
        RuntimeError: if GROQ_API_KEY is not set.
        Exception:    propagates any Groq API / network error (caller should catch).
    """
    if not _API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Set it in your .env file or as an environment variable. "
            "See .env.example for details."
        )

    # Import here so the module loads even when groq isn't installed (tests, etc.)
    try:
        from groq import Groq  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "The 'groq' package is not installed. Run: pip install groq"
        ) from exc

    client = Groq(api_key=_API_KEY)
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(alert)},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
