"""
auth.py — 이메일+비밀번호 회원가입/로그인 + 이메일 인증
"""
from __future__ import annotations

import hashlib
import hmac
import os
import random
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional, Tuple

import requests

CODE_TTL_SECONDS = 10 * 60          # 인증 코드 유효 시간 (10분)
MIN_PASSWORD_LENGTH = 8

# ─────────────────────────────────────────────────────────────
# 이메일 발송 설정
#   Render 무료 티어는 아웃바운드 SMTP 포트(25/465/587)를 차단하므로,
#   HTTP(443) 기반 이메일 API(Resend/Brevo)를 우선 사용한다.
#   provider 자동 판별 순서: resend → brevo → smtp → (미설정: 데모 폴백)
#   EMAIL_PROVIDER 로 강제 지정도 가능 ("resend"|"brevo"|"smtp").
# ─────────────────────────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or "587")
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER).strip()

RESEND_API_KEY = os.getenv("RESEND_API_KEY", "").strip()
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "").strip()
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "").strip().lower()
# 발신 주소 — HTTP API 는 검증된 발신자/도메인이 필요. (Resend 미검증 시 onboarding@resend.dev)
EMAIL_FROM = (os.getenv("EMAIL_FROM", "") or SMTP_FROM or SMTP_USER).strip()
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "LinguaLoop").strip()


def _resolve_email_provider() -> str:
    if EMAIL_PROVIDER in ("resend", "brevo", "smtp"):
        return EMAIL_PROVIDER
    if RESEND_API_KEY:
        return "resend"
    if BREVO_API_KEY:
        return "brevo"
    if SMTP_HOST and SMTP_USER and SMTP_PASSWORD:
        return "smtp"
    return ""


ACTIVE_EMAIL_PROVIDER = _resolve_email_provider()
EMAIL_SENDING_CONFIGURED = bool(ACTIVE_EMAIL_PROVIDER)


# ─────────────────────────────────────────────────────────────
# 비밀번호 해싱
# ─────────────────────────────────────────────────────────────
def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """"salt_hex$hash_hex" 형식의 문자열로 반환."""
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split("$", 1)
    except ValueError:  # pragma: no cover
        return False
    salt = bytes.fromhex(salt_hex)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(candidate.hex(), hash_hex)


def password_is_strong_enough(password: str) -> Tuple[bool, str]:
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"비밀번호는 최소 {MIN_PASSWORD_LENGTH}자 이상이어야 해요."
    return True, ""


# ─────────────────────────────────────────────────────────────
# 인증 코드
# ─────────────────────────────────────────────────────────────
def generate_verification_code() -> str:
    return f"{random.randint(0, 999_999):06d}"


def _email_subject_body(code: str) -> Tuple[str, str]:
    subject = "[LinguaLoop] 이메일 인증 코드"
    text = (
        f"LinguaLoop 회원가입 인증 코드입니다: {code}\n"
        f"{CODE_TTL_SECONDS // 60}분 이내에 입력해 주세요."
    )
    return subject, text


def _send_via_resend(to_email: str, code: str) -> bool:
    """Resend HTTP API (https://api.resend.com/emails) — 포트 443, SMTP 차단 우회."""
    subject, text = _email_subject_body(code)
    from_addr = EMAIL_FROM or "onboarding@resend.dev"  # 도메인 미검증 시 테스트 발신자
    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}",
                 "Content-Type": "application/json"},
        json={"from": f"{EMAIL_FROM_NAME} <{from_addr}>",
              "to": [to_email], "subject": subject, "text": text},
        timeout=15,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Resend {r.status_code}: {r.text[:200]}")
    return True


def _send_via_brevo(to_email: str, code: str) -> bool:
    """Brevo(구 Sendinblue) HTTP API — 포트 443, SMTP 차단 우회."""
    subject, text = _email_subject_body(code)
    if not EMAIL_FROM:
        raise RuntimeError("EMAIL_FROM(검증된 발신자 주소)이 설정되지 않았습니다.")
    r = requests.post(
        "https://api.brevo.com/v3/smtp/email",
        headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json",
                 "accept": "application/json"},
        json={"sender": {"name": EMAIL_FROM_NAME, "email": EMAIL_FROM},
              "to": [{"email": to_email}], "subject": subject, "textContent": text},
        timeout=15,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Brevo {r.status_code}: {r.text[:200]}")
    return True


def _send_via_smtp(to_email: str, code: str) -> bool:
    """전통적 SMTP (로컬/유료 인스턴스용 — Render 무료 티어에선 차단됨)."""
    subject, text = _email_subject_body(code)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = to_email
    msg.set_content(text)
    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
        server.starttls(context=context)
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
    return True


def send_verification_email(to_email: str, code: str) -> bool:
    """실제 발송 성공 시 True. 미설정/실패면 False
    (호출자는 False 일 때 코드를 화면에 직접 보여주는 데모 폴백을 써야 한다)."""
    provider = ACTIVE_EMAIL_PROVIDER
    if not provider:
        return False
    try:
        if provider == "resend":
            return _send_via_resend(to_email, code)
        if provider == "brevo":
            return _send_via_brevo(to_email, code)
        if provider == "smtp":
            return _send_via_smtp(to_email, code)
    except Exception as e:  # pragma: no cover
        print(f"[auth] 이메일 발송 실패({provider}) → 데모 폴백 사용: {e}")
        return False
    return False
