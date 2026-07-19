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

CODE_TTL_SECONDS = 10 * 60          # 인증 코드 유효 시간 (10분)
MIN_PASSWORD_LENGTH = 8

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or "587")
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER).strip()

EMAIL_SENDING_CONFIGURED = bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)


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


def send_verification_email(to_email: str, code: str) -> bool:
    """실제 발송 성공 시 True. SMTP 미설정이거나 실패하면 False
    (호출자는 False 일 때 코드를 화면에 직접 보여주는 데모 폴백을 써야 한다)."""
    if not EMAIL_SENDING_CONFIGURED:
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = "[LinguaLoop] 이메일 인증 코드"
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        msg.set_content(
            f"LinguaLoop 회원가입 인증 코드입니다: {code}\n"
            f"{CODE_TTL_SECONDS // 60}분 이내에 입력해 주세요."
        )
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:  # pragma: no cover
        print(f"[auth] 이메일 발송 실패 → 데모 폴백 사용: {e}")
        return False
