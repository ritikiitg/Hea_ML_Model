"""
Security Middleware — rate limiting, request validation, and security headers.
"""

import time
import logging
from collections import defaultdict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Provides rate limiting, request size validation, and security headers."""

    def __init__(self, app):
        super().__init__(app)
        self._request_counts: dict[str, list] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # 1. Rate limiting
        client_ip = request.client.host if request.client else "unknown"
        if not self._check_rate_limit(client_ip):
            return Response(
                content='{"detail":"Rate limit exceeded. Please try again later."}',
                status_code=429,
                media_type="application/json",
            )

        # 2. Request size validation (prevent oversized payloads)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1_000_000:  # 1MB max
            return Response(
                content='{"detail":"Request payload too large"}',
                status_code=413,
                media_type="application/json",
            )

        # 3. Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # 4. Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Process-Time"] = str(round(process_time, 4))

        # 5. Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"time={process_time:.3f}s "
            f"ip={client_ip}"
        )

        return response

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Simple sliding window rate limiter."""
        now = time.time()
        window = 60  # 1 minute window

        # Clean old entries
        self._request_counts[client_ip] = [
            t for t in self._request_counts[client_ip] if now - t < window
        ]

        if len(self._request_counts[client_ip]) >= settings.RATE_LIMIT_PER_MINUTE:
            return False

        self._request_counts[client_ip].append(now)
        return True
