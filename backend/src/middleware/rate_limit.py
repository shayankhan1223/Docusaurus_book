from fastapi import Request, HTTPException
from typing import Dict
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int = 100, window: int = 3600):  # 100 requests per hour
        self.max_requests = max_requests
        self.window = window  # in seconds
        self.requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        # Remove old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window
        ]

        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        return False

# Global rate limiter instance
rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request):
    # Use client IP as identifier (in production, you might use API keys)
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return