from fastapi import Request
from typing import Dict, Any
import logging
from ..utils.data_retention import data_retention_policy

logger = logging.getLogger(__name__)

class PrivacyMiddleware:
    def __init__(self):
        pass

    async def process_request(self, request: Request, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request data to ensure privacy compliance
        """
        # Remove any potential PII from the request data
        cleaned_data = data_retention_policy.enforce_privacy_policy(data)

        # Log minimal information for debugging without compromising privacy
        logger.info(f"Processing request for endpoint: {request.url.path}")

        return cleaned_data

    async def process_response(self, request: Request, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process response data to ensure privacy compliance
        """
        # Ensure no sensitive data is included in the response
        # This is particularly important for error responses
        if "error" in response and "details" in response["error"]:
            # Remove detailed error information that might contain sensitive data
            response["error"]["details"] = "Error details removed for privacy"

        return response

# Global privacy middleware instance
privacy_middleware = PrivacyMiddleware()