"""
Token Tracking Wrapper for LightRAG

This module provides a wrapper around LightRAG that enables token tracking
without modifying the core LightRAG library. This ensures compatibility
with upstream updates.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional
import httpx
from lightrag import LightRAG
from lightrag.utils import TokenTracker, logger


class TokenTrackingLightRAG:
    """
    Wrapper around LightRAG that provides token tracking capabilities
    without modifying the core library.
    """

    def __init__(self, lightrag_instance: LightRAG):
        self._lightrag = lightrag_instance
        self._token_tracker: Optional[TokenTracker] = None

    def set_token_tracker(self, token_tracker: TokenTracker):
        """Set the token tracker to use for operations."""
        self._token_tracker = token_tracker

    def get_token_tracker(self) -> Optional[TokenTracker]:
        """Get the current token tracker."""
        return self._token_tracker

    def _create_tracked_llm_func(self, original_func: Callable) -> Callable:
        """Create a wrapped LLM function that tracks token usage."""
        if not self._token_tracker:
            return original_func

        @wraps(original_func)
        async def tracked_llm_func(*args, **kwargs):
            # Add token_tracker to kwargs if not already present
            kwargs["token_tracker"] = self._token_tracker
            return await original_func(*args, **kwargs)

        return tracked_llm_func

    async def _execute_with_tracking(self, method_name: str, *args, **kwargs) -> Any:
        """Execute a LightRAG method with token tracking."""
        if not self._token_tracker:
            # If no token tracker is set, just call the method directly
            method = getattr(self._lightrag, method_name)
            return await method(*args, **kwargs)

        # Store original LLM function
        original_llm_func = self._lightrag.llm_model_func

        try:
            # Replace with tracked version
            self._lightrag.llm_model_func = self._create_tracked_llm_func(
                original_llm_func
            )

            # Execute the method
            method = getattr(self._lightrag, method_name)
            result = await method(*args, **kwargs)

            return result

        finally:
            # Always restore the original function
            self._lightrag.llm_model_func = original_llm_func

    # Key methods that should be tracked
    async def ainsert(self, *args, **kwargs) -> str:
        """Insert documents with token tracking."""
        return await self._execute_with_tracking("ainsert", *args, **kwargs)

    async def aquery_llm(self, *args, **kwargs) -> dict[str, Any]:
        """Query with LLM with token tracking."""
        return await self._execute_with_tracking("aquery_llm", *args, **kwargs)

    async def apipeline_enqueue_documents(self, *args, **kwargs) -> None:
        """Enqueue documents for processing with token tracking."""
        return await self._execute_with_tracking(
            "apipeline_enqueue_documents", *args, **kwargs
        )

    async def apipeline_process_enqueue_documents(self, *args, **kwargs) -> None:
        """Process enqueued documents with token tracking and send metrics after completion."""
        from lightrag.base import DocStatus

        # Get documents that are about to be processed (pending, processing, failed)
        pending_docs, processing_docs, failed_docs = await asyncio.gather(
            self._lightrag.doc_status.get_docs_by_status(DocStatus.PENDING),
            self._lightrag.doc_status.get_docs_by_status(DocStatus.PROCESSING),
            self._lightrag.doc_status.get_docs_by_status(DocStatus.FAILED),
        )

        # Collect track_ids from documents that will be processed
        track_ids_before = set()
        for docs in [pending_docs, processing_docs, failed_docs]:
            for doc_id, doc_status in docs.items():
                if doc_status.track_id:
                    track_ids_before.add(doc_status.track_id)

        # Create a fresh token tracker for this processing session
        session_tracker = TokenTracker()
        self.set_token_tracker(session_tracker)

        try:
            # Execute the processing with tracking
            result = await self._execute_with_tracking(
                "apipeline_process_enqueue_documents", *args, **kwargs
            )

            # After processing completes, send metrics for the track_ids we processed
            await self._send_processing_metrics(session_tracker, track_ids_before)

            return result
        finally:
            # Clear the token tracker after processing
            self.set_token_tracker(None)

    async def _send_processing_metrics(
        self, token_tracker: TokenTracker, track_ids: set
    ) -> None:
        """Send metrics for processed documents with the given track_ids."""
        try:
            # Get token usage metrics
            usage = token_tracker.get_usage()

            # If we have metrics and track_ids, send them
            if usage.get("total_tokens", 0) > 0 and track_ids:
                logger.info(
                    f"📊 Sending metrics for {len(track_ids)} track_id(s): {', '.join(sorted(track_ids))}"
                )

                # Log the metrics in Langfuse style
                logger.info(f"\n{format_langfuse_style_usage(usage)}")

                # Send metrics for each track_id
                for track_id in track_ids:
                    success = await send_upload_metrics(track_id, usage)
                    if success:
                        logger.info(
                            f"✅ Successfully sent metrics for track_id: {track_id}"
                        )
                    else:
                        logger.warning(
                            f"⚠️  Failed to send metrics for track_id: {track_id}"
                        )
            else:
                logger.debug(
                    f"No metrics to send (tokens: {usage.get('total_tokens', 0)}, track_ids: {len(track_ids)})"
                )

        except Exception as e:
            logger.error(f"Error sending processing metrics: {str(e)}")
            # Don't raise - metrics sending should not break the main flow

    # Delegate all other attributes/methods to the wrapped instance
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped LightRAG instance."""
        return getattr(self._lightrag, name)

    # Context manager support for easy token tracking
    @asynccontextmanager
    async def track_tokens(self, token_tracker: Optional[TokenTracker] = None):
        """Context manager for token tracking."""
        if token_tracker:
            self.set_token_tracker(token_tracker)
        elif not self._token_tracker:
            self.set_token_tracker(TokenTracker())

        try:
            yield self
        finally:
            # Token tracker remains available for getting usage stats
            pass


# Convenience functions for easy usage
async def create_tracked_lightrag(lightrag_instance: LightRAG) -> TokenTrackingLightRAG:
    """Create a token-tracking wrapper for a LightRAG instance."""
    return TokenTrackingLightRAG(lightrag_instance)


@asynccontextmanager
async def track_lightrag_operations(
    lightrag_instance: LightRAG, token_tracker: Optional[TokenTracker] = None
):
    """Context manager for tracking token usage in LightRAG operations."""
    tracker = TokenTrackingLightRAG(lightrag_instance)

    async with tracker.track_tokens(token_tracker):
        yield tracker


# Backend integration functions
async def send_metrics_to_backend(
    track_id: str, metrics: Dict[str, Any], operation_type: str = "unknown"
) -> bool:
    """
    Send metrics to the backend billing system.

    Args:
        track_id: Tracking ID for the operation
        metrics: Token usage metrics dictionary
        operation_type: Type of operation (upload, query, etc.)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get environment variables
        api_base_url = os.getenv("BIZNDER_API_BASE_URL")
        service_token = os.getenv("BIZNDER_SERVICE_TOKEN")

        if not api_base_url or not service_token:
            logger.warning(
                "BIZNDER_API_BASE_URL or BIZNDER_SERVICE_TOKEN not configured, skipping metrics send"
            )
            return False

        # Prepare the webhook URL
        webhook_url = f"{api_base_url.rstrip('/')}/api/v1/webhooks/knowledge/metrics"

        # Calculate derived metrics matching Langfuse format
        prompt_tokens = metrics.get("prompt_tokens", 0)
        cached_tokens = metrics.get("cached_tokens", 0)
        completion_tokens = metrics.get("completion_tokens", 0)
        reasoning_tokens = metrics.get("output_reasoning_tokens", 0)
        input_regular = prompt_tokens - cached_tokens  # Uncached input tokens
        output_reasoning_tokens = reasoning_tokens
        output_regular = completion_tokens - reasoning_tokens  # Regular output tokens
        output_accepted_prediction_tokens = metrics.get(
            "output_accepted_prediction_tokens", 0
        )
        output_rejected_prediction_tokens = metrics.get(
            "output_rejected_prediction_tokens", 0
        )

        total_usage = metrics.get("total_tokens", 0)

        # Extract model name for billing (must be the actual model, not provider name)
        raw_model = metrics.get("model", "unknown")
        # Backend pricing lookup uses provider+model (e.g. openai/gpt-4o-mini). If the
        # tracker has "openai" or "unknown", use configured LLM model so we don't send
        # openai/openai and hit the fallback.
        if raw_model in (None, "", "unknown", "openai"):
            raw_model = os.getenv("LLM_MODEL", "unknown")
        # Never send provider name as model (would cause openai/openai and fallback pricing)
        if raw_model in (None, "", "openai"):
            raw_model = "unknown"
        model_name = raw_model or "unknown"

        # Determine provider from model name for pricing lookup
        provider = "openai"

        logger.info(
            f"Sending metrics to backend: provider={provider}, model={model_name}"
        )

        # Prepare the payload matching backend's expected format
        backend_metrics = {
            # Backend expected fields
            "provider": provider,
            "model": model_name,
            "inputTokens": prompt_tokens,
            "outputTokens": completion_tokens,
            "cacheReadTokens": cached_tokens,
            "reasoningTokens": reasoning_tokens,
            # Additional detailed fields for internal use
            "input": input_regular,
            "input_cached_tokens": metrics.get("input_cached_tokens", 0),
            "call_count": metrics.get("call_count", 0),
            "output": output_regular,
            "output_reasoning_tokens": output_reasoning_tokens,
            "output_accepted_prediction_tokens": output_accepted_prediction_tokens,
            "output_rejected_prediction_tokens": output_rejected_prediction_tokens,
            "total_usage": total_usage,
        }

        payload = {"track_id": track_id, "metrics": backend_metrics}

        # Send the request
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {service_token}",
                    "Content-Type": "application/json",
                },
            )

            if response.status_code in [200, 201, 202]:
                logger.info(
                    f"Successfully sent metrics for track_id {track_id} to backend"
                )
                return True
            else:
                logger.error(
                    f"Failed to send metrics for track_id {track_id}: HTTP {response.status_code} - {response.text}"
                )
                return False

    except Exception as e:
        logger.error(
            f"Error sending metrics for track_id {track_id} to backend: {str(e)}"
        )
        return False


async def send_upload_metrics(track_id: str, metrics: Dict[str, Any]) -> bool:
    """Send upload operation metrics to backend."""
    return await send_metrics_to_backend(track_id, metrics, "upload")


async def send_query_metrics(track_id: str, metrics: Dict[str, Any]) -> bool:
    """Send query operation metrics to backend."""
    return await send_metrics_to_backend(track_id, metrics, "query")


def format_langfuse_style_usage(usage: Dict[str, Any]) -> str:
    """
    Format token usage in Langfuse style for clear logging.

    Matches the exact format shown in Langfuse:
    Input usage
    5,323
    input_cached_tokens
    2,816
    input
    2,507
    ...
    """
    # Calculate derived metrics like Langfuse
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    reasoning_tokens = usage.get("output_reasoning_tokens", 0)

    # Input calculations
    input_usage = prompt_tokens
    input_cached_tokens = cached_tokens
    input_regular = prompt_tokens - cached_tokens  # Uncached input tokens
    input_audio_tokens = usage.get("input_audio_tokens", 0)

    # Output calculations
    output_usage = completion_tokens
    output_reasoning_tokens = reasoning_tokens
    output_regular = completion_tokens - reasoning_tokens  # Regular output tokens
    output_accepted_prediction_tokens = usage.get(
        "output_accepted_prediction_tokens", 0
    )
    output_audio_tokens = usage.get("output_audio_tokens", 0)
    output_rejected_prediction_tokens = usage.get(
        "output_rejected_prediction_tokens", 0
    )

    # Total usage
    total_usage = usage.get("total_tokens", 0)

    # Format like Langfuse with proper number formatting
    lines = [
        "Input usage",
        f"{input_usage:,}",
        "input_cached_tokens",
        f"{input_cached_tokens:,}",
        "input",
        f"{input_regular:,}",
        "input_audio_tokens",
        f"{input_audio_tokens:,}",
        "Output usage",
        f"{output_usage:,}",
        "output_reasoning_tokens",
        f"{output_reasoning_tokens:,}",
        "output",
        f"{output_regular:,}",
        "output_accepted_prediction_tokens",
        f"{output_accepted_prediction_tokens:,}",
        "output_audio_tokens",
        f"{output_audio_tokens:,}",
        "output_rejected_prediction_tokens",
        f"{output_rejected_prediction_tokens:,}",
        "Total usage",
        f"{total_usage:,}",
    ]

    return "\n".join(lines)
