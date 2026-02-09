# LightRAG Token Tracking

This module provides a non-invasive token tracking solution for LightRAG that doesn't require modifying the core library. This ensures compatibility with upstream updates.

## Problem Solved

- **Maintainability**: No need to modify core LightRAG code, avoiding merge conflicts with upstream updates
- **Token Tracking**: Track LLM token usage for uploads, queries, and other operations
- **Backend Integration**: Send metrics to your main backend server for monitoring and billing

## Architecture

The solution uses a wrapper pattern:

1. **TokenTrackingLightRAG**: A wrapper class that intercepts LightRAG method calls
2. **LLM Function Wrapping**: Temporarily replaces the LLM function to inject token tracking
3. **Automatic Restoration**: Original functions are restored after operations complete

## Usage Examples

### Basic Usage

```python
from lightrag import LightRAG
from lightrag.utils import TokenTracker
from lightrag.api.token_tracking import TokenTrackingLightRAG

# Create your normal LightRAG instance
rag = LightRAG(...)

# Wrap it for token tracking
tracked_rag = TokenTrackingLightRAG(rag)
token_tracker = TokenTracker()
tracked_rag.set_token_tracker(token_tracker)

# Operations are automatically tracked
await tracked_rag.ainsert("document content")
result = await tracked_rag.aquery_llm("your question")

# Get usage metrics
usage = token_tracker.get_usage()
print(f"Tokens used: {usage}")
```

### Context Manager Usage

```python
from lightrag.api.token_tracking import track_lightrag_operations

async with track_lightrag_operations(rag, token_tracker) as tracked_rag:
    await tracked_rag.ainsert("content")
    result = await tracked_rag.aquery_llm("question")
# Token usage automatically tracked
```

### API Integration

The API server automatically uses token tracking for all operations:

#### Document Upload
```json
{
  "status": "success",
  "message": "File uploaded successfully",
  "track_id": "upload_123",
  "metrics": {
    "status": "processing",
    "model": "gpt-4o-mini",
    "note": "Token tracking active"
  }
}
```

Logged metrics after processing (Langfuse-style format):
```
Input usage
200
input_cached_tokens
50
input
150
input_audio_tokens
0
Output usage
150
output_reasoning_tokens
75
output
75
output_accepted_prediction_tokens
0
output_audio_tokens
0
output_rejected_prediction_tokens
0
Total usage
350
```

Complete internal metrics are also available:
```json
{
  "prompt_tokens": 200,
  "completion_tokens": 150,
  "total_tokens": 350,
  "cached_tokens": 50,
  "input_cached_tokens": 50,
  "input_audio_tokens": 0,
  "output_reasoning_tokens": 75,
  "output_audio_tokens": 0,
  "output_accepted_prediction_tokens": 0,
  "output_rejected_prediction_tokens": 0,
  "call_count": 2,
  "model": "gpt-4o-mini"
}
```

#### Query Endpoints
All query endpoints now include metrics in responses:

**POST /query**
```json
{
  "response": "AI-generated answer...",
  "references": [...],
  "metrics": {
    "prompt_tokens": 150,
    "completion_tokens": 200,
    "total_tokens": 350,
    "cached_tokens": 75,
    "input_cached_tokens": 75,
    "input_audio_tokens": 0,
    "output_reasoning_tokens": 125,
    "output_audio_tokens": 0,
    "output_accepted_prediction_tokens": 0,
    "output_rejected_prediction_tokens": 0,
    "call_count": 1,
    "model": "gpt-4o-mini"
  }
}
```

Logged in Langfuse-style format:
```
Input usage
150
input_cached_tokens
75
input
75
input_audio_tokens
0
Output usage
200
output_reasoning_tokens
125
output
75
output_accepted_prediction_tokens
0
output_audio_tokens
0
output_rejected_prediction_tokens
0
Total usage
350
```

**POST /query/data**
```json
{
  "status": "success",
  "data": {...},
  "metadata": {...},
  "metrics": {
    "prompt_tokens": 75,
    "completion_tokens": 50,
    "total_tokens": 125,
    "cached_tokens": 25,
    "call_count": 1,
    "model": "gpt-4o-mini"
  }
}
```

**POST /query/stream**
Metrics are logged automatically and can be sent to your backend server.

## Implementation Details

### How It Works

1. **Method Interception**: Key methods (`ainsert`, `aquery_llm`, etc.) are intercepted
2. **LLM Function Replacement**: The `llm_model_func` is temporarily replaced with a wrapped version
3. **Token Tracking**: Wrapped LLM functions include token tracking via the `token_tracker` parameter
4. **Automatic Cleanup**: Original functions are restored after operation completion

### Supported Operations

- Document insertion (`ainsert`)
- LLM queries (`aquery_llm`)
- Pipeline processing (`apipeline_enqueue_documents`, `apipeline_process_enqueue_documents`)
- Query endpoints (`/query`, `/query/data`, `/query/stream`)

### Detailed Token Tracking

The TokenTracker now extracts detailed token breakdowns from OpenAI's API, matching what Langfuse provides:

#### Available Token Fields
- **prompt_tokens**: Total input tokens
- **completion_tokens**: Total output tokens
- **total_tokens**: Sum of input and output tokens
- **cached_tokens**: Legacy field for cached tokens
- **input_cached_tokens**: Tokens served from cache (input)
- **input_audio_tokens**: Audio tokens in input (multimodal)
- **output_reasoning_tokens**: Reasoning tokens (o1 models)
- **output_audio_tokens**: Audio tokens in output (multimodal)
- **output_accepted_prediction_tokens**: Accepted prediction tokens
- **output_rejected_prediction_tokens**: Rejected prediction tokens

#### Usage Example
```python
tracker = TokenTracker()

# Automatic extraction from OpenAI API
# The system now automatically extracts all detailed fields

usage = tracker.get_usage()
print(f"Input tokens: {usage['prompt_tokens']}")
print(f"Cached tokens: {usage['input_cached_tokens']}")
print(f"Reasoning tokens: {usage['output_reasoning_tokens']}")
print(f"Cache hit rate: {usage['input_cached_tokens'] / usage['prompt_tokens'] * 100:.1f}%")
```

### Token Tracking Scope

Tokens are tracked for:
- Prompt tokens (input to LLM)
- Completion tokens (output from LLM)
- Total tokens
- API call count

## Backend Integration

All token tracking operations automatically send metrics to your backend billing system:

### Webhook Endpoint
```
POST /api/v1/webhooks/knowledge/metrics
Authorization: Bearer <BIZNDER_SERVICE_TOKEN>
```

### Environment Variables
- `BIZNDER_API_BASE_URL`: Base URL for your backend API
- `BIZNDER_SERVICE_TOKEN`: Service authentication token

### Payload Format (Complete with Langfuse-derived fields)
```json
{
  "track_id": "upload_1234567890_abc123",
  "metrics": {
    // Backend expected fields (primary)
    "provider": "openai",
    "model": "gpt-4o-mini",
    "inputTokens": 7648,
    "outputTokens": 7831,
    "cacheReadTokens": 5632,
    "reasoningTokens": 6528,

    // Additional detailed fields for internal use
    "input": 2016,             // = prompt_tokens - cached_tokens
    "input_cached_tokens": 5632,
    "call_count": 2,
    "output": 1303,            // = completion_tokens - output_reasoning_tokens
    "output_reasoning_tokens": 6528,
    "output_accepted_prediction_tokens": 0,
    "output_rejected_prediction_tokens": 0,
    "total_usage": 15479       // = total_tokens
  }
}
```

**Field Mapping:**
- `provider`: Always "openai" (auto-detected from model)
- `model`: The actual model name used (e.g., "gpt-4o-mini", "gpt-4o")
- `inputTokens`: Total input tokens (prompt_tokens)
- `outputTokens`: Total output tokens (completion_tokens)
- `cacheReadTokens`: Tokens served from cache
- `reasoningTokens`: Reasoning tokens for o1/o3 models

### Automatic Sending
- **Upload operations**: Metrics sent automatically after `apipeline_process_enqueue_documents()` completes
  - The system tracks which documents were processed in the session
  - Collects token usage from all LLM calls during processing
  - Sends metrics for each unique `track_id` that was processed
  - Logs metrics in Langfuse-style format for visibility
- **Query operations**: Metrics sent after each query completes
- **Error handling**: Failures are logged but don't break the main operation

### How Upload Metrics Work

When documents are processed via `apipeline_process_enqueue_documents()`:

1. **Before Processing**: System identifies all documents that will be processed (pending, processing, failed statuses) and collects their `track_id` values
2. **During Processing**: Token usage is tracked via `TokenTracker` for all LLM calls
3. **After Processing**: 
   - Metrics are logged in Langfuse-style format
   - `send_upload_metrics()` is called for each unique `track_id`
   - Success/failure is logged for each metric transmission

Example log output:
```
INFO: Enqueued document processing pipeline stopped
INFO: 📊 Sending metrics for 1 track_id(s): upload_20250208_123456_abc123
INFO: 
Input usage
7648
input_cached_tokens
5632
input
2016
...
INFO: ✅ Successfully sent metrics for track_id: upload_20250208_123456_abc123
```

## Benefits

- ✅ **Zero Core Modifications**: Works with any LightRAG version
- ✅ **Automatic Tracking**: No manual token counting needed
- ✅ **Flexible Integration**: Easy to add to existing codebases
- ✅ **Backend Ready**: Metrics formatted for server transmission
- ✅ **Error Safe**: Automatic cleanup prevents function corruption

## Files

- `lightrag/api/token_tracking.py`: Core wrapper implementation
- `examples/token_tracking_example.py`: Usage examples
- `lightrag/api/lightrag_server.py`: API server integration (modified)
- `lightrag/api/routers/document_routes.py`: Document upload integration (modified)
