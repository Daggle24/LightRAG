"""
This module contains multimodal-specific routes for RAGAnything integration.

Provides endpoints for:
- Multimodal document processing (PDFs with images, tables, equations)
- VLM-enhanced queries with multimodal content
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field, field_validator

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger, generate_track_id

router = APIRouter(
    prefix="/multimodal",
    tags=["multimodal"],
)


# ============================================================================
# Request/Response Models
# ============================================================================


class MultimodalContentItem(BaseModel):
    """A single multimodal content item for queries"""

    type: Literal["image", "table", "equation"] = Field(
        description="Type of multimodal content"
    )
    # For images
    image_data: Optional[str] = Field(
        default=None, description="Base64-encoded image data"
    )
    image_url: Optional[str] = Field(default=None, description="URL of the image")
    image_caption: Optional[str] = Field(
        default=None, description="Caption for the image"
    )
    # For tables
    table_data: Optional[str] = Field(
        default=None, description="Table data as CSV or markdown"
    )
    table_caption: Optional[str] = Field(
        default=None, description="Caption for the table"
    )
    # For equations
    latex: Optional[str] = Field(default=None, description="LaTeX equation string")
    equation_caption: Optional[str] = Field(
        default=None, description="Caption for the equation"
    )


class MultimodalQueryRequest(BaseModel):
    """Request model for multimodal queries"""

    query: str = Field(min_length=3, description="The query text")
    multimodal_content: List[MultimodalContentItem] = Field(
        default_factory=list,
        description="List of multimodal content items to include in the query",
    )
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(
        default="hybrid", description="Query mode"
    )
    include_references: Optional[bool] = Field(
        default=True, description="Whether to include references in response"
    )

    @field_validator("query", mode="after")
    @classmethod
    def strip_query(cls, query: str) -> str:
        return query.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Explain this chart and compare with document data",
                "multimodal_content": [
                    {
                        "type": "table",
                        "table_data": "Name,Value\nA,100\nB,200",
                        "table_caption": "Sales data Q1",
                    }
                ],
                "mode": "hybrid",
            }
        }


class MultimodalQueryResponse(BaseModel):
    """Response model for multimodal queries"""

    response: str = Field(description="The generated response")
    references: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Source references"
    )


class MultimodalUploadRequest(BaseModel):
    """Options for multimodal document upload"""

    parse_method: Literal["auto", "ocr", "txt"] = Field(
        default="auto", description="Document parsing method"
    )
    enable_image_processing: bool = Field(
        default=True, description="Process images in the document"
    )
    enable_table_processing: bool = Field(
        default=True, description="Process tables in the document"
    )
    enable_equation_processing: bool = Field(
        default=True, description="Process equations in the document"
    )


class MultimodalUploadResponse(BaseModel):
    """Response model for multimodal document upload"""

    status: Literal["success", "processing", "failure"] = Field(
        description="Upload status"
    )
    message: str = Field(description="Status message")
    track_id: str = Field(description="Tracking ID for monitoring progress")
    file_name: Optional[str] = Field(default=None, description="Uploaded file name")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "processing",
                "message": "Document uploaded and processing started",
                "track_id": "mm_20250101_120000_abc123",
                "file_name": "document.pdf",
            }
        }


class MultimodalStatusResponse(BaseModel):
    """Response model for RAGAnything status"""

    enabled: bool = Field(description="Whether RAGAnything is enabled")
    vision_model: Optional[str] = Field(
        default=None, description="Configured vision model"
    )
    features: Dict[str, bool] = Field(
        description="Enabled multimodal features (images, tables, equations)"
    )


# ============================================================================
# Background Processing Functions
# ============================================================================


async def process_multimodal_document(
    rag_anything,
    temp_file: Path,
    output_dir: Path,
    parse_method: str,
    filename: str,
):
    """Background task to process a multimodal document
    
    Args:
        rag_anything: RAGAnything instance
        temp_file: Path to the temporary file
        output_dir: Output directory for processing results
        parse_method: Parsing method to use
        filename: Original filename for logging
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        await rag_anything.process_document_complete(
            file_path=str(temp_file),
            output_dir=str(output_dir),
            parse_method=parse_method,
        )

        logger.info(f"Multimodal document processing complete: {filename}")

        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()

    except Exception as e:
        logger.error(f"Error processing multimodal document: {str(e)}", exc_info=True)
        # Don't re-raise - this is a background task


# ============================================================================
# Route Factory
# ============================================================================


def create_multimodal_routes(
    create_raganything,
    create_rag,
    api_key: Optional[str] = None,
    args=None,
):
    """Create multimodal routes with dependency injection

    Args:
        create_raganything: Factory function to create RAGAnything instance
        create_rag: Factory function to create LightRAG instance
        api_key: Optional API key for authentication
        args: Server configuration arguments
    """
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/status",
        response_model=MultimodalStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_multimodal_status():
        """Get the status of RAGAnything multimodal features"""
        return MultimodalStatusResponse(
            enabled=args.enable_raganything if args else False,
            vision_model=args.vision_model if args else None,
            features={
                "images": args.enable_image_processing if args else False,
                "tables": args.enable_table_processing if args else False,
                "equations": args.enable_equation_processing if args else False,
            },
        )

    @router.post(
        "/query",
        response_model=MultimodalQueryResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {"description": "Successful multimodal query response"},
            400: {"description": "RAGAnything not enabled or invalid request"},
            500: {"description": "Query processing failed"},
        },
    )
    async def query_multimodal(
        raw_request: Request, request: MultimodalQueryRequest
    ):
        """
        Query with multimodal content using VLM-enhanced RAG.

        This endpoint allows you to include images, tables, or equations
        alongside your text query for context-aware responses.

        **Example Use Cases:**
        - Ask questions about a chart image in context of document data
        - Compare table data with information in your knowledge base
        - Explain mathematical equations relative to document content
        """
        if not args or not args.enable_raganything:
            raise HTTPException(
                status_code=400,
                detail="RAGAnything multimodal features are not enabled. Set ENABLE_RAGANYTHING=true",
            )

        try:
            rag_anything = await create_raganything(raw_request)

            # Convert multimodal content to RAGAnything format
            multimodal_items = []
            for item in request.multimodal_content:
                if item.type == "image":
                    multimodal_items.append(
                        {
                            "type": "image",
                            "image_data": item.image_data,
                            "image_url": item.image_url,
                            "image_caption": item.image_caption,
                        }
                    )
                elif item.type == "table":
                    multimodal_items.append(
                        {
                            "type": "table",
                            "table_data": item.table_data,
                            "table_caption": item.table_caption,
                        }
                    )
                elif item.type == "equation":
                    multimodal_items.append(
                        {
                            "type": "equation",
                            "latex": item.latex,
                            "equation_caption": item.equation_caption,
                        }
                    )

            # Use aquery_with_multimodal if multimodal content provided
            if multimodal_items:
                result = await rag_anything.aquery_with_multimodal(
                    request.query,
                    multimodal_content=multimodal_items,
                    mode=request.mode,
                )
            else:
                # Fall back to regular query
                result = await rag_anything.aquery(
                    request.query,
                    mode=request.mode,
                )

            # Handle different result formats
            if isinstance(result, dict):
                response_text = result.get("response", str(result))
                references = result.get("references", None)
            else:
                response_text = str(result)
                references = None

            return MultimodalQueryResponse(
                response=response_text,
                references=references if request.include_references else None,
            )

        except Exception as e:
            logger.error(f"Error in multimodal query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/upload",
        response_model=MultimodalUploadResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {"description": "Document uploaded for processing"},
            400: {"description": "RAGAnything not enabled or invalid file"},
            500: {"description": "Upload failed"},
        },
    )
    async def upload_multimodal_document(
        raw_request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        parse_method: str = "auto",
        enable_image_processing: bool = True,
        enable_table_processing: bool = True,
        enable_equation_processing: bool = True,
    ):
        """
        Upload and process a document with multimodal content extraction.

        This endpoint uses RAGAnything with MinerU to process documents
        containing images, tables, and equations. The processing happens
        in the background.

        **Supported File Types:**
        - PDF documents with embedded images/charts
        - Documents with tables and mathematical equations

        **Processing:**
        - Images are analyzed using the configured vision model
        - Tables are extracted and converted to structured format
        - Equations are parsed and contextualized
        """
        if not args or not args.enable_raganything:
            raise HTTPException(
                status_code=400,
                detail="RAGAnything multimodal features are not enabled. Set ENABLE_RAGANYTHING=true",
            )

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Check file extension
        allowed_extensions = {".pdf", ".docx", ".pptx"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {allowed_extensions}",
            )

        try:
            # Generate tracking ID
            track_id = generate_track_id("mm")

            # Save file temporarily
            temp_dir = Path(args.working_dir) / "multimodal_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"{track_id}_{file.filename}"

            content = await file.read()
            with open(temp_file, "wb") as f:
                f.write(content)

            # Get RAGAnything instance now (before request context ends)
            rag_anything = await create_raganything(raw_request)
            
            # Prepare output directory
            output_dir = Path(args.working_dir) / "multimodal_output" / track_id

            # Schedule background processing with all required args pre-resolved
            background_tasks.add_task(
                process_multimodal_document,
                rag_anything,
                temp_file,
                output_dir,
                parse_method,
                file.filename,
            )

            return MultimodalUploadResponse(
                status="processing",
                message=f"Document '{file.filename}' uploaded. Multimodal processing started in background.",
                track_id=track_id,
                file_name=file.filename,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error uploading multimodal document: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return router
