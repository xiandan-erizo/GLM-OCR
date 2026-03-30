"""Swagger/OpenAPI specifications for GLM-OCR API."""

SWAGGER_TEMPLATE = {
    "swagger": "2.0",
    "info": {
        "title": "GLM-OCR API",
        "version": "0.1.4",
        "description": "GLM OCR - Optical Character Recognition powered by GLM",
    },
    "basePath": "/",
    "consumes": ["application/json"],
    "produces": ["application/json"],
    "paths": {
        "/glmocr/parse": {
            "post": {
                "tags": ["OCR"],
                "summary": "Parse documents (images/PDFs)",
                "description": (
                    "Parse text from images or PDFs and return structured results.\n\n"
                    "**Supported image formats:**\n"
                    "- HTTP/HTTPS URL: `https://example.com/image.png`\n"
                    "- Local file path: `/path/to/image.png` or `file:///path/to/image.png`\n"
                    "- Base64 image: `data:image/png;base64,<base64_data>`\n"
                    "- Raw base64: `<base64_data>` or `<|base64|><base64_data>`\n\n"
                    "**Supported PDF formats:**\n"
                    "- HTTP/HTTPS URL: `https://example.com/doc.pdf`\n"
                    "- Local file path: `/path/to/doc.pdf` or `file:///path/to/doc.pdf`\n"
                    "- Base64 PDF: `data:application/pdf;base64,<base64_data>`\n"
                    "- Short format: `data:pdf;base64,<base64_data>`"
                ),
                "parameters": [
                    {
                        "name": "body",
                        "in": "body",
                        "required": True,
                        "schema": {
                            "type": "object",
                            "required": ["images"],
                            "properties": {
                                "images": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of image/PDF URLs or base64 data",
                                },
                            },
                        },
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Successful parse result",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "json_result": {
                                    "type": "object",
                                    "description": "Structured JSON result from OCR",
                                },
                                "markdown_result": {
                                    "type": "string",
                                    "description": "Markdown formatted text result",
                                },
                            },
                        },
                    },
                    "400": {
                        "description": "Bad request",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                            },
                        },
                        "examples": {
                            "application/json": {"error": "No images provided"},
                        },
                    },
                    "500": {
                        "description": "Server error",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "/health": {
            "get": {
                "tags": ["System"],
                "summary": "Health check",
                "description": "Check if the service is running and healthy",
                "responses": {
                    "200": {
                        "description": "Service is healthy",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                            },
                        },
                        "examples": {
                            "application/json": {"status": "ok"},
                        },
                    },
                },
            },
        },
    },
}