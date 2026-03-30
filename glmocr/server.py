"""GLM-OCR SDK Flask service."""

import os
import sys
import traceback
import multiprocessing
from typing import TYPE_CHECKING

from flask import Flask, request, jsonify


from flasgger import Swagger


from glmocr.pipeline import Pipeline
from glmocr.config import load_config
from glmocr.swagger_spec import SWAGGER_TEMPLATE
from glmocr.utils.logging import get_logger, configure_logging

if TYPE_CHECKING:
    from glmocr.config import GlmOcrConfig

logger = get_logger(__name__)

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def create_app(config: "GlmOcrConfig") -> Flask:
    """Create a Flask app.

    Args:
        config: GlmOcrConfig instance.

    Returns:
        Flask app instance.
    """

    app = Flask(__name__)

    if Swagger is not None:
        Swagger(app, template=SWAGGER_TEMPLATE)

    # Create pipeline with typed config
    pipeline = Pipeline(config=config.pipeline)

    # Store pipeline and config in app.config
    app.config["pipeline"] = pipeline
    app.config["doc_config"] = config

    @app.route("/glmocr/parse", methods=["POST"])
    def parse():
        """Parse documents (images/PDFs) and return OCR results."""
        logger.debug("[parse] Request received")

        # Validate Content-Type
        if request.headers.get("Content-Type") != "application/json":
            logger.warning(
                "[parse] Invalid Content-Type: %s", request.headers.get(
                    "Content-Type")
            )
            return (
                jsonify(
                    {"error": "Invalid Content-Type. Expected 'application/json'."}
                ),
                400,
            )

        # Parse JSON payload
        try:
            data = request.json
            logger.debug("[parse] JSON payload parsed")
        except Exception as e:
            logger.warning("[parse] Invalid JSON payload: %s", e)
            return jsonify({"error": "Invalid JSON payload"}), 400

        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]

        if not images:
            logger.warning("[parse] No images provided")
            return jsonify({"error": "No images provided"}), 400

        # Build pipeline request
        messages = [{"role": "user", "content": []}]
        for image_url in images:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        request_data = {"messages": messages}
        logger.debug("[parse] Request data built, calling pipeline.process()")

        try:
            # Pipeline.process() yields one result per input unit; merge for single response
            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                    layout_vis_output_dir=None,
                )
            )
            if not results:
                return (
                    jsonify({"json_result": None, "markdown_result": ""}),
                    200,
                )
            if len(results) == 1:
                r = results[0]
                return (
                    jsonify(
                        {
                            "json_result": r.json_result,
                            "markdown_result": r.markdown_result or "",
                        }
                    ),
                    200,
                )
            # Multiple units: merge json as list, markdown with separator
            logger.debug("[parse] Merging %d results", len(results))
            json_result = [r.json_result for r in results]
            markdown_result = "\n\n---\n\n".join(
                r.markdown_result or "" for r in results
            )
            return (
                jsonify(
                    {
                        "json_result": json_result,
                        "markdown_result": markdown_result,
                    }
                ),
                200,
            )

        except Exception as e:
            logger.error("Parse error: %s", e)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok"}), 200

    return app


def main():
    """Main entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="GlmOcr Server")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    # Use spawn for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    app = None

    try:
        config = load_config(args.config)

        # Configure logging
        log_level = args.log_level or config.logging.level
        configure_logging(level=log_level)

        # Create app with typed config
        app = create_app(config)

        # Start pipeline
        pipeline = app.config["pipeline"]
        pipeline.start()

        # Start Flask service
        server_config = config.server
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "GlmOcr Server starting on %s:%d...", server_config.host, server_config.port
        )
        logger.info("API endpoint: /glmocr/parse")
        logger.info("Swagger UI: /apidocs")
        logger.info("=" * 60)
        logger.info("")

        app.run(
            debug=server_config.debug,
            host=server_config.host,
            port=server_config.port,
        )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Error: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        # Stop pipeline
        if app is not None and "pipeline" in app.config:
            app.config["pipeline"].stop()


if __name__ == "__main__":
    main()
