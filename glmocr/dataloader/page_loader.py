"""Page loader - unified document/image loading and preprocessing.

PageLoader is responsible for:
1. Loading various input formats (images, PDFs, base64, URLs)
2. Converting inputs into a list of PIL Images
3. Building OCR API request payloads

Supported inputs:
- Local file paths (images, PDFs)
- file:// URLs
- data:image/... base64 URLs
"""

from __future__ import annotations

import os
import base64
import time
import tempfile
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple, Union

from PIL import Image
import requests
from requests.adapters import HTTPAdapter


from glmocr.utils.image_utils import (
    load_image_to_base64,
    pdf_to_images_pil,
    pdf_to_images_pil_iter,
    PYPDFIUM2_AVAILABLE,
)
from glmocr.utils.logging import get_logger, get_profiler

Image.MAX_IMAGE_PIXELS = None
if TYPE_CHECKING:
    from glmocr.config import PageLoaderConfig

logger = get_logger(__name__)
profiler = get_profiler(__name__)


@dataclass
class _PrefetchedSource:
    """Materialized source ready to be consumed into page images."""

    source: str
    is_pdf: bool = False
    file_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    content_type: str = ""
    cleanup_path: Optional[str] = None


class PageLoader:
    """Page loader.

    Unifies image and PDF inputs and provides two output modes:
    1. load_pages(): returns a list of PIL Images (for layout detection or custom logic)
    2. build_request(): builds OCR API request payloads

    Example:
        from glmocr.config import PageLoaderConfig

        loader = PageLoader(PageLoaderConfig())

        # Load as PIL images
        pages = loader.load_pages(["doc.pdf", "image.png"])

        # Build an API request
        request_data = loader.build_request(request_data)
    """

    def __init__(self, config: "PageLoaderConfig"):
        """Initialize.

        Args:
            config: PageLoaderConfig instance.
        """
        self.config = config

        # Image processing parameters
        self.t_patch_size = config.t_patch_size
        self.patch_expand_factor = config.patch_expand_factor
        self.image_expect_length = config.image_expect_length
        self.image_format = config.image_format
        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels

        # API request parameters
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        # Task prompt mapping
        self.task_prompt_mapping = config.task_prompt_mapping

        # Default OCR instruction (used when user provides images without text)
        self.default_prompt = config.default_prompt

        # PDF-to-image parameters (pypdfium2 only)
        self.pdf_dpi = config.pdf_dpi
        self.pdf_max_pages = config.pdf_max_pages
        self.pdf_verbose = config.pdf_verbose
        self.download_connect_timeout = config.download_connect_timeout
        self.download_read_timeout = config.download_read_timeout
        self.download_max_size_bytes = max(
            1, int(float(config.download_max_size_mb) * 1024 * 1024)
        )
        self.remote_download_workers = max(1, int(config.remote_download_workers))
        # Reuse HTTP connections for remote sources. Keep the pool bounded by
        # the configured source prefetch parallelism.
        self._pool_maxsize = max(4, self.remote_download_workers)
        self._session: Optional[requests.Session] = None

    # =========================================================================
    # Page loading
    # =========================================================================

    def load_pages(self, sources: Union[str, List[str]]) -> List[Image.Image]:
        """Load sources into a list of PIL Images.

        Supports image files and PDFs (PDFs are expanded into multiple pages).

        Args:
            sources: Single path/URL or a list.

        Returns:
            List[PIL.Image.Image]
        """
        if isinstance(sources, str):
            sources = [sources]

        all_pages = []
        for prefetched in self._iter_prefetched_sources_in_order(sources):
            pages = self._load_prefetched_source(prefetched)
            all_pages.extend(pages)

        return all_pages

    def load_pages_with_unit_indices(
        self, sources: Union[str, List[str]]
    ) -> Tuple[List[Image.Image], List[int]]:
        """Load sources into pages and return unit index per page.

        Each input URL is one "unit". For a PDF, all its pages share the same
        unit index. Used by streaming mode to yield one result per input unit.

        Args:
            sources: Single path/URL or a list.

        Returns:
            (all_pages, unit_indices) where unit_indices[i] is the unit index
            of page i (i.e. which input URL it came from).
        """
        if isinstance(sources, str):
            sources = [sources]

        all_pages: List[Image.Image] = []
        unit_indices: List[int] = []
        for unit_idx, prefetched in enumerate(
            self._iter_prefetched_sources_in_order(sources)
        ):
            pages = self._load_prefetched_source(prefetched)
            all_pages.extend(pages)
            unit_indices.extend([unit_idx] * len(pages))
        return all_pages, unit_indices

    def iter_pages_with_unit_indices(self, sources: Union[str, List[str]]):
        """Stream pages one at a time with unit index per page.

        Yields (page, unit_idx) so the pipeline can enqueue each page as soon
        as it is rendered (e.g. PDF: render one page → yield → next page).

        Args:
            sources: Single path/URL or a list.

        Yields:
            (PIL.Image, unit_idx) for each page.
        """
        if isinstance(sources, str):
            sources = [sources]
        for unit_idx, prefetched in enumerate(
            self._iter_prefetched_sources_in_order(sources)
        ):
            for page in self._iter_prefetched_source(prefetched):
                yield page, unit_idx

    def _iter_source(self, source: str):
        """Yield pages from a single source one at a time."""
        yield from self._iter_prefetched_source(self._prefetch_source(source))

    def _compute_end_page(self) -> Optional[int]:
        """Parse pdf_max_pages into 0-based inclusive end page index, or None for last page."""
        if self.pdf_max_pages is None:
            return None
        try:
            mp = int(self.pdf_max_pages)
            if mp > 0:
                return mp - 1  # 0-based inclusive
        except Exception:
            pass
        return None

    def _iter_pdf(self, file_path: str):
        """Yield PDF pages one at a time (streaming)."""
        if not PYPDFIUM2_AVAILABLE:
            raise RuntimeError(
                "PDF support requires pypdfium2. Install: pip install pypdfium2"
            )
        logger.debug("[page_loader] Loading PDF: %s", file_path)
        end_page = self._compute_end_page()
        page_idx = 0
        for image in pdf_to_images_pil_iter(
            file_path,
            dpi=self.pdf_dpi,
            max_width_or_height=3500,
            start_page_id=0,
            end_page_id=end_page,
        ):
            logger.debug("[page_loader] Rendered PDF page %d", page_idx)
            page_idx += 1
            yield image
        logger.debug("[page_loader] PDF loading complete, %d page(s)", page_idx)

    def _load_source(self, source: str) -> List[Image.Image]:
        """Load a single source and return a list of pages.

        PDFs return all pages; images return a single-page list.
        """
        return self._load_prefetched_source(self._prefetch_source(source))

    def _load_prefetched_source(
        self, prefetched: _PrefetchedSource
    ) -> List[Image.Image]:
        return list(self._iter_prefetched_source(prefetched))

    def _iter_prefetched_source(self, prefetched: _PrefetchedSource):
        try:
            if prefetched.is_pdf:
                yield from self._iter_pdf(prefetched.file_path)
            elif prefetched.image_bytes is not None:
                yield self._load_image_from_bytes(
                    prefetched.image_bytes, prefetched.source
                )
            else:
                yield self._load_image(prefetched.file_path or prefetched.source)
        finally:
            if prefetched.cleanup_path:
                try:
                    os.unlink(prefetched.cleanup_path)
                except OSError:
                    logger.warning(
                        "Failed to remove temporary PDF file: %s",
                        prefetched.cleanup_path,
                    )

    def _iter_prefetched_sources_in_order(self, sources: List[str]):
        if self.remote_download_workers <= 1 or len(sources) <= 1:
            for source in sources:
                yield self._prefetch_source(source)
            return

        max_workers = min(self.remote_download_workers, len(sources))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: Dict[int, Future[_PrefetchedSource]] = {}
            next_submit = 0

            while next_submit < max_workers:
                pending[next_submit] = executor.submit(
                    self._prefetch_source, sources[next_submit]
                )
                next_submit += 1

            for idx in range(len(sources)):
                prefetched = pending.pop(idx).result()
                if next_submit < len(sources):
                    pending[next_submit] = executor.submit(
                        self._prefetch_source, sources[next_submit]
                    )
                    next_submit += 1
                yield prefetched

    def _prefetch_source(self, source: str) -> _PrefetchedSource:
        # Handle data:application/pdf;base64,... or data:pdf;base64,... format
        if source.startswith("data:application/pdf") or source.startswith("data:pdf"):
            try:
                header, base64_data = source.split(",", 1)
                pdf_bytes = base64.b64decode(base64_data)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name
                logger.debug(
                    "[page_loader] Base64 PDF decoded to temp file: %s (%d bytes)",
                    tmp_path,
                    len(pdf_bytes),
                )
                return _PrefetchedSource(
                    source=source,
                    is_pdf=True,
                    file_path=tmp_path,
                    cleanup_path=tmp_path,
                )
            except Exception as e:
                raise RuntimeError(f"Error decoding base64 PDF: {e}") from e

        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            return _PrefetchedSource(
                source=source,
                is_pdf=True,
                file_path=file_path,
            )
        if self._is_http_url(source):
            return self._download_remote_source(source)

        return _PrefetchedSource(
            source=source,
            is_pdf=False,
            file_path=file_path if os.path.isfile(file_path) else None,
        )

    def _load_image(self, source: str) -> Image.Image:
        """Load a single image."""
        try:
            # data:image/... URL
            if source.startswith("data:image"):
                header, base64_data = source.split(",", 1)
                image_data = base64.b64decode(base64_data)
                return Image.open(BytesIO(image_data))

            elif source.startswith("file://"):
                return Image.open(source[7:])

            # Local file
            elif os.path.isfile(source):
                return Image.open(source)
            elif self._is_http_url(source):
                prefetched = self._download_remote_source(source)
                if prefetched.image_bytes is None:
                    raise RuntimeError(
                        f"Remote image source '{source}' was resolved as a PDF"
                    )
                return self._load_image_from_bytes(prefetched.image_bytes, source)

            else:
                raise ValueError(f"Invalid image source: {source}")

        except Exception as e:
            raise RuntimeError(f"Error loading image '{source}': {e}")

    @staticmethod
    def _is_http_url(source: str) -> bool:
        return source.startswith("http://") or source.startswith("https://")

    @staticmethod
    def _is_remote_pdf(source: str, content_type: str) -> bool:
        source_no_query = source.split("?", 1)[0].split("#", 1)[0].lower()
        return source_no_query.endswith(".pdf") or "application/pdf" in content_type

    @staticmethod
    def _load_image_from_bytes(image_bytes: bytes, source: str) -> Image.Image:
        try:
            return Image.open(BytesIO(image_bytes))
        except Exception as e:
            raise RuntimeError(f"Error loading image '{source}': {e}") from e

    def _get_session(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=1,
                pool_maxsize=self._pool_maxsize,
                max_retries=0,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._session = session
        return self._session

    def _validate_response_size(
        self, source: str, content_length: Optional[str]
    ) -> None:
        if not content_length:
            return
        try:
            expected_size = int(content_length)
        except (TypeError, ValueError):
            return
        if expected_size > self.download_max_size_bytes:
            raise RuntimeError(
                f"Remote source '{source}' exceeds size limit "
                f"({expected_size} bytes > {self.download_max_size_bytes} bytes)"
            )

    def _download_remote_source(self, source: str) -> _PrefetchedSource:
        try:
            logger.info("Downloading remote source: %s", source)
            with self._get_session().get(
                source,
                timeout=(self.download_connect_timeout, self.download_read_timeout),
                stream=True,
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                self._validate_response_size(
                    source, response.headers.get("Content-Length")
                )

                if self._is_remote_pdf(source, content_type):
                    prefetched = self._download_remote_pdf_to_tempfile(
                        source, response, content_type
                    )
                else:
                    prefetched = self._download_remote_image_to_memory(
                        source, response, content_type
                    )
            logger.info("Remote source downloaded: %s", source)
            return prefetched
        except Exception as e:
            raise RuntimeError(f"Error downloading remote source '{source}': {e}") from e

    def _download_remote_image_to_memory(
        self,
        source: str,
        response: requests.Response,
        content_type: str,
    ) -> _PrefetchedSource:
        total_size = 0
        chunks: List[bytes] = []
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total_size += len(chunk)
            if total_size > self.download_max_size_bytes:
                raise RuntimeError(
                    f"Remote source '{source}' exceeds size limit "
                    f"({total_size} bytes > {self.download_max_size_bytes} bytes)"
                )
            chunks.append(chunk)
        return _PrefetchedSource(
            source=source,
            is_pdf=False,
            image_bytes=b"".join(chunks),
            content_type=content_type,
        )

    def _download_remote_pdf_to_tempfile(
        self,
        source: str,
        response: requests.Response,
        content_type: str,
    ) -> _PrefetchedSource:
        total_size = 0
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total_size += len(chunk)
                    if total_size > self.download_max_size_bytes:
                        raise RuntimeError(
                            f"Remote source '{source}' exceeds size limit "
                            f"({total_size} bytes > {self.download_max_size_bytes} bytes)"
                        )
                    tmp.write(chunk)
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.warning("Failed to remove temporary PDF file: %s", tmp_path)
            raise
        logger.debug(
            "[page_loader] Downloaded PDF to temp file %s (%d bytes)",
            tmp_path,
            total_size,
        )
        return _PrefetchedSource(
            source=source,
            is_pdf=True,
            file_path=tmp_path,
            content_type=content_type,
            cleanup_path=tmp_path,
        )

    def _load_pdf(self, file_path: str) -> List[Image.Image]:
        """Load all pages from a PDF file using pypdfium2 (required)."""
        if not PYPDFIUM2_AVAILABLE:
            raise RuntimeError(
                "PDF support requires pypdfium2. Install: pip install pypdfium2"
            )
        logger.debug("[page_loader] Loading PDF: %s", file_path)
        t0 = time.perf_counter()
        end_page = self._compute_end_page()
        pages = pdf_to_images_pil(
            file_path,
            dpi=self.pdf_dpi,
            max_width_or_height=3500,
            start_page_id=0,
            end_page_id=end_page,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("[page_loader] PDF loaded: %d page(s) in %.1fms", len(pages), elapsed)
        profiler.log(
            f"pdf_to_images_pil({os.path.basename(file_path)})",
            elapsed,
        )
        return pages

    # =========================================================================
    # API request building
    # =========================================================================

    def build_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build an OCR API request from request_data.

        Args:
            request_data: Raw request data containing messages.

        Returns:
            Updated request data.
        """
        # Set default parameters
        if "max_tokens" not in request_data:
            request_data["max_tokens"] = self.max_tokens
        if "top_p" not in request_data:
            request_data["top_p"] = self.top_p
        if "temperature" not in request_data:
            request_data["temperature"] = self.temperature
        if "top_k" not in request_data:
            request_data["top_k"] = self.top_k
        if "repetition_penalty" not in request_data:
            request_data["repetition_penalty"] = self.repetition_penalty

        # Process messages
        messages = request_data["messages"]
        processed_messages = []

        for msg in messages:
            if msg["role"] in ("system", "assistant", "tool"):
                processed_messages.append(msg)
            elif msg["role"] in ("user", "observation"):
                # If user provides images but no text, inject the default OCR instruction
                if isinstance(msg.get("content"), list):
                    has_image = any(
                        c.get("type") == "image_url" for c in msg["content"]
                    )
                    has_text = any(
                        c.get("type") == "text" and str(c.get("text", "")).strip()
                        for c in msg["content"]
                    )
                    if has_image and not has_text:
                        msg = {
                            **msg,
                            "content": [
                                *msg["content"],
                                {"type": "text", "text": self.default_prompt},
                            ],
                        }

                processed_messages.append(self._process_msg_standard(msg))
            else:
                raise ValueError(f"{msg['role']} is not a valid role for a message.")

        request_data["messages"] = processed_messages
        return request_data

    def build_request_from_image(
        self, image: Image.Image, task_type: str = "text"
    ) -> Dict[str, Any]:
        """Build an API request from a PIL Image.

        Args:
            image: PIL Image.
            task_type: Task type (text/table/formula, etc.).

        Returns:
            Full API request payload.
        """
        prompt_text = ""
        if self.task_prompt_mapping:
            prompt_text = self.task_prompt_mapping.get(task_type, "")
        if not str(prompt_text).strip():
            prompt_text = self.default_prompt

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Encode image
        buffered = BytesIO()
        image.save(buffered, format=self.image_format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        original_msg = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{self.image_format.lower()};base64,{img_base64}"
                    },
                },
            ],
        }

        if prompt_text:
            original_msg["content"].append({"type": "text", "text": prompt_text})

        processed_msg = self._process_msg_standard(original_msg)

        return {
            "messages": [processed_msg],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

    def _process_msg_standard(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Standard mode: encode images inside a message."""
        msg_content: List[Dict] = msg["content"]
        processed_content = []

        for content in msg_content:
            if content["type"] == "text":
                processed_content.append(content)
            elif content["type"] == "image_url":
                image_url = content["image_url"]["url"]
                with profiler.measure("load_image_to_base64"):
                    encoded_image = load_image_to_base64(
                        image_url,
                        t_patch_size=self.t_patch_size,
                        max_pixels=self.max_pixels,
                        image_format=self.image_format,
                        patch_expand_factor=self.patch_expand_factor,
                        min_pixels=self.min_pixels,
                    )
                processed_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{self.image_format.lower()};base64,{encoded_image}"
                        },
                    }
                )
            else:
                raise ValueError(f"{content['type']} is not a valid type.")

        return {"role": msg["role"], "content": processed_content}
