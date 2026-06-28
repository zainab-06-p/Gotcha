"""
Gotcha — Data Loading Module
Handles loading candidates from JSON/JSONL and extracting JD text from .docx.
"""

import json
import logging
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)


def load_sample_candidates(path: Optional[str] = None) -> list[dict]:
    """Load sample candidates from a JSON file.

    Args:
        path: Path to sample_candidates.json. Defaults to config path.

    Returns:
        List of candidate dicts.
    """
    if path is None:
        from src.config import SAMPLE_CANDIDATES_JSON
        path = SAMPLE_CANDIDATES_JSON

    path = Path(path)
    if not path.exists():
        logger.error("Sample candidates file not found: %s", path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        logger.info("Loaded %d sample candidates from %s", len(candidates), path)
        return candidates
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("Error parsing sample candidates file: %s", e)
        # Try with latin-1 fallback
        try:
            with open(path, "r", encoding="latin-1") as f:
                candidates = json.load(f)
            logger.info("Loaded %d sample candidates (latin-1 fallback)", len(candidates))
            return candidates
        except Exception as e2:
            logger.error("Failed with fallback encoding too: %s", e2)
            return []


def stream_candidates_jsonl(
    path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Generator[dict, None, None]:
    """Stream candidates one at a time from a JSONL file.

    Memory-efficient generator — never loads the full file.

    Args:
        path: Path to candidates.jsonl. Defaults to config path.
        limit: Maximum number of candidates to yield. None = all.

    Yields:
        One candidate dict per line.
    """
    if path is None:
        from src.config import CANDIDATES_JSONL
        path = CANDIDATES_JSONL

    path = Path(path)
    if not path.exists():
        logger.error("JSONL file not found: %s", path)
        return

    count = 0
    errors = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                    yield candidate
                    count += 1
                    if limit is not None and count >= limit:
                        logger.info("Reached limit of %d candidates", limit)
                        break
                except json.JSONDecodeError as e:
                    errors += 1
                    if errors <= 10:
                        logger.warning("Skipping malformed line %d: %s", line_num, e)
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed, retrying with latin-1")
        with open(path, "r", encoding="latin-1") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                    yield candidate
                    count += 1
                    if limit is not None and count >= limit:
                        break
                except json.JSONDecodeError:
                    errors += 1

    logger.info("Streamed %d candidates (%d errors) from %s", count, errors, path)


def load_job_description(path: Optional[str] = None) -> str:
    """Extract plain text from the job description .docx or .txt file.

    Args:
        path: Path to job_description.docx/txt. Defaults to config path.

    Returns:
        Full JD text as a single string.
    """
    if path is None:
        from src.config import JOB_DESCRIPTION_DOCX
        path = JOB_DESCRIPTION_DOCX

    path = Path(path)
    if not path.exists():
        logger.error("Job description file not found: %s", path)
        return ""

    # Support plain text files directly
    if path.suffix.lower() == ".txt":
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info("Loaded JD from text file: %d chars", len(text))
            return text
        except Exception as e:
            logger.error("Error reading plain text JD file: %s", e)
            return ""

    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        text = "\n".join(paragraphs)
        logger.info("Loaded JD: %d paragraphs, %d chars", len(paragraphs), len(text))
        return text
    except ImportError:
        logger.error("python-docx not installed. Run: pip install python-docx")
        return ""
    except Exception as e:
        logger.error("Error reading .docx file: %s", e)
        return ""
