"""
LLM-based text cleanup stage for OCR corpora before topic modeling.
"""

import csv
import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SYSTEM_PROMPT = (
    "Ты редактор OCR-текста дореволюционного русского корпуса. "
    "Исправляй только OCR-ошибки и явные артефакты. "
    "Не добавляй новые факты, не меняй смысл, имена, даты и топонимы. "
    "Сохраняй структуру текста. "
    "Ответ строго JSON: "
    '{"cleaned_text":"...", "uncertain_spans":["..."], "notes":"..."}'
)


def _cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    cyr_chars = re.findall(r"[А-Яа-яЁёѢѣѲѳѴѵІіѪѫѦѧѮѯѰѱЪъ]", text)
    return len(cyr_chars) / len(text)


def _change_ratio(original: str, cleaned: str) -> float:
    return abs(len(cleaned) - len(original)) / max(1, len(original))


def chunk_text(text: str, max_chunk_chars: int, overlap_chars: int) -> list[str]:
    """Splits text into chunks with optional overlap."""
    if not text:
        return [""]
    if max_chunk_chars <= 0:
        return [text]
    overlap_chars = max(0, min(overlap_chars, max_chunk_chars // 2))
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chunk_chars, text_len)
        if end < text_len:
            window_start = int(start + max_chunk_chars * 0.65)
            split_idx = text.rfind(" ", window_start, end)
            if split_idx != -1 and split_idx > start:
                end = split_idx
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def merge_chunks(chunks: list[str], overlap_chars: int) -> str:
    """Merges chunks by removing duplicated overlap segments."""
    if not chunks:
        return ""
    merged = chunks[0]
    for chunk in chunks[1:]:
        max_overlap = min(overlap_chars, len(merged), len(chunk))
        trim = 0
        for size in range(max_overlap, 0, -1):
            if merged[-size:] == chunk[:size]:
                trim = size
                break
        merged += chunk[trim:]
    return merged


@dataclass
class CleanResult:
    cleaned_text: str
    status: str
    uncertain_spans: list[str]
    notes: str
    change_ratio: float


class LLMTextCleaner:
    """Controlled LLM cleanup for OCR text."""

    def __init__(self, config: dict[str, Any]):
        llm_cfg = config.get("llm_cleaning", {})
        self.enabled = bool(llm_cfg.get("enabled", False))
        self.provider = llm_cfg.get("provider", "openai")
        self.model = llm_cfg.get("model", "gpt-4o-mini")
        self.temperature = float(llm_cfg.get("temperature", 0.0))
        self.max_chunk_chars = int(llm_cfg.get("max_chunk_chars", 1800))
        self.overlap_chars = int(llm_cfg.get("overlap_chars", 200))
        self.strict_mode = bool(llm_cfg.get("strict_mode", True))
        self.max_change_ratio = float(llm_cfg.get("max_change_ratio", 0.35))
        self.min_cyrillic_ratio = float(llm_cfg.get("min_cyrillic_ratio", 0.30))
        self.timeout_sec = int(llm_cfg.get("timeout_sec", 60))
        self.api_base = llm_cfg.get("api_base", "https://api.openai.com/v1")
        self.api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.getenv(self.api_key_env)
        self.only_if_suspect = bool(llm_cfg.get("only_if_suspect", False))
        self.suspect_score_threshold = float(llm_cfg.get("suspect_score_threshold", 0.12))
        self.min_text_length_for_llm = int(llm_cfg.get("min_text_length_for_llm", 0))
        self.max_requests_per_run = int(llm_cfg.get("max_requests_per_run", 0))
        self.enable_chunk_cache = bool(llm_cfg.get("enable_chunk_cache", True))
        self.request_count = 0
        self.chunk_cache: dict[str, tuple[str, list[str], str]] = {}
        self.context_notes = llm_cfg.get("context_notes", "")
        self.cleanup_rules = llm_cfg.get(
            "cleanup_rules",
            [
                "исправляй OCR-ошибки символов в кириллице/дореформенной орфографии",
                "не сокращай имена, фамилии, топонимы, даты, номера и должности",
                "сохраняй исходную структуру абзацев, если это возможно",
                "не выдумывай отсутствующие фрагменты",
            ],
        )

    def clean_text(self, text: str, document_id: str = "") -> CleanResult:
        """Runs LLM cleanup on a single text document."""
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        if not self.enabled:
            return CleanResult(text, "disabled", [], "llm_cleaning.enabled=false", 0.0)

        if (
            self.only_if_suspect
            and (self.min_text_length_for_llm <= 0 or len(text) >= self.min_text_length_for_llm)
            and not self._is_suspect_ocr_text(text)
        ):
            return CleanResult(
                text, "skipped_clean_text", [], "ocr_noise_below_threshold", 0.0
            )

        if self.min_text_length_for_llm > 0 and len(text) < self.min_text_length_for_llm:
            return CleanResult(text, "skipped_short_text", [], "below_min_text_length", 0.0)

        if self.provider == "openai" and not self.api_key:
            note = f"API key not found in env var {self.api_key_env}. Raw text kept."
            return CleanResult(text, "skipped_no_api_key", [], note, 0.0)

        chunks = chunk_text(text, self.max_chunk_chars, self.overlap_chars)
        cleaned_chunks: list[str] = []
        uncertain_spans_all: list[str] = []
        notes_parts: list[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            cleaned_chunk, uncertain_spans, note = self._clean_chunk_with_provider(
                chunk, document_id, idx, len(chunks)
            )
            final_chunk, validate_note = self._validate_chunk(chunk, cleaned_chunk)
            cleaned_chunks.append(final_chunk)
            uncertain_spans_all.extend(uncertain_spans)
            if note:
                notes_parts.append(note)
            if validate_note:
                notes_parts.append(validate_note)

        merged_text = merge_chunks(cleaned_chunks, self.overlap_chars)
        ratio = _change_ratio(text, merged_text)
        return CleanResult(
            cleaned_text=merged_text,
            status="cleaned",
            uncertain_spans=uncertain_spans_all,
            notes=" | ".join(notes_parts)[:2000],
            change_ratio=ratio,
        )

    def _validate_chunk(self, original: str, cleaned: str) -> tuple[str, str]:
        """Validates cleaned text and falls back to original on suspicious edits."""
        if not self.strict_mode:
            return cleaned, ""
        if not cleaned.strip():
            return original, "empty_cleaned_chunk_fallback"
        change_ratio = _change_ratio(original, cleaned)
        if change_ratio > self.max_change_ratio:
            return original, f"high_change_ratio_fallback:{change_ratio:.3f}"
        if _cyrillic_ratio(cleaned) < self.min_cyrillic_ratio and _cyrillic_ratio(original) > 0.15:
            return original, "low_cyrillic_ratio_fallback"
        return cleaned, ""

    def _clean_chunk_with_provider(
        self, chunk: str, document_id: str, chunk_num: int, total_chunks: int
    ) -> tuple[str, list[str], str]:
        cache_key = self._chunk_cache_key(chunk)
        if self.enable_chunk_cache and cache_key in self.chunk_cache:
            return self.chunk_cache[cache_key]

        if self.max_requests_per_run and self.request_count >= self.max_requests_per_run:
            result = (chunk, [], "request_budget_exhausted")
            if self.enable_chunk_cache:
                self.chunk_cache[cache_key] = result
            return result

        if self.provider == "openai":
            result = self._clean_chunk_openai(chunk, document_id, chunk_num, total_chunks)
        else:
            note = f"unsupported provider '{self.provider}', raw chunk used"
            result = (chunk, [], note)
        if self.enable_chunk_cache:
            self.chunk_cache[cache_key] = result
        return result

    def _chunk_cache_key(self, chunk: str) -> str:
        payload = f"{self.provider}|{self.model}|{chunk}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _is_suspect_ocr_text(self, text: str) -> bool:
        score = self._ocr_noise_score(text)
        return score >= self.suspect_score_threshold

    @staticmethod
    def _ocr_noise_score(text: str) -> float:
        if not text.strip():
            return 1.0
        length = max(1, len(text))
        bad_symbols = re.findall(r"[^А-Яа-яЁёѢѣѲѳѴѵІіѪѫѦѧѮѯѰѱЪъA-Za-z0-9\s.,;:!?()\"'«»\-—]", text)
        latin_chars = re.findall(r"[A-Za-z]", text)
        mixed_word_noise = re.findall(
            r"[А-Яа-яЁёѢѣѲѳѴѵІіѪѫѦѧѮѯѰѱЪъ]+[A-Za-z0-9]+|[A-Za-z0-9]+[А-Яа-яЁёѢѣѲѳѴѵІіѪѫѦѧѮѯѰѱЪъ]+",
            text,
        )
        repetitive_punct = re.findall(r"[^\w\s]{3,}", text)
        return (
            0.35 * (len(bad_symbols) / length)
            + 0.25 * (len(latin_chars) / length)
            + 0.30 * (len(mixed_word_noise) / max(1, len(text.split())))
            + 0.10 * (len(repetitive_punct) / max(1, len(text.split())))
        )

    def _clean_chunk_openai(
        self, chunk: str, document_id: str, chunk_num: int, total_chunks: int
    ) -> tuple[str, list[str], str]:
        self.request_count += 1
        rules_block = "\n".join(f"- {rule}" for rule in self.cleanup_rules)
        context_block = self.context_notes.strip() or "нет дополнительного контекста"
        user_prompt = (
            f"document_id={document_id}; chunk={chunk_num}/{total_chunks}\n\n"
            f"Контекст корпуса:\n{context_block}\n\n"
            f"Правила очистки:\n{rules_block}\n\n"
            f"Текст для очистки OCR:\n{chunk}"
        )
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            url=f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = self._parse_llm_json(content)
            cleaned = parsed.get("cleaned_text", chunk)
            uncertain = parsed.get("uncertain_spans", [])
            notes = parsed.get("notes", "")
            return cleaned, uncertain if isinstance(uncertain, list) else [], notes
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as e:
            note = f"openai_error_chunk_{chunk_num}:{e}"
            return chunk, [], note

    @staticmethod
    def _parse_llm_json(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return {"cleaned_text": content, "uncertain_spans": [], "notes": "non_json_response"}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {
                    "cleaned_text": content,
                    "uncertain_spans": [],
                    "notes": "json_parse_failed",
                }


def clean_csv_with_llm(input_csv: str, output_csv: str, config: dict[str, Any]) -> str:
    """Cleans text column in CSV using LLM and writes enriched output CSV."""
    cleaner = LLMTextCleaner(config)
    data_cfg = config.get("data", {})
    text_column = data_cfg.get("text_column", "text")
    replace_text_column = bool(config.get("llm_cleaning", {}).get("replace_text_column", True))

    with open(input_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {input_csv}")
        if text_column not in reader.fieldnames:
            raise ValueError(f"Text column '{text_column}' not found in {input_csv}")

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    extra_columns = [
        "raw_text",
        "text_cleaned",
        "llm_clean_status",
        "llm_clean_notes",
        "llm_change_ratio",
        "llm_uncertain_spans",
        "llm_provider",
        "llm_model",
    ]
    for column in extra_columns:
        if column not in fieldnames:
            fieldnames.append(column)

    for row in rows:
        original_text = row.get(text_column, "") or ""
        doc_id = row.get("id", "")
        result = cleaner.clean_text(original_text, document_id=doc_id)
        row["raw_text"] = original_text
        row["text_cleaned"] = result.cleaned_text
        if replace_text_column:
            row[text_column] = result.cleaned_text
        row["llm_clean_status"] = result.status
        row["llm_clean_notes"] = result.notes
        row["llm_change_ratio"] = f"{result.change_ratio:.5f}"
        row["llm_uncertain_spans"] = json.dumps(result.uncertain_spans, ensure_ascii=False)
        row["llm_provider"] = cleaner.provider
        row["llm_model"] = cleaner.model

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(
        "LLM cleanup completed: %s -> %s (%d rows, %d API calls)",
        input_csv,
        output_csv,
        len(rows),
        cleaner.request_count,
    )
    return str(output_path)
