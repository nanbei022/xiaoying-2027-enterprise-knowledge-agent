"""Validated local corpus and read-only retrieval tools.

This module performs retrieval only. A later LangGraph module will let the
model choose when and how to call these tools.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class CorpusError(ValueError):
    """The source document set is missing or invalid."""


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    section_id: str = Field(pattern=r"^D[0-9]+-S[0-9]+$")
    text: str

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("section text must not be empty")
        return value


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    doc_id: str = Field(pattern=r"^D[0-9]+$")
    title: str
    version_date: str
    scope: str | None
    priority: int | None
    supersedes: list[str]
    sections: list[Section] = Field(min_length=1)

    @field_validator("title")
    @classmethod
    def nonempty_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document title must not be empty")
        return value

    @field_validator("version_date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
            raise ValueError("version_date must use YYYY-MM-DD")
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("version_date must be a valid date") from exc
        return value


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: str
    message: str


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    doc_id: str
    section_id: str
    title: str
    version_date: str
    scope: str | None
    priority: int | None
    supersedes: list[str]
    snippet: str
    score: float = Field(ge=0, le=1)


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["success", "no_results", "error"]
    query: str
    scope: str | None
    hits: list[SearchHit]
    error: ToolError | None

    @model_validator(mode="after")
    def coherent_status(self) -> SearchResult:
        if self.status == "success" and (not self.hits or self.error is not None):
            raise ValueError("success requires hits and no error")
        if self.status == "no_results" and (self.hits or self.error is not None):
            raise ValueError("no_results requires no hits or error")
        if self.status == "error" and (self.hits or self.error is None):
            raise ValueError("error requires an error and no hits")
        return self


class ReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["success", "error"]
    document: Document | None
    error: ToolError | None

    @model_validator(mode="after")
    def coherent_status(self) -> ReadResult:
        if self.status == "success" and (self.document is None or self.error is not None):
            raise ValueError("success requires a document and no error")
        if self.status == "error" and (self.document is not None or self.error is None):
            raise ValueError("error requires an error and no document")
        return self


def _terms(value: str) -> set[str]:
    """Split Chinese into character bigrams and retain Latin words."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    groups = re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", normalized)
    terms: set[str] = set()
    for group in groups:
        if "\u3400" <= group[0] <= "\u9fff":
            terms.update(group[index : index + 2] for index in range(len(group) - 1))
            if len(group) == 1:
                terms.add(group)
        else:
            terms.add(group)
    return terms


class KnowledgeBase:
    """A small deterministic corpus backend that never decides the answer."""

    MAX_QUERY_LENGTH = 300
    MAX_RESULTS = 10

    def __init__(self, documents: list[Document]):
        self._documents: dict[str, Document] = {}
        section_ids: set[str] = set()
        for document in documents:
            if document.doc_id in self._documents:
                raise CorpusError(f"duplicate doc_id: {document.doc_id}")
            for section in document.sections:
                if section.section_id in section_ids:
                    raise CorpusError(f"duplicate section_id: {section.section_id}")
                if not section.section_id.startswith(f"{document.doc_id}-"):
                    raise CorpusError(f"section belongs to another document: {section.section_id}")
                section_ids.add(section.section_id)
            self._documents[document.doc_id] = document

        for document in documents:
            for old_id in document.supersedes:
                if old_id not in self._documents or old_id == document.doc_id:
                    raise CorpusError(f"invalid supersedes reference in {document.doc_id}")
        self._check_supersedes_cycles()

    @classmethod
    def from_json(cls, path: str | Path) -> KnowledgeBase:
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CorpusError(f"cannot read document corpus: {source}") from exc
        if not isinstance(data, list):
            raise CorpusError("document corpus must be a JSON array")
        try:
            documents = [Document.model_validate(item) for item in data]
        except ValidationError as exc:
            raise CorpusError(f"invalid document corpus: {exc}") from exc
        return cls(documents)

    def _check_supersedes_cycles(self) -> None:
        visited: set[str] = set()
        active: set[str] = set()

        def visit(doc_id: str) -> None:
            if doc_id in active:
                raise CorpusError(f"supersedes cycle involving {doc_id}")
            if doc_id in visited:
                return
            active.add(doc_id)
            for old_id in self._documents[doc_id].supersedes:
                visit(old_id)
            active.remove(doc_id)
            visited.add(doc_id)

        for doc_id in self._documents:
            visit(doc_id)

    def _search_hits(self, query: str, scope: str | None) -> list[SearchHit]:
        query_terms = _terms(query)
        hits: list[SearchHit] = []
        for document in self._documents.values():
            # Unknown scope remains visible so the model can detect ambiguity.
            if scope and document.scope not in (None, scope, "全体员工"):
                continue
            title_terms = _terms(document.title)
            scope_terms = _terms(document.scope or "")
            for section in document.sections:
                body_terms = _terms(section.text)
                matched = query_terms & (title_terms | scope_terms | body_terms)
                coverage = len(matched) / len(query_terms)
                if len(matched) < 2 or coverage < 0.15:
                    continue
                weighted = (
                    2 * len(query_terms & body_terms)
                    + len(query_terms & title_terms)
                    + 0.25 * len(query_terms & scope_terms)
                )
                score = round(min(1.0, weighted / (3 * len(query_terms))), 4)
                hits.append(
                    SearchHit(
                        doc_id=document.doc_id,
                        section_id=section.section_id,
                        title=document.title,
                        version_date=document.version_date,
                        scope=document.scope,
                        priority=document.priority,
                        supersedes=list(document.supersedes),
                        snippet=section.text,
                        score=score,
                    )
                )
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id, hit.section_id))
        return hits[: self.MAX_RESULTS]

    def search_documents(self, query: Any, scope: Any = None) -> dict[str, Any]:
        """Return ranked candidates without resolving conflicts or versions."""
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > self.MAX_QUERY_LENGTH
            or not _terms(query)
        ):
            return SearchResult(
                status="error",
                query=query if isinstance(query, str) else "",
                scope=scope if isinstance(scope, str) else None,
                hits=[],
                error=ToolError(code="INVALID_QUERY", message="query must be searchable text of at most 300 characters"),
            ).model_dump(mode="json")
        if scope is not None and (not isinstance(scope, str) or not scope.strip()):
            return SearchResult(
                status="error",
                query=query,
                scope=scope if isinstance(scope, str) else None,
                hits=[],
                error=ToolError(code="INVALID_SCOPE", message="scope must be nonempty text when provided"),
            ).model_dump(mode="json")
        query = query.strip()
        scope = scope.strip() if isinstance(scope, str) else None
        try:
            hits = self._search_hits(query, scope)
        except Exception:
            return SearchResult(
                status="error",
                query=query,
                scope=scope,
                hits=[],
                error=ToolError(code="TOOL_EXECUTION_ERROR", message="document search failed"),
            ).model_dump(mode="json")
        return SearchResult(
            status="success" if hits else "no_results",
            query=query,
            scope=scope,
            hits=hits,
            error=None,
        ).model_dump(mode="json")

    def read_document(self, doc_id: Any) -> dict[str, Any]:
        """Read exact original sections and metadata by document ID."""
        if not isinstance(doc_id, str) or re.fullmatch(r"D[0-9]+", doc_id) is None:
            return ReadResult(
                status="error",
                document=None,
                error=ToolError(code="INVALID_DOC_ID", message="doc_id must look like D01"),
            ).model_dump(mode="json")
        document = self._documents.get(doc_id)
        if document is None:
            return ReadResult(
                status="error",
                document=None,
                error=ToolError(code="DOC_NOT_FOUND", message=f"document {doc_id} was not found"),
            ).model_dump(mode="json")
        return ReadResult(status="success", document=document, error=None).model_dump(mode="json")
