from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from settings import app_data_dir


def default_memory_path() -> Path:
    return app_data_dir() / "memory.json"


@dataclass
class StyleMemory:
    path: Path = field(default_factory=default_memory_path)
    style_summary: str = "偏短句、微信口语、少解释、少承诺。"
    examples: list[dict[str, str]] = field(default_factory=list)
    common_phrases: Counter = field(default_factory=Counter)
    max_examples: int = 3000

    @property
    def vector_path(self) -> Path:
        return self.path.with_name("vector_memory.json")

    @property
    def sop_library_path(self) -> Path:
        return self.path.with_name("sop_library.json")

    def load(self) -> "StyleMemory":
        if not self.path.exists():
            return self
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.style_summary = str(data.get("style_summary", self.style_summary))
        raw_examples = list(data.get("examples", []))[-self.max_examples :]
        self.examples = [_normalize_example(item) for item in raw_examples if _normalize_example(item)]
        self.common_phrases = Counter(data.get("common_phrases", {}))
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.examples = _dedupe_memory_examples(self.examples)[-self.max_examples :]
        data = {
            "style_summary": self.style_summary,
            "examples": self.examples,
            "common_phrases": dict(self.common_phrases.most_common(120)),
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._save_vector_index()

    def learn_from_chat_text(self, conversation_text: str) -> None:
        lines = [line.strip() for line in conversation_text.splitlines() if line.strip()]
        observed_examples: list[dict[str, str]] = []
        for index, line in enumerate(lines):
            if not _is_own_line(line):
                continue
            reply = line.split("]", 1)[-1].strip()
            cue = "\n".join(lines[max(0, index - 8) : index])
            if cue and reply:
                observed_examples.append(
                    _build_example(
                        partner="训练模式观察",
                        cue=cue,
                        reply=reply,
                        source="observed_chat",
                    )
                )
        for item in observed_examples[-8:]:
            self.examples.append(item)
            for phrase in self._extract_phrases(item["reply"]):
                self.common_phrases[phrase] += 1
        self._refresh_summary()
        self.save()

    def learn_from_sent_reply(
        self,
        partner: str,
        conversation_text: str,
        sent_reply: str,
        source: str = "manual_edit",
    ) -> None:
        cleaned = sent_reply.strip()
        if not cleaned:
            return
        self.examples.append(
            _build_example(
                partner=partner,
                cue=conversation_text,
                reply=cleaned,
                source=_clean_source(source),
            )
        )
        for phrase in self._extract_phrases(cleaned):
            self.common_phrases[phrase] += 3
        self._refresh_summary()
        self.save()

    def learn_from_training_examples(self, examples: list[dict[str, str]]) -> None:
        for item in examples:
            reply = str(item.get("reply", "")).strip()
            cue = str(item.get("cue", "")).strip()
            if not reply or not cue:
                continue
            self.examples.append(
                _with_training_metadata(
                    _build_example(
                        partner=str(item.get("partner", "聊天记录训练")),
                        cue=cue,
                        reply=reply,
                        source=str(item.get("source", "chat_zip_training")),
                        created_at=str(item.get("created_at", "")),
                    ),
                    item,
                )
            )
            for phrase in self._extract_phrases(reply):
                self.common_phrases[phrase] += 2
        self._refresh_summary()
        self.save()

    def prompt_block(self, conversation_text: str = "") -> str:
        phrases = "、".join(phrase for phrase, _ in self.common_phrases.most_common(12))
        relevant = self.relevant_examples(conversation_text, limit=5) if conversation_text else self.examples[-5:]
        examples = "\n".join(_format_prompt_example(index, item) for index, item in enumerate(relevant, 1))
        return (
            f"用户风格摘要：{self.style_summary}\n"
            f"常用表达：{phrases or '可以、我看下、晚点、确认下'}\n"
            "相似历史聊天片段与真实回复：\n"
            f"{examples or '- 暂无历史样例，可按当前上下文自然接话'}"
        )

    def relevant_examples(self, conversation_text: str, limit: int = 6) -> list[dict[str, str]]:
        relevant = self.strict_relevant_examples(conversation_text, limit=limit)
        return relevant or self.examples[-limit:]

    def strict_relevant_examples(self, conversation_text: str, limit: int = 6) -> list[dict[str, str]]:
        query_tokens = _text_tokens(conversation_text)
        if not query_tokens:
            return []
        last_other = _last_other_text(conversation_text)
        last_tokens = _text_tokens(last_other)
        scored = []
        for index, item in enumerate(self.examples):
            cue = str(item.get("cue", ""))
            reply = str(item.get("reply", ""))
            cue_tokens = _text_tokens(cue)
            reply_tokens = _text_tokens(reply)
            score = len(query_tokens & cue_tokens) * 3 + len(query_tokens & reply_tokens)
            if last_tokens:
                score += len(last_tokens & cue_tokens) * 4
            score += _phrasebook_intent_bonus(conversation_text, item)
            if score:
                score += _example_priority(item) / 10
                scored.append((score, index, item))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected = [item for _, _, item in scored[:limit]]
        if len(selected) < limit:
            seen = {item.get("conversation_hash", "") for item in selected}
            for item in self.vector_relevant_examples(conversation_text, limit=limit * 2):
                key = item.get("conversation_hash", "")
                if key and key not in seen:
                    selected.append(item)
                    seen.add(key)
                if len(selected) >= limit:
                    break
        return selected[:limit]

    def vector_relevant_examples(self, conversation_text: str, limit: int = 6) -> list[dict[str, str]]:
        query_vector = _text_vector(conversation_text)
        if not query_vector or not self.vector_path.exists():
            return []
        try:
            data = json.loads(self.vector_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        by_hash = {item.get("conversation_hash", ""): item for item in self.examples}
        scored: list[tuple[float, int, dict[str, str]]] = []
        for index, item in enumerate(data.get("items", [])):
            if not isinstance(item, dict):
                continue
            key = str(item.get("conversation_hash", ""))
            example = by_hash.get(key)
            if not example:
                continue
            score = _cosine_sparse(query_vector, item.get("vector", {}))
            if score >= 0.08:
                scored.append((score, index, example))
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        return [item for _, _, item in scored[:limit]]

    def library_summaries(self) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        for item in self.examples:
            key = str(item.get("source", "manual_edit") or "manual_edit")
            grouped.setdefault(key, []).append(item)
        summaries = []
        for key, items in grouped.items():
            summaries.append(
                {
                    "key": key,
                    "name": _library_name(key),
                    "count": len(items),
                    "latest": max((str(item.get("created_at", "")) for item in items), default=""),
                }
            )
        summaries.sort(key=lambda item: (int(item["count"]), str(item["latest"])), reverse=True)
        return summaries

    def examples_for_library(self, library_key: str = "", limit: int = 300) -> list[dict[str, str]]:
        items = self.examples
        if library_key:
            items = [item for item in items if item.get("source") == library_key]
        return [dict(item) for item in items[-limit:]][::-1]

    def scored_examples(self, conversation_text: str, library_key: str = "", limit: int = 50) -> list[dict[str, object]]:
        query_tokens = _text_tokens(conversation_text)
        query_vector = _text_vector(conversation_text)
        vector_by_hash = self._vector_by_hash()
        rows: list[dict[str, object]] = []
        for item in self.examples:
            if library_key and item.get("source") != library_key:
                continue
            cue = str(item.get("cue", ""))
            reply = str(item.get("reply", ""))
            cue_tokens = _text_tokens(cue)
            reply_tokens = _text_tokens(reply)
            keyword_hits = len(query_tokens & cue_tokens)
            reply_hits = len(query_tokens & reply_tokens)
            vector_score = _cosine_sparse(query_vector, vector_by_hash.get(str(item.get("conversation_hash", "")), {}))
            priority = _example_priority(item)
            raw_score = keyword_hits * 16 + reply_hits * 8 + vector_score * 45
            raw_score += _phrasebook_intent_bonus(conversation_text, item) * 6
            if keyword_hits or reply_hits or vector_score >= 0.08:
                raw_score += priority * 0.22
            score = max(0, min(100, int(round(raw_score))))
            reasons = []
            if keyword_hits:
                reasons.append("keyword")
            if reply_hits:
                reasons.append("reply")
            if vector_score >= 0.08:
                reasons.append("vector")
            if _phrasebook_intent_bonus(conversation_text, item):
                reasons.append("intent")
            if priority:
                reasons.append(f"priority:{priority}")
            if not reasons:
                reasons.append("recent")
            row = dict(item)
            row.update(
                {
                    "library_key": item.get("source", ""),
                    "library_name": _library_name(str(item.get("source", ""))),
                    "score": score,
                    "reasons": reasons,
                    "priority": priority,
                    "vector_score": round(vector_score, 4),
                }
            )
            rows.append(row)
        rows.sort(
            key=lambda item: (
                int(item.get("score", 0)),
                int(item.get("priority", 0) or 0),
                str(item.get("created_at", "")),
            ),
            reverse=True,
        )
        return rows[:limit]

    def update_example(self, conversation_hash: str, cue: str | None = None, reply: str | None = None) -> bool:
        target = str(conversation_hash or "")
        if not target:
            return False
        for index, item in enumerate(self.examples):
            if str(item.get("conversation_hash", "")) != target:
                continue
            new_cue = _normalize_context(cue if cue is not None else str(item.get("cue", "")))
            new_reply = _normalize_reply(reply if reply is not None else str(item.get("reply", "")))
            if not new_reply:
                return False
            updated = dict(item)
            updated["cue"] = new_cue
            updated["reply"] = new_reply
            updated["conversation_hash"] = _conversation_hash(new_cue, new_reply)
            updated["created_at"] = _utc_now()
            self.examples[index] = updated
            self.save()
            return True
        return False

    def _vector_by_hash(self) -> dict[str, dict[str, float]]:
        if not self.vector_path.exists():
            return {}
        try:
            data = json.loads(self.vector_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        result: dict[str, dict[str, float]] = {}
        for item in data.get("items", []):
            if isinstance(item, dict):
                key = str(item.get("conversation_hash", ""))
                vector = item.get("vector", {})
                if key and isinstance(vector, dict):
                    result[key] = vector
        return result

    def suggest_replies(self, conversation_text: str, limit: int = 3) -> list[str]:
        return self.suggest_replies_strict(conversation_text, limit=limit) or _unique_replies(
            self.relevant_examples(conversation_text, limit=limit * 4),
            limit,
        )

    def suggest_replies_strict(self, conversation_text: str, limit: int = 3) -> list[str]:
        return _unique_replies(self.strict_relevant_examples(conversation_text, limit=limit * 4), limit)

    def _refresh_summary(self) -> None:
        examples = [item["reply"] for item in self.examples[-24:] if item.get("reply")]
        avg_len = int(sum(len(x) for x in examples) / len(examples)) if examples else 18
        emoji = any(re.search(r"[\U0001F300-\U0001FAFF]", text) for text in examples)
        question = any("？" in text or "?" in text or "吗" in text for text in examples)
        length_note = "短句" if avg_len <= 24 else "中等长度"
        emoji_note = "会少量用表情" if emoji else "基本不用表情"
        question_note = "常用反问或确认句" if question else "表达直接"
        self.style_summary = f"{length_note}、微信口语、{emoji_note}、{question_note}、少客服腔。"

    def _save_vector_index(self) -> None:
        items = []
        for item in self.examples:
            vector = _text_vector(f"{item.get('cue', '')}\n{item.get('reply', '')}")
            if not vector:
                continue
            items.append(
                {
                    "conversation_hash": item.get("conversation_hash", ""),
                    "source": item.get("source", ""),
                    "reply": item.get("reply", ""),
                    "vector": vector,
                }
            )
        data = {"version": 1, "dim": 384, "items": items}
        self.vector_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _extract_phrases(text: str) -> list[str]:
        pieces = re.split(r"[，。！？、,.!?\s]+", text)
        return [piece for piece in pieces if 2 <= len(piece) <= 10]


def _unique_replies(examples: list[dict[str, str]], limit: int) -> list[str]:
    suggestions: list[str] = []
    for item in examples:
        reply = str(item.get("reply", "")).strip()
        if reply and reply not in suggestions:
            suggestions.append(reply)
        if len(suggestions) >= limit:
            break
    return suggestions


def _library_name(source: str) -> str:
    names = {
        "tiantian_html_training": "甜甜话术库",
        "chat_zip_training": "导入话术库",
        "wechat_export_training": "微信记录话术库",
        "sop_analysis_training": "SOP分析话术库",
        "manual_edit": "手动发送记忆",
        "selected_candidate": "候选发送记忆",
        "observed_chat": "训练模式观察",
        "managed_auto": "全托管记录",
    }
    return names.get(source, source or "默认话术库")


def _build_example(
    partner: str,
    cue: str,
    reply: str,
    source: str,
    created_at: str = "",
) -> dict[str, str]:
    full_cue = _normalize_context(cue)
    full_reply = _normalize_reply(reply)
    return {
        "partner": partner.strip()[:50] or "聊天记录训练",
        "cue": full_cue,
        "reply": full_reply,
        "source": _clean_source(source),
        "created_at": created_at or _utc_now(),
        "conversation_hash": _conversation_hash(full_cue, full_reply),
    }


def _with_training_metadata(example: dict[str, str], raw_item: dict[str, object]) -> dict[str, str]:
    for key in ("scenario_id", "scenario_title", "why"):
        value = str(raw_item.get(key, "")).strip()
        if value:
            example[key] = value[:240]
    priority = _parse_priority(raw_item.get("priority", ""))
    if priority:
        example["priority"] = str(priority)
    return example


def _normalize_example(item: object) -> dict[str, str]:
    if not isinstance(item, dict):
        return {}
    cue = _normalize_context(str(item.get("cue", "")))
    reply = _normalize_reply(str(item.get("reply", "")))
    if not reply:
        return {}
    source = _clean_source(str(item.get("source", "manual_edit")))
    result = {
        "partner": str(item.get("partner", "聊天记录训练")).strip()[:50],
        "cue": cue,
        "reply": reply,
        "source": source,
        "created_at": str(item.get("created_at", "")) or _utc_now(),
        "conversation_hash": str(item.get("conversation_hash", "")) or _conversation_hash(cue, reply),
    }
    for key in ("scenario_id", "scenario_title", "why"):
        value = str(item.get(key, "")).strip()
        if value:
            result[key] = value[:240]
    priority = _parse_priority(item.get("priority", ""))
    if priority:
        result["priority"] = str(priority)
    return result


def _parse_priority(value: object) -> int:
    try:
        number = int(float(str(value).strip()))
    except Exception:
        return 0
    return max(0, min(100, number))


def _example_priority(item: dict[str, str]) -> int:
    return _parse_priority(item.get("priority", ""))


def _phrasebook_intent_bonus(conversation_text: str, item: dict[str, str]) -> int:
    query = conversation_text or ""
    haystack = "\n".join(
        str(item.get(key, ""))
        for key in ("cue", "reply", "scenario_title", "why")
        if item.get(key)
    )
    if _contains_any(query, ("怎么放", "放哪里", "放哪", "摆哪里", "摆哪", "位置", "手机后面", "收到", "收到了")):
        placement_hits = _contains_any(
            haystack,
            ("怎么放", "放哪里", "放置", "使用说明", "位置", "手机壳", "钱包", "床头柜", "衣柜", "财位", "正南"),
        )
        if placement_hits and _contains_any(query, ("怎么放", "放哪里", "放哪", "摆哪里", "摆哪", "位置", "手机后面")):
            return 80
        if placement_hits and _contains_any(query, ("收到", "收到了")):
            return 5
    if _contains_any(query, ("刚加", "添加", "开始聊", "可以开始")) and _contains_any(
        haystack,
        ("刚添加", "初始接待", "开始聊天"),
    ):
        return 12
    return 0


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _normalize_context(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _normalize_reply(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _format_prompt_example(index: int, item: dict[str, str]) -> str:
    cue = _truncate_for_prompt(str(item.get("cue", "")), 700)
    reply = _truncate_for_prompt(str(item.get("reply", "")), 180)
    return f"{index}. 历史聊天片段：\n{cue}\n当时真实回复：{reply}"


def _truncate_for_prompt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _conversation_hash(cue: str, reply: str) -> str:
    payload = f"{_compact_for_hash(cue)}\n---\n{_compact_for_hash(reply)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _compact_for_hash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_own_line(line: str) -> bool:
    return line.startswith("[我]") or line.startswith("[我 ")


def _last_other_text(conversation_text: str) -> str:
    for line in reversed(conversation_text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("[对方]"):
            return stripped.split("]", 1)[-1].strip()
    return ""


def _text_tokens(text: str) -> set[str]:
    text = re.sub(r"\[(?:对方|我)\]", "", text)
    cleaned = re.sub(r"\s+", "", text)
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9]{2,}", cleaned))
    for size in (2, 3, 4):
        for index in range(max(0, len(cleaned) - size + 1)):
            piece = cleaned[index : index + size]
            if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
                tokens.add(piece)
    return tokens


def _text_vector(text: str, dim: int = 384) -> dict[str, float]:
    features = Counter()
    cleaned = re.sub(r"\s+", "", text or "").lower()
    for token in _text_tokens(text):
        features[token] += 2
    for size in (2, 3, 4):
        for index in range(max(0, len(cleaned) - size + 1)):
            features[cleaned[index : index + size]] += 1
    if not features:
        return {}
    buckets: Counter[str] = Counter()
    for feature, weight in features.items():
        digest = hashlib.sha1(feature.encode("utf-8", errors="ignore")).hexdigest()
        buckets[str(int(digest[:8], 16) % dim)] += float(weight)
    norm = sum(value * value for value in buckets.values()) ** 0.5
    if not norm:
        return {}
    return {key: round(value / norm, 6) for key, value in buckets.items()}


def _cosine_sparse(left: dict[str, float], right: object) -> float:
    if not isinstance(right, dict) or not left:
        return 0.0
    score = 0.0
    for key, value in left.items():
        try:
            score += value * float(right.get(key, 0.0))
        except Exception:
            continue
    return score


def _dedupe_memory_examples(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for raw_item in examples:
        item = _normalize_example(raw_item)
        if not item:
            continue
        key = f"{item.get('conversation_hash', '')}:{_compact_for_hash(item.get('reply', ''))}"
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _clean_source(source: str) -> str:
    allowed = {
        "selected_candidate",
        "manual_edit",
        "managed_auto",
        "observed_chat",
        "chat_zip_training",
        "tiantian_html_training",
        "wechat_export_training",
        "sop_analysis_training",
    }
    return source if source in allowed else "manual_edit"
