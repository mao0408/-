from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from llm_clients import build_text_generation_request, extract_text_from_response, provider_source_label
from memory import StyleMemory
from scenario_bank import replies_for_intent
from settings import AppSettings


LOW_INFORMATION_ACKS = {"好", "好的", "嗯", "嗯嗯", "知道了", "收到", "ok", "OK", "行", "可以"}
GENERIC_LOW_EFFORT = {
    "收到",
    "收到啊",
    "收到啦",
    "好的",
    "嗯嗯",
    "了解了",
    "理解了",
    "我看下",
    "帮你看看",
    "晚点回你",
    "我试试看",
    "怎么弄",
    "怎么弄？",
}
UNSUPPORTED_JARGON = {"福宝", "能量", "加持", "祝福孩子", "考试顺利"}


def parse_numbered_replies(content: str) -> list[str]:
    text = (content or "").strip()
    if not text:
        return []
    matches = re.findall(
        r"(?:^|\n)\s*\d+\s*[\.\)、]\s*(.+?)(?=\n\s*\d+\s*[\.\)、]|\Z)",
        text,
        re.S,
    )
    raw_items = matches if matches else text.splitlines()
    replies: list[str] = []
    for item in raw_items:
        cleaned = _clean_reply_line(str(item))
        if cleaned:
            replies.append(cleaned)
        if len(replies) >= 3:
            break
    return replies


def classify_intent(conversation_text: str) -> str:
    last = last_meaningful_other_message(conversation_text)
    all_text = conversation_text or ""
    if _is_low_information_ack(last):
        return "low_ack"
    if _is_how_to_receive_scene(all_text):
        return "how_to_receive"
    if _is_thanks_or_received_scene(last):
        return "thanks"
    if _is_job_or_recruitment_card(all_text):
        return "job_card"
    if _is_food_share_scene(all_text):
        return "food_share"
    if _is_venting_scene(all_text):
        return "venting"
    if _is_question(last):
        return "question"
    return "general"


def auto_reply_decision(conversation_text: str) -> tuple[bool, str]:
    intent = classify_intent(conversation_text)
    if intent in {"low_ack", "thanks"}:
        return True, intent
    return False, intent


class ReplyEngine:
    def __init__(self, settings: AppSettings, memory: StyleMemory) -> None:
        self.settings = settings
        self.memory = memory
        self.last_source = "未生成"
        self.last_error = ""

    def generate(self, conversation_text: str, partner: str = "", target_turn: list[str] | None = None) -> list[str]:
        self.last_error = ""
        effective_target_turn = target_turn if target_turn is not None else _infer_unreplied_other_messages(conversation_text)

        if self.settings.reply_source_mode == "memory":
            memory_replies = self._memory_replies(conversation_text)
            if memory_replies:
                self.last_source = "话术库"
                return memory_replies

        if self.settings.api_key and requests is not None:
            api_replies = self._generate_from_api(conversation_text, partner, effective_target_turn)
            if api_replies or self.settings.reply_source_mode == "model":
                return api_replies

        if self.settings.reply_source_mode == "model":
            self.last_source = "未生成"
            if not self.last_error:
                self.last_error = "未配置文本接口 API Key，无法使用大模型生成"
            return []

        return self._fallback(conversation_text)

    def _generate_from_api(
        self,
        conversation_text: str,
        partner: str,
        target_turn: list[str] | None,
    ) -> list[str]:
        original_model = self.settings.model
        models = _candidate_models(self.settings.model) if self.settings.api_provider == "zhipu" else [self.settings.model]
        errors: list[str] = []
        for model in models:
            self.settings.model = model
            try:
                replies, call_count = self._generate_with_current_model(conversation_text, partner, target_turn)
                if replies:
                    label = provider_source_label(self.settings)
                    if self.settings.api_provider == "zhipu":
                        label = f"{label}（{self.settings.model}）"
                    self.last_source = f"{label}，调用{call_count}次"
                    self.last_error = ""
                    return replies
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                continue
            finally:
                self.settings.model = original_model

        if errors:
            self.last_error = f"{provider_source_label(self.settings)} 调用失败：" + "；".join(errors)
        if self.settings.reply_source_mode == "model":
            self.last_source = provider_source_label(self.settings)
            return []
        return []

    def _generate_with_current_model(
        self,
        conversation_text: str,
        partner: str,
        target_turn: list[str] | None,
    ) -> tuple[list[str], int]:
        collected: list[str] = []
        call_count = 0
        prompt = self._build_prompt(conversation_text, partner, target_turn=target_turn)
        base_messages = [
            {
                "role": "system",
                "content": (
                    "你是微信回复建议器。只输出3条候选回复，每条一行并用1/2/3编号。"
                    "必须优先回答当前未回复轮次和最后一句，不能复述对方原句，不能编造事实。"
                ),
            },
            {"role": "user", "content": prompt},
        ]

        for attempt in range(3):
            messages = base_messages if attempt == 0 else _refill_messages(base_messages, collected, 3 - len(collected))
            content = self._request_model_content(messages)
            call_count += 1
            if not (content or "").strip():
                self.last_error = (
                    f"{provider_source_label(self.settings)} 返回里没有识别到文本，"
                    "请检查接口、模型或自定义返回文本路径"
                )
                return [], call_count
            raw = parse_numbered_replies(content)
            raw = _drop_parrot_replies(raw, last_meaningful_other_message("\n".join(target_turn or []) or conversation_text))
            raw = self._quality_filter(raw, "\n".join(target_turn or []) or conversation_text)
            raw = _filter_context_echo_replies(raw, conversation_text)
            collected = _merge_unique_replies(collected, raw)
            normalized = _normalize_api_replies(collected)
            if len(normalized) >= 3:
                return normalized[:3], call_count

        normalized = _normalize_api_replies(collected)
        if normalized:
            return normalized[:3], call_count
        self.last_error = (
            f"{provider_source_label(self.settings)} 返回内容未通过质量过滤，"
            "请检查模型能力、提示词或接口返回内容"
        )
        self.last_source = provider_source_label(self.settings)
        return [], call_count

    def _request_model_content(self, messages: list[dict[str, str]]) -> str:
        request = build_text_generation_request(self.settings, messages)
        response = requests.post(
            request.url,
            headers=request.headers,
            json=request.payload,
            timeout=20,
        )
        if int(getattr(response, "status_code", 200) or 200) >= 400:
            detail = _response_error_detail(response)
            raise RuntimeError(f"HTTP {response.status_code}: {detail}")
        response.raise_for_status()
        return extract_text_from_response(response.json(), self.settings)

    def _build_prompt(self, conversation_text: str, partner: str = "", target_turn: list[str] | None = None) -> str:
        target_messages = [message.strip() for message in (target_turn or []) if message and message.strip()]
        if not target_messages:
            target_messages = _infer_unreplied_other_messages(conversation_text)
        latest_target = target_messages[-1] if target_messages else last_meaningful_other_message(conversation_text)
        target_text = "\n".join(target_messages).strip()
        complex_requirement = _is_complex_requirement(target_text or conversation_text)
        length_rule = (
            "复杂需求、方案咨询允许每条35-70字；普通闲聊尽量30字内。"
            if complex_requirement
            else "每条尽量不超过30个中文字符，像微信里随手发的话。"
        )
        source_rule = (
            "回复来源：大模型。你必须自己理解整句和上下文，不要按关键词套模板，不要参考本地场景库。"
            if self.settings.reply_source_mode == "model"
            else "回复来源：大模型 + 长期记忆。长期记忆只提供历史表达参考，当前回复必须重新理解上下文。"
        )
        memory_block = "" if self.settings.reply_source_mode == "model" else self.memory.prompt_block(conversation_text)
        memory_rule = (
            "7. 不使用历史话术库，只根据当前上下文自然生成。"
            if self.settings.reply_source_mode == "model"
            else "7. 优先参考相似历史聊天片段与真实回复的接话方式，但不要照抄。"
        )
        return f"""当前聊天对象：{partner or "当前微信联系人"}

最近聊天记录，越下面越新：
{conversation_text.strip() or "（没有识别到聊天内容）"}

当前需要回复的未回复轮次：
{target_text or latest_target or "（使用最后一轮未回复对方消息）"}

本轮最后一句（最高优先级）：
{latest_target or "（无）"}

{memory_block}

回复要求：
1. 整段上下文只作参考，对方发的内容优先级更高，对方最后一句优先级最高。
2. 如果对方有多个未回复问题，每个未回复的问题都要照顾到，并合并成一条自然回复。
3. 如果前面的问题我已经回复过，只参考背景，只回复最后一句。
4. 不能编造，不懂就自然确认，不要假装看过链接或图片细节。
5. 不要输出“理解了 / 帮你看看 / 怎么弄？”这类敷衍句。
6. {length_rule}
{memory_rule}
8. {source_rule}

请输出3条候选，只要候选本身：
1. xxx
2. xxx
3. xxx"""

    def _fallback(self, conversation_text: str) -> list[str]:
        intent = classify_intent(conversation_text)
        target = "\n".join(_infer_unreplied_other_messages(conversation_text)) or conversation_text
        if self.settings.reply_source_mode != "model":
            memory = self._memory_replies(conversation_text)
            if memory:
                self.last_source = "长期记忆"
                return memory
        self.last_source = "本地兜底规则"
        candidates = _rule_replies(intent, target)
        return _normalize_api_replies(self._quality_filter(candidates, target)) or candidates[:3]

    def _memory_replies(self, conversation_text: str) -> list[str]:
        if hasattr(self.memory, "suggest_replies_strict"):
            replies = self.memory.suggest_replies_strict(conversation_text, limit=3)
        else:
            replies = self.memory.suggest_replies(conversation_text, limit=3)
        replies = self._quality_filter(replies, conversation_text)
        return _normalize_api_replies(replies)[:3]

    def _quality_filter(self, replies: list[str], context: str) -> list[str]:
        if not replies:
            return []
        substantive = _is_substantive_context(context)
        complex_requirement = _is_complex_requirement(context)
        context_tokens = _text_tokens(context)
        result: list[str] = []
        for reply in replies:
            cleaned = _clean_reply_line(reply)
            if not cleaned:
                continue
            compact = _compact(cleaned)
            if substantive and _is_generic_low_effort(cleaned):
                continue
            if _contains_unsupported_jargon(cleaned, context):
                continue
            if complex_requirement and len(cleaned) < 12:
                continue
            if complex_requirement and not (_text_tokens(cleaned) & context_tokens):
                continue
            result.append(cleaned)
        return result


def latest_meaningful_message(conversation_text: str) -> tuple[str, str]:
    for role, text in reversed(_parse_messages(conversation_text)):
        if _is_meaningful_text(text):
            return role, text
    return "", ""


def last_meaningful_other_message(conversation_text: str) -> str:
    for role, text in reversed(_parse_messages(conversation_text)):
        if role == "对方" and _is_meaningful_text(text):
            return text
    return ""


def unreplied_other_messages(conversation_text: str) -> list[str]:
    messages = [(role, text) for role, text in _parse_messages(conversation_text) if _is_meaningful_text(text)]
    if not messages or messages[-1][0] != "对方":
        return []
    last_own = -1
    for index, (role, _text) in enumerate(messages):
        if role == "我":
            last_own = index
    return [text for role, text in messages[last_own + 1 :] if role == "对方"]


def select_best_reply_for_context(conversation_text: str, replies: list[str]) -> str:
    latest_role, _latest_text = latest_meaningful_message(conversation_text)
    if latest_role == "我":
        return ""
    filtered = ReplyEngine(AppSettings(), StyleMemory())._quality_filter(replies, conversation_text)
    if not filtered:
        return ""
    last_other = last_meaningful_other_message(conversation_text)
    own_texts = {text for role, text in _parse_messages(conversation_text) if role == "我"}
    intent = classify_intent(conversation_text)
    scored: list[tuple[int, int, str]] = []
    context_tokens = _text_tokens(conversation_text)
    for index, reply in enumerate(filtered):
        score = 0
        if any(_is_same_or_contained_reply(reply, own) for own in own_texts):
            continue
        if _similar(_compact(reply), _compact(last_other)) > 0.82:
            score -= 100
        score += len(_text_tokens(reply) & context_tokens) * 5
        if _is_generic_low_effort(reply):
            score -= 30
        if intent == "food_share":
            score += _keyword_score(reply, ["香", "好吃", "上头", "想吃", "辣", "麻", "米线"], 8)
        elif intent == "venting":
            score += _keyword_score(reply, ["确实", "绷", "血压", "离谱", "普通员工"], 8)
        elif intent == "job_card":
            score += _keyword_score(reply, ["岗位", "五险", "机会", "转给", "招聘"], 8)
        elif intent == "how_to_receive":
            score += _keyword_score(reply, ["领取", "方法", "位置", "放", "使用"], 8)
        scored.append((score, -index, reply))
    scored.sort(reverse=True)
    return scored[0][2] if scored and scored[0][0] > -80 else ""


def _infer_unreplied_other_messages(conversation_text: str) -> list[str]:
    messages = unreplied_other_messages(conversation_text)
    if messages:
        return messages
    last = last_meaningful_other_message(conversation_text)
    return [last] if last else []


def _parse_messages(conversation_text: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for raw in (conversation_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[可见内容理解]") or line.startswith("[鍙鍐呭鐞嗚В]"):
            continue
        match = re.match(r"^\[(对方|我|瀵规柟|鎴[^\]]*)\]\s*(.*)$", line)
        if match:
            role = _normalize_role(match.group(1))
            text = match.group(2).strip()
            result.append((role, text))
        elif result:
            role, prev = result[-1]
            result[-1] = (role, f"{prev} {line}".strip())
        else:
            result.append(("对方", line))
    return result


def _normalize_role(role: str) -> str:
    return "对方" if role.startswith("对方") or role.startswith("瀵规柟") else "我"


def _is_meaningful_text(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if re.fullmatch(r"\d{1,3}\s*['\"]", cleaned):
        return False
    if re.fullmatch(r"(昨天|今天|星期.|周.)?\s*\d{1,2}:\d{2}", cleaned):
        return False
    if cleaned in {"[表情]", "[图片]", "[视频]", "[语音]", "动画表情"}:
        return False
    return True


def _clean_reply_line(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\s*\d+\s*[\.\)、]\s*", "", cleaned)
    cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned)
    if re.match(r"^(消息理解|回复建议|分析|原因|候选回复|对方在|当前需要回复|最近聊天记录)[:：]", cleaned):
        return ""
    cleaned = cleaned.strip("\"'“”")
    return cleaned.strip()


def _is_low_information_ack(text: str) -> bool:
    compact = _compact(text)
    return compact in {_compact(item) for item in LOW_INFORMATION_ACKS}


def _is_thanks_or_received_scene(text: str) -> bool:
    if _is_question(text):
        return False
    return any(word in text for word in ["谢谢", "谢啦", "收了", "收到", "拿到了"])


def _is_job_or_recruitment_card(text: str) -> bool:
    return any(word in text for word in ["招聘", "岗位", "五险", "一金", "招人"])


def _is_food_share_scene(text: str) -> bool:
    return any(word in text for word in ["米线", "麻麻", "好吃", "香", "辣", "菜", "饭"])


def _is_venting_scene(text: str) -> bool:
    return any(word in text for word in ["血压", "普通员工", "离谱", "绷不住", "烦", "吐槽"])


def _is_how_to_receive_scene(text: str) -> bool:
    if "领取" in text and any(word in text for word in ["使用", "方法", "步骤", "哪里", "怎么", "放"]):
        return True
    return any(word in text for word in ["收到这个该放哪里", "聚宝盆里面放", "放点什么"])


def _is_question(text: str) -> bool:
    return "?" in text or "？" in text or any(word in text for word in ["怎么", "什么", "哪里", "哪", "吗", "能不能", "可不可以"])


def _is_complex_requirement(text: str) -> bool:
    markers = ["想用", "希望", "需要", "能不能", "怎么", "自动", "机器人", "系统", "流程", "对接", "处理", "查单", "核销"]
    return len(text) >= 35 and sum(1 for marker in markers if marker in text) >= 2


def _is_substantive_context(text: str) -> bool:
    intent = classify_intent(text)
    if intent in {"low_ack", "thanks"}:
        return False
    return intent != "general" or len(_compact(text)) > 20


def _rule_replies(intent: str, target: str) -> list[str]:
    direct: dict[str, list[str]] = {
        "low_ack": ["嗯嗯", "好的", "可以，先这样"],
        "thanks": ["不客气，拿到就行", "收到就好", "哈哈不客气"],
        "food_share": ["看着就挺香的", "麻麻的是花椒味吗", "这碗米线有点上头"],
        "job_card": ["这个岗位看着还可以", "有五险一金还挺实在", "可以转给需要的人看看"],
        "venting": ["这谁看了不血压上来", "普通员工也太难了", "绷不住了这工作量"],
        "how_to_receive": ["领取方法很简单，我发你看下", "你看这个位置就行", "我把使用方法发你"],
    }
    if intent in direct:
        return direct[intent]
    seeded = replies_for_intent(intent)
    if seeded:
        return seeded
    if _is_complex_requirement(target):
        return [
            "这个可以做，核心是先识别对方意图，再接到上架和售后流程里",
            "可以先拆成意图识别、活动上架、查单核销这几步",
            "这个思路可行，但要先把商家发活动的字段整理清楚",
        ]
    return ["可以，我看下具体情况", "这个我确认下再回你", "我先看下，晚点回你"]


def _normalize_api_replies(replies: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for reply in replies:
        cleaned = _clean_generated_reply(reply)
        key = _compact(cleaned)
        if cleaned and key not in seen:
            normalized.append(cleaned)
            seen.add(key)
    return normalized[:3]


def _drop_parrot_replies(replies: list[str], last_message: str) -> list[str]:
    if not last_message:
        return replies
    return [reply for reply in replies if _similar(_compact(reply), _compact(last_message)) < 0.82]


def _filter_context_echo_replies(replies: list[str], context: str) -> list[str]:
    own_texts = [_compact(text) for role, text in _parse_messages(context) if role == "我"]
    result = []
    for reply in replies:
        compact = _compact(reply)
        if any(_similar(compact, own) > 0.86 for own in own_texts):
            continue
        result.append(reply)
    return result


def _merge_unique_replies(existing: list[str], incoming: list[str]) -> list[str]:
    result = list(existing)
    seen = {_compact(item) for item in result}
    for item in incoming:
        key = _compact(item)
        if key and key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _refill_messages(messages: list[dict[str, str]], collected: list[str], needed: int) -> list[dict[str, str]]:
    cloned = [dict(item) for item in messages]
    cloned.append(
        {
            "role": "user",
            "content": (
                f"上一次只有{len(collected)}条可用回复，还需要补充{needed}条。"
                "不要重复已有回复，不要输出敷衍句，继续只输出编号候选。"
            ),
        }
    )
    return cloned


def _candidate_models(model: str) -> list[str]:
    models = [model]
    if model != "glm-4-flash-250414":
        models.append("glm-4-flash-250414")
    return models


def _response_error_detail(response: object) -> str:
    text = str(getattr(response, "text", "") or "").strip()
    return text[:500] if text else "empty response"


def _contains_unsupported_jargon(reply: str, context: str) -> bool:
    return any(word in reply and word not in context for word in UNSUPPORTED_JARGON)


def _is_generic_low_effort(reply: str) -> bool:
    compact = _compact(reply)
    generic = {_compact(item) for item in GENERIC_LOW_EFFORT}
    if compact in generic:
        return True
    return any(compact.startswith(item) and len(compact) <= len(item) + 2 for item in generic if item)


def _clean_generated_reply(text: str, max_chars: int = 70) -> str:
    result = _clean_reply_line(text)
    result = re.sub(r"\s+", " ", result).strip(" ，。；;")
    if len(result) > max_chars:
        parts = re.split(r"[，。；;！!？?]", result)
        result = "，".join(part.strip() for part in parts[:2] if part.strip()) or result[:max_chars]
    return result[:max_chars].rstrip("，。；; ")


def _keyword_score(text: str, words: Iterable[str], weight: int) -> int:
    return sum(weight for word in words if word in text)


def _text_tokens(text: str) -> set[str]:
    cleaned = _compact(re.sub(r"\[(?:对方|我|瀵规柟|鎴[^\]]*)\]", "", text or ""))
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z0-9_]{2,}", cleaned))
    for size in (2, 3, 4):
        for index in range(max(0, len(cleaned) - size + 1)):
            piece = cleaned[index : index + size]
            if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
                tokens.add(piece)
    return tokens


def _compact(text: str) -> str:
    return re.sub(r"[\W_]+", "", text or "").lower()


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _is_same_or_contained_reply(reply: str, old_text: str) -> bool:
    left = _compact(reply)
    right = _compact(old_text)
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    return _similar(left, right) > 0.82


_ORIGINAL_QUALITY_FILTER = ReplyEngine._quality_filter


def _readable_build_prompt(
    self: ReplyEngine,
    conversation_text: str,
    partner: str = "",
    target_turn: list[str] | None = None,
) -> str:
    target_messages = [message.strip() for message in (target_turn or []) if message and message.strip()]
    if not target_messages:
        target_messages = _infer_unreplied_other_messages(conversation_text)
    latest_target = target_messages[-1] if target_messages else last_meaningful_other_message(conversation_text)
    target_text = "\n".join(target_messages).strip()
    complex_requirement = _is_complex_requirement(target_text or conversation_text)
    length_rule = (
        "复杂需求、方案咨询允许每条 45-70 字；普通闲聊尽量 30 字内。"
        if complex_requirement
        else "每条尽量不超过 30 个中文字符，像微信里随手发的话。"
    )
    source_rule = (
        "回复来源：大模型。你必须自己理解整句和上下文，不要按关键词套模板，不要参考本地话术库。"
        if self.settings.reply_source_mode == "model"
        else "回复来源：大模型 + 话术库。话术库只提供历史表达参考，当前回复必须重新理解上下文。"
    )
    memory_block = "" if self.settings.reply_source_mode == "model" else self.memory.prompt_block(conversation_text)
    memory_rule = (
        "7. 不使用历史话术库，只根据当前上下文自然生成。"
        if self.settings.reply_source_mode == "model"
        else "7. 优先参考相似历史聊天片段与真实回复的接话方式，但不要照抄。"
    )
    visible_rule = (
        "10. 如果上下文包含[可见内容理解]，它是图片/卡片/链接预览的可见内容说明；"
        "必须结合图片/卡片内容和最后一句文字来回复，可以使用其中的可见信息；"
        "不要只按文字或只按图片泛泛回复。"
        if "[可见内容理解]" in conversation_text
        else ""
    )
    image_detail_rule = (
        "5. 已有[可见内容理解]时，可以引用其中的可见信息；不在可见内容里的细节不要编造。"
        if "[可见内容理解]" in conversation_text
        else "5. 不要假装看过链接或图片细节；看不清就让对方补充或说我确认下。"
    )
    return f"""当前聊天对象：{partner or "当前聊天联系人"}

最近聊天记录，越下面越新：
{conversation_text.strip() or "（没有识别到聊天内容）"}

当前需要回复的未回复轮次：
{target_text or latest_target or "（使用最后一轮未回复对方消息）"}

本轮最后一句（最高优先级）：
{latest_target or "（无）"}

{memory_block}

回复要求：
1. 整段上下文只作参考，对方发的内容优先级更高，对方最后一句优先级最高。
2. 如果对方有多个未回复问题，每个未回复的问题都要照顾到，并合并成一条自然回复。
3. 如果前面的问题我已经回复过，只参考背景，只回复最后一句。
4. 不能编造，不知道就说需要确认；不能自己编数字、金额、订单量、比例、时间或效果数据。
{image_detail_rule}
6. 不要输出“理解了 / 帮你看看 / 怎么弄？”这类敷衍句。
{memory_rule}
8. {length_rule}
9. {source_rule}
{visible_rule}

请输出 3 条候选，只要候选本身：
1. xxx
2. xxx
3. xxx"""


def _readable_generate_with_current_model(
    self: ReplyEngine,
    conversation_text: str,
    partner: str,
    target_turn: list[str] | None,
) -> tuple[list[str], int]:
    collected: list[str] = []
    call_count = 0
    prompt = self._build_prompt(conversation_text, partner, target_turn=target_turn)
    base_messages = [
        {
            "role": "system",
            "content": (
                "你是微信回复建议器。只输出 3 条候选回复，每条一行并用 1/2/3 编号。"
                "必须优先回答当前未回复轮次和最后一句，不能复述对方原句，不能编造事实、数字或效果数据。"
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        messages = base_messages if attempt == 0 else _refill_messages(base_messages, collected, 3 - len(collected))
        content = self._request_model_content(messages)
        call_count += 1
        if not (content or "").strip():
            self.last_error = (
                f"{provider_source_label(self.settings)} 返回里没有识别到文本，"
                "请检查接口、模型或自定义返回文本路径"
            )
            return [], call_count
        raw = parse_numbered_replies(content)
        raw = _drop_parrot_replies(raw, last_meaningful_other_message("\n".join(target_turn or []) or conversation_text))
        raw = self._quality_filter(raw, "\n".join(target_turn or []) or conversation_text)
        raw = _filter_context_echo_replies(raw, conversation_text)
        collected = _merge_unique_replies(collected, raw)
        normalized = _normalize_api_replies(collected)
        if len(normalized) >= 3:
            return normalized[:3], call_count

    normalized = _normalize_api_replies(collected)
    if normalized:
        return normalized[:3], call_count
    self.last_error = (
        f"{provider_source_label(self.settings)} 返回内容未通过质量过滤，"
        "可能是模型输出太敷衍、编造数据，或没有按编号返回候选"
    )
    self.last_source = provider_source_label(self.settings)
    return [], call_count


def _quality_filter_with_numeric_guard(self: ReplyEngine, replies: list[str], context: str) -> list[str]:
    filtered = _ORIGINAL_QUALITY_FILTER(self, replies, context)
    return [reply for reply in filtered if not _reply_invents_numeric_data(reply, context)]


def _reply_invents_numeric_data(reply: str, context: str) -> bool:
    if not _asks_for_numeric_or_effect_data(context):
        return False
    context_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", context or ""))
    reply_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", reply or ""))
    if reply_numbers - context_numbers:
        return True
    chinese_number_markers = ["几十", "几百", "几千", "上万", "几万", "十几", "百分之", "一半", "翻倍"]
    return any(marker in reply and marker not in context for marker in chinese_number_markers)


def _asks_for_numeric_or_effect_data(context: str) -> bool:
    text = context or ""
    question_words = ["多少", "几个", "几单", "多大", "多高", "数据", "订单", "转化", "比例", "金额", "效果", "销量", "能带来"]
    return any(word in text for word in question_words)


def _refill_messages(messages: list[dict[str, str]], collected: list[str], needed: int) -> list[dict[str, str]]:
    cloned = [dict(item) for item in messages]
    cloned.append(
        {
            "role": "user",
            "content": (
                f"上一轮只有 {len(collected)} 条可用回复，还需要补充 {needed} 条。"
                "不要重复已有回复，不要敷衍，不要编造数字、金额、订单量、比例或效果数据。"
                "继续只输出编号候选。"
            ),
        }
    )
    return cloned


ReplyEngine._build_prompt = _readable_build_prompt
ReplyEngine._generate_with_current_model = _readable_generate_with_current_model
ReplyEngine._quality_filter = _quality_filter_with_numeric_guard
