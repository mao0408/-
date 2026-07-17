from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory import StyleMemory


SOURCE_NAMES = {
    "selected_candidate": "手动发送候选回复",
    "manual_edit": "手动编辑后发送",
    "observed_chat": "训练模式观察",
    "chat_zip_training": "导入话术包",
    "wechat_export_training": "微信聊天记录导入",
    "sop_analysis_training": "SOP 分析生成",
    "uploaded_training_analysis": "上传训练分析",
}


def main() -> None:
    memory = StyleMemory().load()
    out_dir = memory.path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = [item for item in memory.examples if _clean(item.get("reply", ""))]
    summary_path = out_dir / "话术总结.txt"
    detail_path = out_dir / "话术明细.txt"

    summary_path.write_text(_build_summary(memory, examples), encoding="utf-8")
    detail_path.write_text(_build_details(examples), encoding="utf-8")

    print(f"已导出：{summary_path}")
    print(f"已导出：{detail_path}")
    print(f"话术条数：{len(examples)}")


def _build_summary(memory: StyleMemory, examples: list[dict[str, str]]) -> str:
    source_counts = Counter(_source_name(item.get("source", "")) for item in examples)
    reply_counts = Counter(
        reply
        for reply in (_clean(item.get("reply", "")) for item in examples)
        if _is_useful_reply(reply)
    )
    phrase_counts = _filtered_phrase_counts(memory.common_phrases)
    scenes = _group_scenes(examples)

    lines: list[str] = []
    lines.append("微信回复助手话术总结")
    lines.append("=" * 28)
    lines.append("")
    lines.append(f"记忆文件：{memory.path}")
    lines.append(f"话术总数：{len(examples)}")
    lines.append(f"风格摘要：{memory.style_summary}")
    lines.append("")

    lines.append("一、来源统计")
    for name, count in source_counts.most_common():
        lines.append(f"- {name}：{count} 条")
    lines.append("")

    lines.append("二、常用表达")
    if phrase_counts:
        for phrase, count in phrase_counts.most_common(40):
            lines.append(f"- {phrase}（{count}）")
    else:
        lines.append("- 暂无可用常用表达")
    lines.append("")

    lines.append("三、高频完整回复")
    for reply, count in reply_counts.most_common(40):
        lines.append(f"- {reply}（{count}）")
    lines.append("")

    lines.append("四、按场景归纳的话术")
    for scene, items in scenes.items():
        lines.append(f"【{scene}】{len(items)} 条")
        scene_replies = Counter(
            reply
            for reply in (_clean(item.get("reply", "")) for item in items)
            if _is_useful_reply(reply)
        )
        if not scene_replies:
            lines.append("- 暂无可总结的有效话术")
            lines.append("")
            continue
        for reply, count in scene_replies.most_common(12):
            lines.append(f"- {reply}（{count}）")
        lines.append("")

    lines.append("五、说明")
    lines.append("- 本文件是从 user_data\\memory.json 自动整理出来的文本版总结。")
    lines.append("- 话术明细.txt 保存每一条原始上下文和回复，方便人工筛选、删改、再训练。")
    lines.append("- 如果训练模式 OCR 识别不准，URL、数字、乱码也可能进入记忆；建议定期检查话术明细。")
    return "\n".join(lines).rstrip() + "\n"


def _build_details(examples: list[dict[str, str]]) -> str:
    lines: list[str] = []
    lines.append("微信回复助手话术明细")
    lines.append("=" * 28)
    lines.append("")
    for index, item in enumerate(examples, 1):
        lines.append(f"#{index}")
        lines.append(f"来源：{_source_name(item.get('source', ''))}")
        lines.append(f"对象：{_clean(item.get('partner', '')) or '未知'}")
        lines.append(f"时间：{_clean(item.get('created_at', '')) or '未知'}")
        if item.get("scenario_title"):
            lines.append(f"场景：{_clean(item.get('scenario_title', ''))}")
        if item.get("why"):
            lines.append(f"理由：{_clean(item.get('why', ''))}")
        lines.append("上下文：")
        lines.append(_clean_context(item.get("cue", "")) or "（空）")
        lines.append("回复：")
        lines.append(_clean(item.get("reply", "")) or "（空）")
        lines.append("")
        lines.append("-" * 28)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _group_scenes(examples: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in examples:
        scene = _clean(item.get("scenario_title", ""))
        text = f"{item.get('cue', '')}\n{item.get('reply', '')}"
        if not scene:
            scene = _infer_scene(text)
        groups[scene].append(item)
    return dict(sorted(groups.items(), key=lambda row: len(row[1]), reverse=True))


def _infer_scene(text: str) -> str:
    rules = [
        ("吃饭/邀约", ("吃饭", "来吃", "约", "几点", "晚上", "中午")),
        ("问题求助", ("怎么", "咋", "什么", "哪里", "能不能", "可不可以", "帮")),
        ("确认/收到", ("收到", "好的", "可以", "嗯", "行", "知道")),
        ("链接/资料", ("http", "github", "链接", "文档", "文件", "下载")),
        ("吐槽/情绪", ("烦", "离谱", "血压", "难受", "绷", "累")),
        ("工作/流程", ("系统", "流程", "客户", "订单", "核销", "售后", "岗位")),
    ]
    for name, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return name
    return "其他闲聊"


def _filtered_phrase_counts(counter: Counter) -> Counter:
    result: Counter = Counter()
    for phrase, count in counter.items():
        cleaned = _clean(str(phrase))
        if _is_useful_phrase(cleaned):
            result[cleaned] += int(count)
    return result


def _is_useful_phrase(text: str) -> bool:
    if len(text) < 2 or len(text) > 30:
        return False
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return False
    if re.search(r"https?://|www\.|条新消息|ZABS|Windows", text, re.I):
        return False
    if sum(ch.isdigit() for ch in text) >= max(2, len(text) // 2):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _is_useful_reply(text: str) -> bool:
    if len(text) < 2 or len(text) > 80:
        return False
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return False
    if re.fullmatch(r"\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{2}", text):
        return False
    if re.search(r"https?://|www\.|条新消息|ZABS|收起\\^|群聊名称|Windows 批处理文件|文本文档", text, re.I):
        return False
    if re.fullmatch(r"[\[\]A-Za-z0-9:：><._ -]+", text):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _source_name(source: str | None) -> str:
    key = _clean(source or "")
    return SOURCE_NAMES.get(key, key or "未知来源")


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_context(value: object) -> str:
    lines = []
    for line in str(value or "").splitlines():
        cleaned = _clean(line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
