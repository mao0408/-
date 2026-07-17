from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory import StyleMemory


LOCAL_TZ = timezone(timedelta(hours=8))

SOURCE_NAMES = {
    "selected_candidate": "候选发送",
    "manual_edit": "手动编辑发送",
    "observed_chat": "训练模式观察",
    "chat_zip_training": "导入话术",
    "wechat_export_training": "聊天记录导入",
    "sop_analysis_training": "SOP分析生成",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export replies by day.")
    parser.add_argument("--date", help="日期，例如 2026-07-09。不填则导出今天。")
    parser.add_argument("--all", action="store_true", help="按日期导出全部回复。")
    parser.add_argument("--open", action="store_true", help="导出后用记事本打开。")
    args = parser.parse_args()

    memory = StyleMemory().load()
    output_dir = memory.path.parent / "每日回复"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _group_by_local_day(memory.examples)
    if args.all:
        written = [_write_day(output_dir, day, items) for day, items in sorted(grouped.items())]
        print(f"已导出 {len(written)} 个日期到：{output_dir}")
        for path in written:
            print(path)
        if args.open and written:
            _open_notepad(written[-1])
        return

    day = args.date or datetime.now(LOCAL_TZ).date().isoformat()
    path = _write_day(output_dir, day, grouped.get(day, []))
    print(f"已导出：{path}")
    print(f"回复条数：{len(grouped.get(day, []))}")
    if args.open:
        _open_notepad(path)


def _group_by_local_day(examples: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in examples:
        reply = str(item.get("reply", "")).strip()
        if not reply:
            continue
        day = _local_day(str(item.get("created_at", "")))
        grouped[day].append(item)
    return grouped


def _local_day(value: str) -> str:
    parsed = _parse_datetime(value)
    return parsed.astimezone(LOCAL_TZ).date().isoformat()


def _parse_datetime(value: str) -> datetime:
    text = (value or "").strip()
    if not text:
        return datetime.now(LOCAL_TZ)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(LOCAL_TZ)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed


def _write_day(output_dir: Path, day: str, items: list[dict[str, str]]) -> Path:
    path = output_dir / f"{day}_回复记录.txt"
    lines: list[str] = []
    lines.append(f"{day} 回复记录")
    lines.append("=" * 32)
    lines.append(f"共 {len(items)} 条")
    lines.append("")

    if not items:
        lines.append("这一天没有收集到回复。")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    for index, item in enumerate(sorted(items, key=lambda row: str(row.get("created_at", ""))), 1):
        time_label = _local_time(str(item.get("created_at", "")))
        partner = str(item.get("partner", "") or "未知对象").strip()
        source = SOURCE_NAMES.get(str(item.get("source", "")), str(item.get("source", "")) or "未知来源")
        reply = _clean_text(item.get("reply", ""))
        cue = _clean_context(item.get("cue", ""))

        lines.append(f"【{index:03d}】{time_label}｜{source}｜{partner}")
        lines.append("回复：")
        lines.append(reply or "（空）")
        lines.append("")
        lines.append("当时上下文：")
        lines.append(cue or "（无）")
        lines.append("")
        lines.append("-" * 32)
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _local_time(value: str) -> str:
    return _parse_datetime(value).astimezone(LOCAL_TZ).strftime("%H:%M:%S")


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _clean_context(value: object) -> str:
    lines = []
    for line in str(value or "").splitlines():
        cleaned = _clean_text(line)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _open_notepad(path: Path) -> None:
    subprocess.Popen(["notepad.exe", str(path)])


if __name__ == "__main__":
    main()
