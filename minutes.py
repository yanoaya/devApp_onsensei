"""
議事録生成モジュール
Anthropic Claude API を使用して文字起こしから議事録を生成
"""
import os
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
あなたは優秀な議事録作成アシスタントです。
提供された会議の文字起こしを分析し、構造化された議事録をMarkdown形式で作成してください。
文字起こしに含まれる情報のみを使用し、推測や補完は最小限にとどめてください。
参加者が不明な場合は「不明」と記載してください。
"""

MINUTES_TEMPLATE = """\
以下の会議の文字起こしから議事録を作成してください。

【文字起こし】
{transcript}

【出力形式】
以下のMarkdown形式で出力してください：

# 議事録
**日時**: {datetime}
**参加者**: （文字起こしから推定。不明な場合は「不明」）

## アジェンダ
（会議で扱われたトピックを箇条書きで）

## 議論のサマリー
（主要な議論内容を簡潔にまとめる）

## 決定事項
（会議で決定・合意された事項を箇条書きで。なければ「特になし」）

## アクションアイテム
（担当者・期限・タスクを箇条書きで。なければ「特になし」）

## 次回MTGについて
（次回MTGの言及があれば記載。なければこのセクションは省略）
"""


async def generate_minutes(transcript: str) -> str:
    """
    文字起こしテキストから議事録を生成する。
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY が設定されていません")

    if not transcript.strip():
        raise ValueError("文字起こしテキストが空です")

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    user_message = MINUTES_TEMPLATE.format(
        transcript=transcript,
        datetime=now,
    )

    message = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    return message.content[0].text


def save_minutes(content: str, output_dir: str = "outputs") -> str:
    """議事録をMarkdownファイルとして保存し、ファイルパスを返す"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minutes_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


def save_to_firestore(db, uid: str, transcript: str, minutes_text: str) -> str:
    """Firestore に議事録を保存して document ID を返す"""
    from firebase_admin import firestore as fb_fs
    title = _extract_title(minutes_text)
    _, doc_ref = db.collection("users").document(uid).collection("minutes").add({
        "transcript": transcript,
        "minutes_text": minutes_text,
        "title": title,
        "created_at": fb_fs.SERVER_TIMESTAMP,
    })
    return doc_ref.id


def list_from_firestore(db, uid: str) -> list[dict]:
    """Firestore からユーザーの議事録一覧を取得（新しい順）"""
    from google.cloud.firestore_v1 import Query as FSQuery
    docs = (
        db.collection("users").document(uid).collection("minutes")
        .order_by("created_at", direction=FSQuery.DESCENDING)
        .limit(20)
        .stream()
    )
    result = []
    for doc in docs:
        data = doc.to_dict()
        created_at = data.get("created_at")
        result.append({
            "id": doc.id,
            "title": data.get("title", "議事録"),
            "created_at": created_at.isoformat() if created_at else "",
            "minutes_text": data.get("minutes_text", ""),
        })
    return result


def _extract_title(minutes_text: str) -> str:
    """議事録テキストから1行目をタイトルとして抽出"""
    for line in minutes_text.split("\n"):
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:40]
    return "議事録"
