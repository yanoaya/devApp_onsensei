"""
FastAPI サーバー + WebSocket エンドポイント
音声チャンクを受け取りリアルタイムで文字起こし、録音終了後に議事録を生成
"""
import asyncio
import json
import os
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

import firebase_admin
from firebase_admin import auth as firebase_auth, firestore as fb_firestore
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Query as QueryParam
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from transcribe import transcribe_audio, preload_model
from minutes import generate_minutes, save_minutes, save_to_firestore, list_from_firestore

load_dotenv()

TEMP_DIR = Path("temp")
OUTPUTS_DIR = Path("outputs")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

db = None


async def _verify_token(token: str) -> str:
    """Firebase ID token を検証して uid を返す"""
    loop = asyncio.get_event_loop()
    decoded = await loop.run_in_executor(None, firebase_auth.verify_id_token, token)
    return decoded["uid"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    firebase_admin.initialize_app()
    db = fb_firestore.client()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, preload_model)
    yield


app = FastAPI(title="音声議事録ツール", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/minutes")
async def get_minutes_list(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="認証が必要です")
    token = authorization[7:]
    try:
        uid = await _verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="トークンが無効です")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, list_from_firestore, db, uid)
    return {"minutes": result}


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, token: str = QueryParam(None)):
    await websocket.accept()

    # 認証
    uid = None
    try:
        uid = await _verify_token(token or "")
    except Exception:
        await websocket.send_json({"type": "error", "message": "認証に失敗しました。再ログインしてください。"})
        await websocket.close()
        return

    session_id = str(uuid.uuid4())[:8]
    pending_audio: bytes | None = None
    final_transcript: str = ""
    whisper_mode = os.getenv("WHISPER_MODE", "local")

    print(f"[{session_id}] WebSocket接続 uid={uid}")

    async def transcribe_pending(label: str) -> str:
        """pending_audio をファイルに書いて文字起こし、結果を返す"""
        if not pending_audio:
            return ""
        path = TEMP_DIR / f"{session_id}_{label}.wav"
        try:
            with open(path, "wb") as f:
                f.write(pending_audio)
            text = await transcribe_audio(str(path), mode=whisper_mode)
            print(f"[{session_id}] [{label}] {len(text)}文字: {text[:60]}...")
            return text
        except Exception as e:
            print(f"[{session_id}] [{label}] 文字起こしエラー: {e}")
            raise
        finally:
            path.unlink(missing_ok=True)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                pending_audio = message["bytes"]
                print(f"[{session_id}] 音声受信 ({len(pending_audio):,} bytes)")

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "set_mode":
                    whisper_mode = data.get("mode", whisper_mode)
                    print(f"[{session_id}] Whisperモード: {whisper_mode}")

                elif msg_type == "partial":
                    try:
                        text = await transcribe_pending("partial")
                        if text.strip():
                            await websocket.send_json({"type": "transcript_partial", "text": text})
                    except Exception as e:
                        await websocket.send_json({
                            "type": "warning",
                            "message": f"一部の文字起こしに失敗しました: {str(e)}",
                        })

                elif msg_type == "end":
                    print(f"[{session_id}] 終了シグナル受信")

                    if not pending_audio:
                        await websocket.send_json({
                            "type": "error",
                            "message": "音声データが受信できませんでした。",
                        })
                        break

                    try:
                        final_transcript = await transcribe_pending("final")
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"文字起こしに失敗しました: {str(e)}",
                        })
                        break

                    if not final_transcript.strip():
                        await websocket.send_json({
                            "type": "error",
                            "message": "文字起こし結果が空です。音声が録音されているか確認してください。",
                        })
                        break

                    await websocket.send_json({"type": "transcript", "text": final_transcript})

                    try:
                        await websocket.send_json({"type": "generating_minutes"})
                        minutes_text = await generate_minutes(final_transcript)
                        save_minutes(minutes_text, str(OUTPUTS_DIR))
                        await websocket.send_json({"type": "minutes", "text": minutes_text})
                        print(f"[{session_id}] 議事録生成完了")
                    except Exception as e:
                        print(f"[{session_id}] 議事録生成エラー: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"議事録の生成に失敗しました: {str(e)}",
                        })
                        break

                    # Firestore 保存（失敗しても続行）
                    try:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, save_to_firestore, db, uid, final_transcript, minutes_text
                        )
                        print(f"[{session_id}] Firestore保存完了")
                    except Exception as e:
                        print(f"[{session_id}] Firestore保存エラー: {e}")

                    break

    except WebSocketDisconnect:
        print(f"[{session_id}] WebSocket切断")
    except Exception as e:
        print(f"[{session_id}] 予期しないエラー: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"サーバーエラーが発生しました: {str(e)}",
            })
        except Exception:
            pass
    finally:
        print(f"[{session_id}] セッション終了")
