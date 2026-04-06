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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from transcribe import transcribe_audio, preload_model
from minutes import generate_minutes, save_minutes

load_dotenv()

TEMP_DIR = Path("temp")
OUTPUTS_DIR = Path("outputs")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ローカルWhisperモデルを起動時にプリロード
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, preload_model)
    yield


app = FastAPI(title="音声議事録ツール", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    pending_audio: bytes | None = None  # 直前に受け取ったバイナリ
    final_transcript: str = ""
    whisper_mode = os.getenv("WHISPER_MODE", "local")

    print(f"[{session_id}] WebSocket接続")

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

            # バイナリ（音声データ）：次のJSONコマンドと対になる
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
                    # リアルタイム文字起こし（15秒チャンク）
                    try:
                        text = await transcribe_pending("partial")
                        if text.strip():
                            await websocket.send_json({"type": "transcript_partial", "text": text})
                    except Exception as e:
                        # リアルタイム失敗は警告のみ、録音継続
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

                    # 全音声の最終文字起こし
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

                    # 議事録生成
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
