"""
音声文字起こしモジュール
ローカルWhisper または OpenAI Whisper API を使用
話者分離: resemblyzer + scikit-learn (HuggingFace不要)
"""
import os
import asyncio
import subprocess
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

WHISPER_MODE = os.getenv("WHISPER_MODE", "local").lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANG = os.getenv("WHISPER_LANG", "ja")
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "これは日本語の会議の録音です。参加者が話し合いを行っています。"
)
WHISPER_DIARIZE = os.getenv("WHISPER_DIARIZE", "false").lower() == "true"

# モデルキャッシュ
_local_model = None
_voice_encoder = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        import whisper
        print(f"[transcribe] Whisperモデル '{WHISPER_MODEL}' をロード中...")
        _local_model = whisper.load_model(WHISPER_MODEL)
        print(f"[transcribe] ロード完了")
    return _local_model


def _get_voice_encoder():
    global _voice_encoder
    if _voice_encoder is None:
        from resemblyzer import VoiceEncoder
        print(f"[transcribe] VoiceEncoderをロード中...")
        _voice_encoder = VoiceEncoder()
        print(f"[transcribe] VoiceEncoderロード完了")
    return _voice_encoder


async def transcribe_audio(audio_path: str, mode: str | None = None) -> str:
    """
    音声ファイルを文字起こしする。
    mode: 'local' | 'api' | None (環境変数から取得)
    ffmpegで16kHzモノラルWAVに正規化してからWhisperに渡す。
    """
    effective_mode = mode or WHISPER_MODE
    normalized_path = await _normalize_to_wav(audio_path)
    try:
        if effective_mode == "api":
            return await _transcribe_with_api(normalized_path)
        else:
            return await _transcribe_local(normalized_path)
    finally:
        Path(normalized_path).unlink(missing_ok=True)


async def _normalize_to_wav(audio_path: str) -> str:
    """ffmpegで16kHz/モノラルWAVに正規化する"""
    p = Path(audio_path)
    normalized = str(p.with_name(p.stem + "_norm.wav"))
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        normalized,
    ]
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_ffmpeg, cmd)
    return normalized


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg変換失敗: {result.stderr.decode()}")


async def _transcribe_local(audio_path: str) -> str:
    """ローカルWhisperで文字起こし（ブロッキング処理を別スレッドで実行）"""
    loop = asyncio.get_event_loop()
    if WHISPER_DIARIZE:
        result = await loop.run_in_executor(None, _run_diarized_whisper, audio_path)
    else:
        result = await loop.run_in_executor(None, _run_local_whisper, audio_path)
    return result


def _run_local_whisper(audio_path: str) -> str:
    model = _get_local_model()
    options = {
        "temperature": 0,          # ハルシネーション抑制
        "fp16": False,             # CPU/MPS で安定動作
        "condition_on_previous_text": False,  # チャンク間の誤伝播を防ぐ
        "no_speech_threshold": 0.5,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
    }
    if WHISPER_LANG:
        options["language"] = WHISPER_LANG
    if WHISPER_PROMPT:
        options["initial_prompt"] = WHISPER_PROMPT
    result = model.transcribe(audio_path, **options)
    return result["text"].strip()


def _run_diarized_whisper(audio_path: str) -> str:
    """openai-whisper + resemblyzer で話者分離つき文字起こし"""
    import numpy as np
    from resemblyzer import preprocess_wav
    from sklearn.cluster import AgglomerativeClustering

    # 1. Whisperでセグメント単位の文字起こし
    model = _get_local_model()
    options = {
        "temperature": 0,
        "fp16": False,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.5,
    }
    if WHISPER_LANG:
        options["language"] = WHISPER_LANG
    if WHISPER_PROMPT:
        options["initial_prompt"] = WHISPER_PROMPT

    whisper_result = model.transcribe(audio_path, **options)
    segments = whisper_result.get("segments", [])
    if not segments:
        return ""

    # 2. resemblyzerで各セグメントの話者埋め込みを取得
    try:
        encoder = _get_voice_encoder()
        wav = preprocess_wav(audio_path)
        sample_rate = 16000

        embeddings = []
        for seg in segments:
            start = int(seg["start"] * sample_rate)
            end = int(seg["end"] * sample_rate)
            clip = wav[start:end]
            # 短すぎるクリップはスキップ（resemblyzerの最小要件）
            if len(clip) < sample_rate * 0.5:
                embeddings.append(None)
            else:
                embeddings.append(encoder.embed_utterance(clip))

        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
        if len(valid_indices) < 2:
            return whisper_result["text"].strip()

        valid_embeddings = np.array([embeddings[i] for i in valid_indices])

        # 3. コサイン距離で階層クラスタリング（話者数自動推定）
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.45,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(valid_embeddings)

        speaker_map = {i: f"SPEAKER_{label:02d}" for i, label in zip(valid_indices, labels)}

    except Exception as e:
        print(f"[transcribe] 話者分離失敗、テキストのみ返します: {e}")
        return whisper_result["text"].strip()

    # 4. テキスト整形
    lines = []
    current_speaker = None
    current_texts = []

    for i, seg in enumerate(segments):
        speaker = speaker_map.get(i, "SPEAKER_??")
        text = seg["text"].strip()
        if not text:
            continue
        if speaker != current_speaker:
            if current_speaker and current_texts:
                lines.append(f"{current_speaker}: {''.join(current_texts)}")
            current_speaker = speaker
            current_texts = [text]
        else:
            current_texts.append(text)

    if current_speaker and current_texts:
        lines.append(f"{current_speaker}: {''.join(current_texts)}")

    return "\n".join(lines)


async def _transcribe_with_api(audio_path: str) -> str:
    """OpenAI Whisper APIで文字起こし"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY が設定されていません")

    client = openai.AsyncOpenAI(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": "whisper-1",
            "file": audio_file,
        }
        if WHISPER_LANG:
            kwargs["language"] = WHISPER_LANG

        response = await client.audio.transcriptions.create(**kwargs)

    return response.text.strip()


def preload_model():
    """サーバー起動時にローカルモデルをプリロード"""
    if WHISPER_MODE == "local":
        _get_local_model()
        if WHISPER_DIARIZE:
            _get_voice_encoder()
