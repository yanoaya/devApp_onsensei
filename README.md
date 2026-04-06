# 音声議事録ツール（リアルタイム版）

マイクで録音した音声をリアルタイムでWhisperが文字起こしし、録音停止後にClaude AIが議事録を自動生成するWebアプリです。

---

## 必要条件

- **Python 3.10 以上**
- **ffmpeg**（Whisperが音声変換に使用）

```bash
brew install ffmpeg
```

---

## インストール手順

```bash
cd meeting-minutes

# 仮想環境の作成（推奨）
python3 -m venv .venv
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

> ローカルWhisperモデルの初回ダウンロードは自動で行われます（`base`モデル：約140MB）。

---

## .env の設定

`.env.example` をコピーして `.env` を作成し、APIキーを設定してください。

```bash
cp .env.example .env
```

```dotenv
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx   # 必須：議事録生成に使用
OPENAI_API_KEY=sk-xxxxxxxx          # オプション：Whisper APIモード使用時のみ

WHISPER_MODE=local    # local（オフライン） or api（OpenAI Whisper API）
WHISPER_MODEL=base    # base / small / medium / large（localモード時）
WHISPER_LANG=ja       # 文字起こし言語（ja = 日本語固定）
CHUNK_SECONDS=30      # 何秒ごとに文字起こしするか
```

---

## 起動方法

```bash
uvicorn main:app --reload
```

ブラウザで [http://localhost:8000](http://localhost:8000) を開いてください。

---

## Insta360 Mic Air の設定方法

### Mac のシステム設定で指定する場合

1. **システム設定** → **サウンド** → **入力**
2. デバイス一覧から **Insta360 Mic Air** を選択
3. 入力音量を適切に調整（70〜80%推奨）

### ブラウザのデバイス選択で指定する場合

アプリ上部の **「マイクデバイス」** ドロップダウンから直接 **Insta360 Mic Air** を選択できます（より確実な方法です）。

---

## ブラウザのマイク許可について

- Chrome / Edge：アドレスバーの鍵アイコン → **サイトの設定** → **マイク** → **許可**
- Safari：**環境設定** → **Webサイト** → **マイク** → `localhost` を「許可」
- **HTTPS 環境では自動的に許可が求められます。**  
  ローカル（`http://localhost`）では通常許可なしでアクセス可能です。

---

## 使い方

1. ブラウザで `http://localhost:8000` を開く
2. マイクデバイスを選択
3. Whisperモードを選択（ローカル推奨）
4. **「録音開始」** をクリック → マイク許可を承認
5. 会議を進める。30秒ごとに文字起こし結果が画面に追記される
6. **「録音停止」** をクリック → 数秒待つと議事録が生成・表示される
7. **「Markdownダウンロード」** で議事録を保存

生成された議事録は `outputs/` ディレクトリにも自動保存されます。

---

## ディレクトリ構成

```
meeting-minutes/
├── main.py          # FastAPI サーバー + WebSocketエンドポイント
├── transcribe.py    # Whisper文字起こしモジュール
├── minutes.py       # Claude議事録生成モジュール
├── static/
│   └── index.html   # フロントエンド（単一ファイル）
├── temp/            # 音声チャンク一時保存（処理後自動削除）
├── outputs/         # 生成された議事録の保存先
├── .env.example
├── requirements.txt
└── README.md
```

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| 「マイクへのアクセスが拒否されました」 | ブラウザのマイク許可設定を確認 |
| 文字起こしが空になる | ffmpegがインストールされているか確認（`ffmpeg -version`） |
| ローカルモデルの読み込みが遅い | `small` → `base` にモデルを変更 |
| 議事録生成に失敗する | `.env` の `ANTHROPIC_API_KEY` が正しく設定されているか確認 |
