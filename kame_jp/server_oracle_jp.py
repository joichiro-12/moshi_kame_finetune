"""Japanese KAME inference server (J-KAME).

Drop-in replacement for `kame.server_oracle` with three Japanese-specific overrides:

1. STT language code: "en-US" → "ja-JP"  (or configurable via --asr-language)
2. Backend LLM: hardcoded OpenAI gpt-4.1 → LLM-jp-4-8B via vLLM
   (any OpenAI-compatible endpoint configurable via --llm-base-url / --llm-model)
3. Harmony format parsing: strips <|channel|>analysis…<|channel|>final<|message|>…<|return|>
   so only the final answer reaches the oracle stream

The front-end S2S model and audio loop are **unchanged** from kame.server_oracle.

Usage:
    # Start vLLM server first:
    vllm serve llm-jp/llm-jp-4-8b-thinking \
        --reasoning-parser llmjp4 \
        --max-model-len 8192 \
        --port 8000

    # Then start J-KAME:
    uv run -m kame_jp.server_oracle_jp \
        --moshi-weight output/j-moshi-kame-finetuned/step_XXXXX_fp32_cleaned/model.safetensors \
        --config-path  output/j-moshi-kame-finetuned/step_XXXXX_fp32_cleaned/moshi_lm_kwargs.json \
        --tokenizer    /path/to/j-moshi-ext/tokenizer_spm_32k.model \
        --llm-base-url http://localhost:8000/v1 \
        --llm-model    llm-jp/llm-jp-4-8b-thinking \
        --port 8998
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
from typing import Any

import torch
from openai import AsyncOpenAI

# Import everything we want to reuse from the upstream kame server.
import kame.server_oracle as _upstream
from kame.server_oracle import (
    AsyncASRProcessor,
    ServerState,
    configure_save_dir,
    get_conversation_snapshot,
    log,
    seed_all,
)
from kame.models import loaders

# -----------------------
# Japanese server configuration
# -----------------------
JA_SYSTEM_PROMPT = """あなたは Moshi の日本語版です。ユーザーと会話しています。
ユーザーの発話の流れを予測し、適切な次の応答を生成してください。
会話テキストのみを出力してください。余分なコメントは不要です。
自信を持って話題に取り込み、確認を求めないこと。
回答は最大30語以内で簡潔にまとめてください。
出力される単語は音声に変換されるため、発音に関係のない記号（「" ー ;」など）は含めないこと。
"""

JA_ASR_DEFAULT_LANGUAGE = "ja-JP"

# Regex to extract the `final` channel from LLM-jp-4 Harmony output.
# Format: <|channel|>analysis<|message|>[CoT]<|channel|>final<|message|>[answer]<|return|>
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)",
    re.DOTALL,
)


def parse_harmony_response(raw: str) -> str:
    """Extract the final-channel text from LLM-jp-4-thinking Harmony format.

    Falls back to returning the raw string if no Harmony markup is found
    (handles llm-jp-4-instruct which does not use the thinking format).
    """
    m = _HARMONY_FINAL_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


# -----------------------
# Japanese ASR processor (ja-JP)
# -----------------------
class JaASRProcessor(AsyncASRProcessor):
    """AsyncASRProcessor with Japanese language code."""

    def __init__(self, sample_rate: int = 24000, language: str = JA_ASR_DEFAULT_LANGUAGE):
        self._ja_language = language
        # Temporarily patch the module-level constant so the parent __init__
        # picks up the Japanese language code.
        _orig = _upstream.ASR_LANGUAGE_CODE
        _upstream.ASR_LANGUAGE_CODE = language
        try:
            super().__init__(sample_rate=sample_rate)
        finally:
            _upstream.ASR_LANGUAGE_CODE = _orig


# -----------------------
# Japanese LLM stream manager (LLM-jp-4 via vLLM)
# -----------------------
class JaLLMStreamManager(_upstream.LLMStreamManager):
    """LLMStreamManager replacement that queries LLM-jp-4 via an OpenAI-compatible endpoint.

    Differences from the upstream:
    - Configurable base_url / model (vLLM endpoint)
    - Applies Harmony response parsing to extract the final-channel text
    - Uses Japanese system prompt by default
    """

    def __init__(
        self,
        server_state: Any,
        interval: float = 0.25,
        system_prompt: str = JA_SYSTEM_PROMPT,
        llm_base_url: str = "http://localhost:8000/v1",
        llm_model: str = "llm-jp/llm-jp-4-8b-thinking",
        llm_api_key: str = "EMPTY",
    ):
        # Bypass the parent __init__ which validates OPENAI_API_KEY unconditionally.
        # We replicate the necessary initialisation manually.
        self.server_state = server_state
        self.interval = interval
        self.system_prompt = system_prompt
        self.current_stream = None
        self.running = False

        self.llm_base_url = llm_base_url
        self.llm_model = llm_model

        resolved_key = llm_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=llm_base_url,
        )

        import collections

        self.last_start_time = 0.0
        self.restart_history = collections.deque(maxlen=10)
        self.last_start_total_units = 0
        self.min_units_delta = 2
        self.max_restarts_per_2s = 5

    async def _stream_llm_response(
        self, committed_conversation: str, pending_user_text: str
    ) -> None:
        """Stream LLM-jp-4 tokens, parse Harmony format, and enqueue oracle updates."""
        stream_start_ms = int(time.time() * 1000)
        stream_tokens: list[str] = []
        accumulated = ""

        try:
            messages: list[dict[str, Any]] = self._build_messages(
                committed_conversation, pending_user_text
            )
            stream = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
                # Hint vLLM to use low reasoning effort to keep within latency budget.
                extra_body={"reasoning_effort": "low"},
            )
            first_oracle_chunk = True

            async for chunk in stream:
                if not (
                    chunk.choices
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content
                ):
                    continue

                chunk_text = chunk.choices[0].delta.content
                accumulated += chunk_text

                # Parse Harmony on every new chunk to handle streaming format.
                final_text = parse_harmony_response(accumulated)

                # Only forward text that has cleared the analysis channel.
                stripped = final_text.strip()
                if not stripped:
                    continue

                # Find new text since the last enqueue
                new_text = stripped[len("".join(stream_tokens)):]
                if not new_text.strip():
                    continue

                if first_oracle_chunk:
                    try:
                        self.server_state.oracle_queue.put_nowait(("reset", ""))
                    except asyncio.QueueFull:
                        log("warning", "Oracle queue full; reset command dropped")
                    first_oracle_chunk = False

                word = new_text.strip()
                stream_tokens.append(word)
                log("info", f"[LLM-jp4] {word}")

                try:
                    self.server_state.oracle_queue.put_nowait(("append", word))
                except asyncio.QueueFull:
                    log("warning", "Oracle queue full; dropping LLM-jp4 chunk.")

            if stream_tokens:
                _upstream._append_session_log(
                    "llm_stream_words.txt",
                    f"{stream_start_ms}: {' '.join(stream_tokens)}\n",
                )

        except asyncio.CancelledError:
            if stream_tokens:
                _upstream._append_session_log(
                    "llm_stream_words.txt",
                    f"{stream_start_ms}: [CANCELLED] {' '.join(stream_tokens)}\n",
                )
            raise
        except Exception as e:
            log("error", f"LLM-jp4 streaming error: {e}")
            if stream_tokens:
                _upstream._append_session_log(
                    "llm_stream_words.txt",
                    f"{stream_start_ms}: [ERROR] {' '.join(stream_tokens)}\n",
                )


# -----------------------
# Japanese ServerState
# -----------------------
class JaServerState(ServerState):
    """ServerState with Japanese ASR and LLM-jp-4 backend.

    Overrides the ASR processor and LLM stream manager created in the parent
    __init__ without duplicating the model-loading logic.
    """

    def __init__(
        self,
        model_type: str,
        mimi: Any,
        text_tokenizer: Any,
        lm: Any,
        cfg_coef: float,
        device: str | torch.device,
        enable_asr: bool = True,
        asr_language: str = JA_ASR_DEFAULT_LANGUAGE,
        llm_base_url: str = "http://localhost:8000/v1",
        llm_model: str = "llm-jp/llm-jp-4-8b-thinking",
        llm_api_key: str = "EMPTY",
        **kwargs: Any,
    ):
        # Call the parent with enable_asr=False so it skips creating the
        # English ASR processor. We install the Japanese one afterwards.
        super().__init__(
            model_type=model_type,
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            lm=lm,
            cfg_coef=cfg_coef,
            device=device,
            enable_asr=False,  # we replace it below
            **kwargs,
        )

        # Replace ASR processor with Japanese version
        if enable_asr:
            self.asr_processor = JaASRProcessor(
                sample_rate=int(self.mimi.sample_rate),
                language=asr_language,
            )

        # Replace LLM stream manager with LLM-jp-4 version
        self.llm_stream_manager = JaLLMStreamManager(
            server_state=self,
            interval=0.5,
            system_prompt=JA_SYSTEM_PROMPT,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )


# -----------------------
# Entry point
# -----------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Japanese KAME inference server (J-KAME).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Re-use upstream args
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--moshi-weight", type=str, required=True)
    parser.add_argument("--mimi-weight", type=str, default=None)
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
    )
    parser.add_argument("--lora-weight", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--cfg-coef", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
    )
    parser.add_argument(
        "--no_fuse_lora",
        action="store_false",
        dest="fuse_lora",
        default=True,
    )
    parser.add_argument(
        "--enable-asr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ssl",
        type=str,
        default=None,
    )
    # Japanese-specific args
    parser.add_argument(
        "--asr-language",
        type=str,
        default=JA_ASR_DEFAULT_LANGUAGE,
        help="BCP-47 language code for Google STT (e.g. ja-JP, en-US).",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the OpenAI-compatible LLM endpoint (vLLM).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llm-jp/llm-jp-4-8b-thinking",
        help="Model name passed to the vLLM endpoint.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default="EMPTY",
        help='API key for the LLM endpoint ("EMPTY" for local vLLM).',
    )

    args = parser.parse_args()
    seed_all(42424242)
    configure_save_dir(args.log_dir or os.environ.get("MOSHI_LOG_DIR"))

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        args.moshi_weight,
        args.mimi_weight,
        args.tokenizer,
        lora_weights=args.lora_weight,
        config_path=args.config_path,
    )
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "loading language model")
    lm = checkpoint_info.get_moshi(
        device=args.device, dtype=args.dtype, fuse_lora=args.fuse_lora
    )
    log("info", "language model loaded")

    state = JaServerState(
        checkpoint_info.model_type,
        mimi,
        text_tokenizer,
        lm,
        args.cfg_coef,
        args.device,
        enable_asr=args.enable_asr,
        asr_language=args.asr_language,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        **checkpoint_info.lm_gen_config,
    )
    log("info", "warming up the model")
    state.warmup()

    import tarfile
    from pathlib import Path
    from aiohttp import web
    from huggingface_hub import hf_hub_download
    from kame._tar_utils import extract_data_archive

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    static_path: str | None = None
    if args.static is None:
        log("info", "retrieving static content")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                extract_data_archive(tar, dist_tgz.parent)
        static_path = str(dist)
    elif args.static != "none":
        static_path = args.static

    if static_path is not None:
        import os as _os

        async def handle_root(_):
            return web.FileResponse(_os.path.join(static_path, "index.html"))

        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=False, name="static")

    ssl_context = None
    protocol = "http"
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(
            certfile=os.path.join(args.ssl, "cert.pem"),
            keyfile=os.path.join(args.ssl, "key.pem"),
        )
        protocol = "https"

    log("info", f"J-KAME server ready at {protocol}://{args.host}:{args.port}")
    log("info", f"ASR language: {args.asr_language}")
    log("info", f"LLM endpoint: {args.llm_base_url} (model={args.llm_model})")
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


def cli() -> None:
    with torch.no_grad():
        main()


if __name__ == "__main__":
    cli()
