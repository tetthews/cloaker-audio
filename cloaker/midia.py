"""Entrada e saída de mídia via ffmpeg (sistema ou o binário embutido do imageio-ffmpeg)."""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

EXT_VIDEO = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
EXT_AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}


@functools.lru_cache(maxsize=1)
def ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ffmpeg não encontrado. No Mac: `brew install ffmpeg`. "
            "No Windows: `winget install Gyan.FFmpeg`. Ou `pip install imageio-ffmpeg`."
        ) from e


def _run(args: list[str]) -> subprocess.CompletedProcess:
    r = subprocess.run([ffmpeg(), "-hide_banner", "-nostdin", *args], capture_output=True)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", "replace")[-1500:]
        raise RuntimeError(f"ffmpeg falhou ({' '.join(args[:6])} ...):\n{msg}")
    return r


@dataclass
class InfoMidia:
    tem_video: bool
    tem_audio: bool
    sr: int
    canais: int
    duracao: float


def info(path: str | Path) -> InfoMidia:
    r = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True)
    txt = r.stderr.decode("utf-8", "replace")
    tem_video = bool(re.search(r"Stream #.*Video:", txt))
    m = re.search(r"Stream #.*Audio:.*?(\d+) Hz, ([^,]+)", txt)
    dur = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", txt)
    duracao = 0.0
    if dur:
        h, mi, s = dur.groups()
        duracao = int(h) * 3600 + int(mi) * 60 + float(s)
    if not m:
        return InfoMidia(tem_video, False, 0, 0, duracao)
    sr = int(m.group(1))
    layout = m.group(2).strip()
    canais = 1 if layout.startswith("mono") else 2 if layout.startswith("stereo") else 2
    ch = re.match(r"(\d+) channels", layout)
    if ch:
        canais = int(ch.group(1))
    return InfoMidia(tem_video, True, sr, canais, duracao)


def carregar_audio(path: str | Path, sr: int | None = None, canais: int | None = None) -> tuple[np.ndarray, int]:
    """Retorna (audio[C, T] float32 em -1..1, sr)."""
    inf = info(path)
    if not inf.tem_audio:
        raise RuntimeError(f"{path} não tem trilha de áudio")
    sr = sr or inf.sr
    canais = canais or inf.canais
    r = _run(["-i", str(path), "-vn", "-ac", str(canais), "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"])
    a = np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, canais).T.copy()
    return a, sr


def salvar_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    sf.write(str(path), np.clip(a, -1, 1).T, sr, subtype="FLOAT")


def salvar_wav16(path: str | Path, audio: np.ndarray, sr: int) -> None:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    sf.write(str(path), np.clip(a, -1, 1).T, sr, subtype="PCM_16")


def montar_saida(entrada: Path, wav_protegido: Path, saida: Path, sr: int) -> None:
    """Junta o áudio protegido ao vídeo original (vídeo copiado sem recompressão)."""
    inf = info(entrada)
    if inf.tem_video:
        _run([
            "-y", "-i", str(entrada), "-i", str(wav_protegido),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "320k", "-ar", str(sr),
            "-map_metadata", "0", "-movflags", "+faststart", "-shortest", str(saida),
        ])
    else:
        ext = saida.suffix.lower()
        if ext == ".wav":
            shutil.copyfile(wav_protegido, saida)
        elif ext in (".m4a", ".aac", ".mp4"):
            _run(["-y", "-i", str(wav_protegido), "-c:a", "aac", "-b:a", "320k", str(saida)])
        elif ext == ".mp3":
            _run(["-y", "-i", str(wav_protegido), "-c:a", "libmp3lame", "-b:a", "320k", str(saida)])
        else:
            _run(["-y", "-i", str(wav_protegido), str(saida)])


@functools.lru_cache(maxsize=1)
def encoders_disponiveis() -> set[str]:
    r = subprocess.run([ffmpeg(), "-hide_banner", "-encoders"], capture_output=True)
    txt = r.stdout.decode("utf-8", "replace")
    achados = set()
    for nome in ("aac", "libmp3lame", "libopus", "libfdk_aac"):
        if re.search(rf"\s{nome}\s", txt):
            achados.add(nome)
    return achados


# (encoder, extensão, bitrate, taxa de saída do codec)
CODECS_SOCIAIS = [
    ("aac", ".m4a", "128k", 44100),   # Instagram/Facebook típico
    ("aac", ".m4a", "96k", 44100),
    ("aac", ".m4a", "64k", 44100),    # download em qualidade baixa
    ("aac", ".m4a", "160k", 48000),
    ("libmp3lame", ".mp3", "128k", 44100),  # ferramentas de download que convertem pra MP3
    ("libopus", ".ogg", "48k", 48000),      # WhatsApp / Telegram
]


def codecs_validos() -> list[tuple[str, str, str, int]]:
    disp = encoders_disponiveis()
    return [c for c in CODECS_SOCIAIS if c[0] in disp]


def _alinhar(ref: np.ndarray, sinal: np.ndarray, max_lag: int = 8192) -> np.ndarray:
    """Corrige o atraso do codificador (priming) por correlação cruzada."""
    n = len(ref)
    if len(sinal) < n:
        sinal = np.pad(sinal, (0, n - len(sinal)))
    seg = min(n, 48000 * 8)
    a = ref[:seg]
    b = sinal[: seg + max_lag]
    nfft = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)
    lag = int(np.argmax(corr[: max_lag + 1]))
    neg = corr[-max_lag:]
    if neg.size and neg.max() > corr[lag]:
        lag = -(max_lag - int(np.argmax(neg)))
    if lag >= 0:
        out = sinal[lag: lag + n]
    else:
        out = np.concatenate([np.zeros(-lag, dtype=sinal.dtype), sinal])[:n]
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out.astype(np.float32)


def ida_e_volta_codec(audio: np.ndarray, sr: int, codec: tuple[str, str, str, int], pasta: str | None = None) -> np.ndarray:
    """Codifica e decodifica (mono) com um codec real; devolve alinhado e no mesmo sr."""
    enc, ext, br, sr_codec = codec
    with tempfile.TemporaryDirectory(dir=pasta) as td:
        src = os.path.join(td, "in.wav")
        cod = os.path.join(td, "cod" + ext)
        sf.write(src, np.clip(audio, -1, 1).astype(np.float32), sr, subtype="FLOAT")
        extra = ["-application", "audio"] if enc == "libopus" else []
        _run(["-y", "-i", src, "-ac", "1", "-ar", str(sr_codec), "-c:a", enc, "-b:a", br, *extra, cod])
        r = _run(["-i", cod, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"])
    dec = np.frombuffer(r.stdout, dtype=np.float32)
    return _alinhar(audio.astype(np.float32), dec)


def simular_rede(path_in: str | Path, path_out: str | Path, bitrate: str = "128k", sr: int = 44100) -> None:
    """Simula o que a rede social entrega pro download: AAC re-encodado."""
    _run(["-y", "-i", str(path_in), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "aac", "-b:a", bitrate, str(path_out)])
