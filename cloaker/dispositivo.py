"""Escolha de dispositivo: CUDA (Windows/Linux com NVIDIA), MPS (Mac Apple Silicon) ou CPU."""

from __future__ import annotations

import torch


def _limitar_cuda() -> None:
    """No Windows, passar da VRAM física faz o driver usar a RAM como extensão e
    tudo fica ~100x mais lento. Com um teto, o PyTorch recicla o próprio cache."""
    try:
        livre, total = torch.cuda.mem_get_info()
        # o que outros programas (Windows, navegador, workers) já usam fica de fora
        torch.cuda.set_per_process_memory_fraction(max(0.3, min(0.9, livre * 0.93 / total)))
    except Exception:
        pass


def escolher(pedido: str = "auto") -> torch.device:
    if pedido and pedido != "auto":
        dev = torch.device(pedido)
        if dev.type == "cuda":
            _limitar_cuda()
        return dev
    if torch.cuda.is_available():
        _limitar_cuda()
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_modelo(dev: torch.device) -> torch.dtype:
    # fp16 corta pela metade a memória dos modelos; na CPU fica em fp32.
    return torch.float16 if dev.type in ("cuda", "mps") else torch.float32


def memoria_total_gb(dev: torch.device) -> float:
    if dev.type == "cuda":
        return torch.cuda.get_device_properties(dev).total_memory / 1024**3
    if dev.type == "mps":
        try:
            import psutil  # type: ignore

            return psutil.virtual_memory().total / 1024**3 * 0.6
        except Exception:
            return 10.0
    return 64.0


def limpar(dev: torch.device) -> None:
    import gc

    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def descrever(dev: torch.device) -> str:
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        return f"CUDA · {p.name} · {p.total_memory / 1024**3:.1f} GB"
    if dev.type == "mps":
        return "MPS · Apple Silicon"
    return "CPU (vai funcionar, mas é bem mais lento)"
