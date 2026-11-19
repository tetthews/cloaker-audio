"""Paridade torch x MLX: mesma perda, mesmo sucesso e gradiente parecido em relação ao áudio.

Uso (da raiz do projeto):
    python tests/paridade_mlx.py --modelos base,small --dispositivo mps   # torch fp16 no MPS x MLX fp16 na GPU
    python tests/paridade_mlx.py --modelos base --dispositivo cpu          # fp32 nos dois, em CPU

Áudio: fala sintética (comando `say` do macOS, se existir) ou ruído modulado como fala.
Meta: cosseno do gradiente > 0,95 e perda dentro de 2%.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloaker import config as cfgmod  # noqa: E402
from cloaker.substitutos import N_AMOSTRAS, WhisperSubstituto  # noqa: E402


def audio_teste() -> np.ndarray:
    """~8 s de fala sintética a 16 kHz (mono), dentro de uma janela de 30 s com zeros no fim."""
    texto = "Olá pessoal, hoje eu vou mostrar como funciona o nosso método para vender na internet, passo a passo."
    x = None
    try:
        with tempfile.TemporaryDirectory() as d:
            arq = os.path.join(d, "f.wav")
            subprocess.run(["say", "-v", "Luciana", "-o", arq, "--data-format=LEI16@16000", texto], check=True,
                           capture_output=True, timeout=60)
            import scipy.io.wavfile as w

            sr, a = w.read(arq)
            x = a.astype(np.float32) / 32768.0
    except Exception:
        rng = np.random.default_rng(0)
        t = np.arange(16000 * 8) / 16000
        env = np.clip(np.sin(2 * np.pi * 3 * t), 0, 1)
        x = (0.1 * env * rng.standard_normal(t.size)).astype(np.float32)
    x = x[: 12 * 16000]
    lote = np.zeros(N_AMOSTRAS, dtype=np.float32)
    lote[: len(x)] = x
    return lote, len(x) / 16000


def roteiro(dur: float) -> list[dict]:
    return [
        {"ini": 0.2, "fim": dur / 2, "original": "olá pessoal hoje eu vou mostrar como funciona",
         "falso": "bom dia, vamos falar de receita de bolo de cenoura"},
        {"ini": dur / 2, "fim": dur, "original": "o nosso método para vender na internet passo a passo",
         "falso": "a previsão do tempo para amanhã é de chuva forte"},
    ]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def rodar(m, x: torch.Tensor, chamada) -> tuple[float, np.ndarray, np.ndarray, float]:
    folha = x.clone().detach().requires_grad_(True)
    t0 = time.time()
    out = chamada(m, folha)
    out.perda.backward()
    if folha.device.type == "mps":
        torch.mps.synchronize()
    seg = time.time() - t0
    val = out.valor if out.valor is not None else float(out.perda.detach())
    return val, out.sucesso.float().cpu().numpy(), folha.grad.float().cpu().numpy(), seg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", default="base")
    ap.add_argument("--dispositivo", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--modo", default="roteiro", choices=["roteiro", "trocar", "silenciar"])
    ap.add_argument("--lote", type=int, default=1)
    ap.add_argument("--ref-fp32", action="store_true",
                    help="calcula também o gradiente do torch em fp32 na CPU e mostra o cosseno de cada motor contra ele")
    a = ap.parse_args()

    dev = torch.device(a.dispositivo)
    dtype = torch.float16 if dev.type == "mps" else torch.float32
    if dev.type == "cpu":
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
    from cloaker.substitutos_mlx import MLXWhisperSubstituto

    x0, dur = audio_teste()
    x = torch.from_numpy(np.stack([x0] * a.lote)).to(dev)
    alvos = [roteiro(dur)] * a.lote
    comps = [int(dur * 16000)] * a.lote
    cfg = cfgmod.Config() if hasattr(cfgmod, "Config") else None
    cfg.modo = a.modo
    cfg.idiomas = ["pt"]

    if a.modo == "roteiro":
        chamada = lambda m, f: m.perda_roteiro(f, alvos, kappa=cfg.kappa)  # noqa: E731
    else:
        chamada = lambda m, f: m.perda(f, comps, kappa=cfg.kappa)  # noqa: E731

    falhou = False
    print(f"dispositivo={a.dispositivo} dtype={dtype} modo={a.modo} lote={a.lote}")
    print(f"{'modelo':8} {'perda torch':>12} {'perda mlx':>12} {'dif rel':>9} {'sucesso t/m':>12} {'cosseno':>9} {'s torch':>8} {'s mlx':>7}")
    for nome in a.modelos.split(","):
        mt = WhisperSubstituto(nome, cfg, dtype)
        mt.para(dev)
        mm = MLXWhisperSubstituto(nome, cfg, dtype)
        # aquece o MLX (compilação dos kernels) e o torch antes de medir
        rodar(mm, x, chamada)
        vt, st, gt, tt = rodar(mt, x, chamada)
        vm, sm, gm, tm = rodar(mm, x, chamada)
        dif = abs(vm - vt) / max(abs(vt), 1e-9)
        c = cos(gt, gm)
        ok = dif < 0.02 and c > 0.95
        falhou |= not ok
        print(f"{nome:8} {vt:12.5f} {vm:12.5f} {dif:9.2%} {st.mean():5.2f}/{sm.mean():5.2f} {c:9.4f} {tt:8.2f} {tm:7.2f}  {'OK' if ok else 'FALHOU'}")
        if a.ref_fp32:
            mr = WhisperSubstituto(nome, cfg, torch.float32)
            _, _, gr, _ = rodar(mr, x.cpu(), chamada)
            print(f"         contra torch fp32 CPU: cosseno torch={cos(gt, gr):.4f}  mlx={cos(gm, gr):.4f}")
            del mr
        del mt, mm
    return 1 if falhou else 0


if __name__ == "__main__":
    raise SystemExit(main())
