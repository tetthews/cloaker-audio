"""Modelo psicoacústico simplificado (inspirado no MPEG-1 modelo 1 / Qin et al. 2019).

Ideia: um som alto "esconde" sons mais baixos de frequência próxima (mascaramento).
Calculamos, para cada quadro e cada frequência, o nível máximo que o ruído pode ter
sem ser percebido, e penalizamos a perturbação só onde ela passa desse limite.
"""

from __future__ import annotations

import math

import torch

N_FFT = 2048
HOP = 512


def _bark(f: torch.Tensor) -> torch.Tensor:
    return 13 * torch.atan(0.00076 * f) + 3.5 * torch.atan((f / 7500.0) ** 2)


def _limiar_absoluto_db(f: torch.Tensor) -> torch.Tensor:
    """Limiar absoluto de audição (Terhardt), em dB SPL."""
    k = f.clamp(min=20.0) / 1000.0
    ath = 3.64 * k.pow(-0.8) - 6.5 * torch.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k.pow(4)
    return ath.clamp(max=120.0)


def _espectro_db(x: torch.Tensor, janela: torch.Tensor) -> torch.Tensor:
    X = torch.stft(x, N_FFT, HOP, window=janela, return_complex=True, center=True)
    return 10 * torch.log10(X.real.pow(2) + X.imag.pow(2) + 1e-20)  # (F, Tq)


class Mascara:
    """Limiar de mascaramento do áudio original + perda perceptual da perturbação."""

    def __init__(self, x: torch.Tensor, sr: int, offset_db: float):
        dev = x.device
        x = x.float()
        self.janela = torch.hann_window(N_FFT, device=dev)
        p_db = _espectro_db(x, self.janela)
        # normaliza para que o pico do áudio seja 96 dB SPL (convenção do MPEG)
        self.ref = p_db.max().detach()
        p_norm = p_db - self.ref + 96.0
        F = p_norm.shape[0]
        freqs = torch.linspace(0, sr / 2, F, device=dev)
        ath = _limiar_absoluto_db(freqs)[:, None]  # (F,1)

        # maskers tonais: picos locais acima do limiar absoluto
        pico = torch.zeros_like(p_norm, dtype=torch.bool)
        pico[1:-1] = (p_norm[1:-1] > p_norm[:-2]) & (p_norm[1:-1] >= p_norm[2:])
        pico &= p_norm > ath
        # junta a energia dos vizinhos no pico (largura de 3 bins)
        lin = torch.pow(10.0, p_norm / 10.0)
        viz = lin.clone()
        viz[1:-1] = lin[:-2] + lin[1:-1] + lin[2:]
        maskers = torch.where(pico, viz, torch.zeros_like(viz))

        # função de espalhamento de Schroeder no eixo Bark
        z = _bark(freqs)
        dz = z[:, None] - z[None, :]  # (alvo, masker)
        sf_db = 15.81 + 7.5 * (dz + 0.474) - 17.5 * torch.sqrt(1 + (dz + 0.474) ** 2)
        # atenuação adicional do masker (tonal ~ -6 - 0.275 z) + offset de segurança
        atten = -6.025 - 0.275 * z[None, :]
        espalh = torch.pow(10.0, (sf_db + atten - offset_db) / 10.0)  # (F,F)

        mascarado = espalh @ maskers  # (F, Tq)
        teto = 10 * torch.log10(mascarado + torch.pow(10.0, ath / 10.0))
        self.limiar = teto.detach()  # dB "SPL", mesma escala de p_norm
        self.sinal_db = p_norm.detach()
        self.sr = sr

    def perda(self, delta: torch.Tensor) -> torch.Tensor:
        d_db = _espectro_db(delta.float(), self.janela) - self.ref + 96.0
        excesso = torch.relu(d_db - self.limiar)
        return excesso.mean()

    @torch.no_grad()
    def projetar(self, delta: torch.Tensor, margem_db: float = 0.0, fmax: float = 8000.0, iteracoes: int = 2) -> torch.Tensor:
        """Restrição dura: corta, quadro a quadro e frequência a frequência, a parte do ruído
        que passa do limiar de mascaramento (menos `margem_db`). Também zera tudo acima de
        `fmax` (o Whisper só escuta até 8 kHz; acima disso o ruído só incomoda o ouvido)."""
        T = delta.numel()
        teto = torch.pow(10.0, (self.limiar - margem_db + self.ref - 96.0) / 20.0)  # magnitude linear
        F = teto.shape[0]
        corte = int(round(fmax / (self.sr / 2) * (F - 1))) + 1
        d = delta.float()
        for _ in range(iteracoes):
            D = torch.stft(d, N_FFT, HOP, window=self.janela, return_complex=True, center=True)
            fator = torch.clamp(teto[:, : D.shape[1]] / D.abs().clamp(min=1e-12), max=1.0)
            fator[corte:] = 0.0
            d = torch.istft(D * fator, N_FFT, HOP, window=self.janela, center=True, length=T)
        return d.to(delta.dtype)

    @torch.no_grad()
    def estatisticas(self, delta: torch.Tensor) -> dict:
        d_db = _espectro_db(delta.float(), self.janela) - self.ref + 96.0
        exc = d_db - self.limiar
        # só conta faixas audíveis (até 16 kHz)
        F = exc.shape[0]
        fmax = min(F, int(16000 / (self.sr / 2) * F))
        e = exc[:fmax]
        return {
            "fracao_acima_limiar": float((e > 0).float().mean()),
            "excesso_medio_db": float(torch.relu(e).mean()),
            "excesso_p95_db": float(torch.quantile(torch.relu(e).flatten()[:: max(1, e.numel() // 200000)], 0.95)),
        }


@torch.no_grad()
def metricas_qualidade(x: torch.Tensor, delta: torch.Tensor, sr: int) -> dict:
    x = x.float()
    d = delta.float()
    p_x = x.pow(2).mean().clamp(min=1e-12)
    p_d = d.pow(2).mean().clamp(min=1e-12)
    snr = 10 * math.log10(float(p_x / p_d))
    # SNR segmental (janelas de 20 ms, só onde há sinal)
    w = int(sr * 0.02)
    n = (len(x) // w) * w
    xs = x[:n].reshape(-1, w).pow(2).mean(1)
    ds = d[:n].reshape(-1, w).pow(2).mean(1).clamp(min=1e-12)
    ativo = xs > xs.max() * 1e-4
    seg = 10 * torch.log10(xs[ativo].clamp(min=1e-12) / ds[ativo]).clamp(-10, 60)
    pico = float(d.abs().max())
    return {
        "snr_db": round(snr, 2),
        "snr_segmental_db": round(float(seg.mean()) if seg.numel() else snr, 2),
        "pico_ruido_dbfs": round(20 * math.log10(max(pico, 1e-9)), 2),
        "rms_ruido_dbfs": round(10 * math.log10(float(p_d)), 2),
    }
