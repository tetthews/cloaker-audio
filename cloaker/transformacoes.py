"""Expectativa sobre transformações (EOT): o ruído é treinado já passando pelo que
acontece com o áudio no mundo real, para não morrer no upload.

- codec real (AAC/MP3/Opus via ffmpeg) com gradiente "atravessando" (BPDA)
- reamostragem 48k -> 44.1k -> 16k
- ganho aleatório (normalização de volume da plataforma)
- remoção de silêncios (o que o VAD do faster-whisper faz)
- recortes de tamanho e posição aleatórios (janelas de 30 s em qualquer ponto)
- ruído de fundo leve
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torchaudio.functional as AF

from . import midia
from .substitutos import N_AMOSTRAS, SR_ASR


class Transformador:
    def __init__(self, x: torch.Tensor, sr: int, cfg, pasta_tmp: str | None = None):
        self.sr = sr
        self.cfg = cfg
        self.dev = x.device
        self.pasta_tmp = pasta_tmp
        self.codecs = midia.codecs_validos()
        self.residuo: torch.Tensor | None = None
        self.codec_atual = None
        self.rng = random.Random(cfg.semente)
        # VAD aproximado a partir do áudio limpo (quadros de 30 ms em 16 kHz)
        with torch.no_grad():
            x16 = self._para16(x, via44=False)
            q = 480
            n = (len(x16) // q) * q
            en = x16[:n].reshape(-1, q).pow(2).mean(1)
            en_db = 10 * torch.log10(en + 1e-10)
            self.fala_quadros = en_db > (en_db.max() - 35)
            self.q = q
        self.n16 = len(x16)

    # ------------------------------------------------------------------ util
    def _para16(self, s: torch.Tensor, via44: bool) -> torch.Tensor:
        if via44 and self.sr != 44100:
            s = AF.resample(s, self.sr, 44100)
            return AF.resample(s, 44100, SR_ASR)
        if self.sr == SR_ASR:
            return s
        return AF.resample(s, self.sr, SR_ASR)

    def atualizar_codec(self, s: torch.Tensor) -> None:
        """Roda o codec de verdade no sinal atual e guarda o resíduo (codec(s) - s)."""
        if not self.codecs:
            self.residuo = None
            return
        self.codec_atual = self.rng.choice(self.codecs)
        a = s.detach().float().cpu().numpy()
        dec = midia.ida_e_volta_codec(a, self.sr, self.codec_atual, self.pasta_tmp)
        self.residuo = torch.from_numpy(dec - a).to(self.dev)

    # ------------------------------------------------------------------ amostragem
    def amostrar(self, s: torch.Tensor, B: int, passo: int) -> tuple[torch.Tensor, list[int]]:
        c = self.cfg
        if c.prob_codec > 0 and (self.residuo is None or passo % c.codec_a_cada == 0):
            self.atualizar_codec(s)
        if self.residuo is not None and self.rng.random() < c.prob_codec:
            s = s + self.residuo  # BPDA: o gradiente passa como identidade
        s16 = self._para16(s, via44=self.rng.random() < c.prob_44k)

        # remoção de silêncio (VAD)
        if self.rng.random() < c.prob_vad:
            n = len(self.fala_quadros) * self.q
            quadros = s16[:n].reshape(-1, self.q)
            mant = self.fala_quadros.clone()
            # mantém um pouco de silêncio (o VAD real deixa margens)
            mant |= torch.rand_like(mant, dtype=torch.float32) < 0.15
            if mant.sum() > 50:
                s16 = quadros[mant].reshape(-1)

        lotes, comps = [], []
        T = len(s16)
        for _ in range(B):
            r = self.rng.random()
            if r < 0.35:
                L = min(T, N_AMOSTRAS)
            else:
                L = min(T, int(self.rng.uniform(2.5, 30.0) * SR_ASR))
            ini = self.rng.randint(0, max(0, T - L))
            # metade das vezes alinha no começo (é onde a 1ª janela do Whisper sempre cai)
            if self.rng.random() < 0.25:
                ini = 0
            seg = s16[ini: ini + L]
            ganho = 10 ** (self.rng.uniform(*c.ganho_db) / 20)
            seg = seg * ganho
            if self.rng.random() < c.prob_ruido:
                p = seg.pow(2).mean().detach().clamp(min=1e-10)
                snr = self.rng.uniform(30, 50)
                seg = seg + torch.randn_like(seg) * torch.sqrt(p / 10 ** (snr / 10))
            seg = seg.clamp(-1, 1)
            lotes.append(torch.nn.functional.pad(seg, (0, N_AMOSTRAS - len(seg))))
            comps.append(len(seg))
        return torch.stack(lotes), comps

    def amostrar_roteiro(self, s: torch.Tensor, segs: list[dict], B: int, passo: int):
        """Recortes que começam e terminam em fronteiras de trecho (é onde o Whisper
        posiciona as janelas), devolvendo também os trechos com tempo relativo ao recorte.
        Sem simulação de VAD: aqui o tempo precisa bater."""
        c = self.cfg
        if c.prob_codec > 0 and (self.residuo is None or passo % c.codec_a_cada == 0):
            self.atualizar_codec(s)
        if self.residuo is not None and self.rng.random() < c.prob_codec:
            s = s + self.residuo
        s16 = self._para16(s, via44=self.rng.random() < c.prob_44k)
        T = len(s16)
        lotes, comps, alvos = [], [], []
        for _ in range(B):
            pesos = [t.get("peso", 1.0) for t in segs]
            if sum(pesos) <= 0:
                pesos = [1.0] * len(segs)
            # recorte começando um pouco antes do trecho sorteado (pelo peso), pra ele não ficar sempre na borda
            alvo_i = self.rng.choices(range(len(segs)), weights=pesos)[0]
            i0 = alvo_i if self.rng.random() < 0.6 else max(0, alvo_i - self.rng.randint(1, 2))
            fins = [j for j in range(i0, len(segs)) if segs[j]["fim"] - segs[i0]["ini"] <= 29.0]
            j = self.rng.choice(fins) if self.rng.random() < 0.5 else fins[-1]
            t_ini = 0.0 if i0 == 0 and self.rng.random() < 0.5 else max(0.0, segs[i0]["ini"] - self.rng.uniform(0, 0.25))
            t_fim = min(T / SR_ASR, segs[j]["fim"] + self.rng.uniform(0.05, 0.4), t_ini + 30.0)
            a, b = int(t_ini * SR_ASR), int(t_fim * SR_ASR)
            seg = s16[a:b] * 10 ** (self.rng.uniform(*c.ganho_db) / 20)
            if self.rng.random() < c.prob_ruido:
                p = seg.pow(2).mean().detach().clamp(min=1e-10)
                seg = seg + torch.randn_like(seg) * torch.sqrt(p / 10 ** (self.rng.uniform(30, 50) / 10))
            seg = seg.clamp(-1, 1)
            lotes.append(torch.nn.functional.pad(seg, (0, N_AMOSTRAS - len(seg))))
            comps.append(len(seg))
            alvos.append([
                {"ini": max(0.0, segs[k]["ini"] - t_ini), "fim": min(30.0, segs[k]["fim"] - t_ini),
                 "falso": segs[k]["falso"], "original": segs[k]["original"]}
                for k in range(i0, j + 1)
            ])
        return torch.stack(lotes), comps, alvos


def semear(n: int) -> None:
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
