"""Otimização da perturbação: Adam sobre o ruído, com restrição L-inf e perda psicoacústica
de peso adaptativo (aumenta quando a proteção já funciona, diminui quando falha)."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from . import dispositivo, substitutos
from .psico import Mascara, metricas_qualidade
from .transformacoes import Transformador, semear


@dataclass
class Resultado:
    delta: np.ndarray  # (T,) no sr original
    historico: list[dict] = field(default_factory=list)
    sucesso_final: dict = field(default_factory=dict)
    qualidade: dict = field(default_factory=dict)
    percepcao: dict = field(default_factory=dict)
    segundos: float = 0.0
    passo_escolhido: int = -1


def _orcamento(dev: torch.device, cfg) -> float:
    if cfg.memoria_modelos_gb > 0:
        return cfg.memoria_modelos_gb
    total = dispositivo.memoria_total_gb(dev)
    if dev.type == "cuda":
        return max(1.0, total * 0.5)
    return total * 0.7


def carregar_substitutos(cfg, dev: torch.device, log=print) -> substitutos.Piscina:
    dtype = dispositivo.dtype_modelo(dev)
    modelos = []
    for nome in cfg.modelos:
        log(f"  carregando substituto {nome} ...")
        try:
            modelos.append(substitutos.criar(nome, cfg, dtype))
        except Exception as e:
            log(f"  [aviso] não deu pra carregar {nome}: {e}")
    if not modelos:
        raise RuntimeError("nenhum modelo substituto carregou")
    return substitutos.Piscina(modelos, dev, _orcamento(dev, cfg))


def proteger(x: np.ndarray, sr: int, cfg, piscina: substitutos.Piscina, log=print, pasta_tmp: str | None = None) -> Resultado:
    """x: (C, T) float32. Devolve a perturbação mono a somar em todos os canais."""
    t0 = time.time()
    semear(cfg.semente)
    dev = piscina.dev
    xm = torch.from_numpy(x.mean(0)).to(dev).float()
    T = xm.numel()

    mascara = Mascara(xm, sr, cfg.offset_db)
    trans = Transformador(xm, sr, cfg, pasta_tmp)

    segs = None
    mascara_tempo = None
    if cfg.modo == "roteiro":
        import json as _json

        segs = _json.loads(open(cfg.roteiro, encoding="utf-8").read())["trechos"]
        faltando = [i for i, t in enumerate(segs) if not t.get("falso", "").strip()]
        if faltando:
            raise RuntimeError(f"roteiro sem texto falso nos trechos {faltando[:10]}")
        mascara_tempo = None
        if cfg.foco:
            from .roteiro_auto import priorizar

            segs, rf = priorizar(segs)
            log(f"  foco: {rf['ativos']} trechos precisam de ruído, {rf['pulados']} pulados (fala já parecida com o texto falso) "
                f"· {rf['fracao_tempo_ativa']:.0%} do tempo de fala")
            if rf["pulados"]:
                # ruído só nos trechos ativos (com folga de 0,3 s e rampa de 20 ms pra não estalar)
                mt = torch.zeros(T, device=dev)
                for t in segs:
                    if t["ativo"]:
                        a = max(0, int((t["ini"] - 0.3) * sr)); b = min(T, int((t["fim"] + 0.3) * sr))
                        mt[a:b] = 1.0
                k = max(1, int(0.02 * sr))
                rampa = torch.ones(1, 1, 2 * k + 1, device=dev) / (2 * k + 1)
                mascara_tempo = torch.nn.functional.conv1d(mt[None, None], rampa, padding=k)[0, 0].clamp(0, 1)
        fora = [m.nome for m in piscina.modelos if not hasattr(m, "perda_roteiro")]
        if fora:
            log(f"  [aviso] modo roteiro só usa Whisper; ignorando {fora}")
            piscina.modelos[:] = [m for m in piscina.modelos if hasattr(m, "perda_roteiro")]

    if cfg.modo == "embaralhar":
        # referência: o que o maior Whisper do conjunto entende do original
        from .avaliacao import palavras_conteudo

        whispers = [m for m in piscina.modelos if hasattr(m, "transcrever")]
        if whispers:
            ref_m = max(whispers, key=lambda m: m.tamanho_gb)
            piscina.garantir(ref_m)
            with torch.no_grad():
                a16 = trans._para16(xm, via44=False).float().cpu().numpy()
                texto = ref_m.transcrever(a16)
            pal = set(palavras_conteudo(texto))
            log(f"  referência ({ref_m.nome}): {texto[:140]}{'...' if len(texto) > 140 else ''}")
            for m in whispers:
                m.ref_palavras = pal
            # só o Whisper de decodificador mais leve transcreve a cada passo; os outros
            # são empurrados pra longe dessa mesma leitura (mesmo vocabulário)
            comp: dict = {}
            dec = min(whispers, key=lambda m: m.m.dims.n_text_layer)
            for m in whispers:
                m.compartilhado = comp
                m.decodifica = m is dec
            piscina.modelos.sort(key=lambda m: 0 if m is dec else 1)
            log(f"  decodificador do embaralhar: {dec.nome}")
    eps = cfg.eps

    delta = (torch.rand(T, device=dev) * 2 - 1) * eps * 0.05
    delta.requires_grad_(True)
    lr0 = eps * cfg.lr_rel
    opt = torch.optim.Adam([delta], lr=lr0)
    alpha = cfg.alpha0

    modelos = piscina.modelos
    bloco = 1 if piscina.cabem_todos else 12
    ordem: list[int] = []
    rng = random.Random(cfg.semente)
    while len(ordem) < cfg.passos:
        idx = list(range(len(modelos)))
        rng.shuffle(idx)
        for i in idx:
            ordem.extend([i] * bloco)
    ordem = ordem[: cfg.passos]

    ema = {m.nome: 0.0 for m in modelos}
    vistos = {m.nome: 0 for m in modelos}
    hist: list[dict] = []
    melhor = None  # (perda_perceptual, passo, delta)

    # soma: todos os modelos em todo passo (melhor para conjunto); só se couberem juntos na GPU
    somar = piscina.cabem_todos and cfg.ensemble == "soma"
    if somar:
        for m in modelos:
            piscina.garantir(m)

    barra = tqdm(range(cfg.passos), desc="protegendo", unit="passo", dynamic_ncols=True)
    for passo in barra:
        ativos = modelos if somar else [modelos[ordem[passo]]]
        if not somar:
            piscina.garantir(ativos[0])
        # taxa de aprendizado com cosseno
        lr = lr0 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * passo / max(1, cfg.passos))))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        s = xm + delta
        if segs is not None:
            lote, comps, alvos = trans.amostrar_roteiro(s, segs, cfg.lote, passo)
        else:
            lote, comps = trans.amostrar(s, cfg.lote, passo)
        # cada modelo faz o backward até o lote (economiza memória), depois o lote até o ruído
        folha = lote.detach().requires_grad_(True)
        perdas_asr = {}
        for m in ativos:
            if segs is not None:
                out = m.perda_roteiro(folha, alvos, kappa=cfg.kappa)
            else:
                out = m.perda(folha, comps, kappa=cfg.kappa, peso_ns=cfg.peso_nospeech)
            (out.perda / len(ativos)).backward()
            perdas_asr[m.nome] = out.valor if out.valor is not None else float(out.perda.detach())
            taxa = float(out.sucesso.float().mean())
            vistos[m.nome] += 1
            k = 0.15 if vistos[m.nome] > 5 else 0.5
            ema[m.nome] = (1 - k) * ema[m.nome] + k * taxa
            del out
        lote.backward(folha.grad)
        l_perc = mascara.perda(delta)
        (alpha * l_perc).backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
            if cfg.margem_db is not None:
                # teto duro: começa folgado (acha a direção) e aperta até a margem final em 60% dos passos
                f = min(1.0, passo / max(1, cfg.passos * 0.6))
                marg = cfg.margem_db - cfg.folga_inicial_db * (1 - f)
                delta.copy_(mascara.projetar(delta, margem_db=marg, iteracoes=1))
            if mascara_tempo is not None:
                delta.mul_(mascara_tempo)
            delta.copy_((xm + delta).clamp(-1, 1) - xm)
        m = ativos[0]

        # ajuste do peso perceptual a cada 10 passos
        if passo % 10 == 9 and all(v > 3 for v in vistos.values()):
            pior = min(ema.values())
            if pior >= 0.9:
                alpha = min(alpha * 1.25, 50.0)
            elif pior < 0.6:
                alpha = max(alpha * 0.8, 1e-4)

        # guarda o melhor: todos os modelos protegidos e menor ruído perceptual
        if passo > cfg.passos * (0.6 if cfg.margem_db is not None else 0.3) and all(v > 3 for v in vistos.values()) and min(ema.values()) >= 0.9:
            lp = float(l_perc.detach())
            if melhor is None or lp < melhor[0]:
                melhor = (lp, passo, delta.detach().clone())

        if passo % 5 == 0 or passo == cfg.passos - 1:
            reg = {
                "passo": passo,
                "modelo": "+".join(perdas_asr),
                "perda_asr": {k2: round(v, 4) for k2, v in perdas_asr.items()},
                "perda_perc": round(float(l_perc.detach()), 4),
                "alpha": round(alpha, 5),
                "sucesso": {k2: round(v, 3) for k2, v in ema.items()},
                "codec": trans.codec_atual[0] + "@" + trans.codec_atual[2] if trans.codec_atual else None,
            }
            hist.append(reg)
            barra.set_postfix_str(
                " ".join(f"{k2[:6]}={v:.2f}" for k2, v in ema.items()) + f" perc={float(l_perc.detach()):.2f} a={alpha:.3f}"
            )

    final = melhor[2] if melhor is not None else delta.detach()
    if cfg.margem_db is not None:
        final = mascara.projetar(final, margem_db=cfg.margem_db, iteracoes=2)
    if mascara_tempo is not None:
        final = final * mascara_tempo
    escolhido = melhor[1] if melhor is not None else cfg.passos - 1
    res = Resultado(
        delta=final.float().cpu().numpy(),
        historico=hist,
        sucesso_final={k2: round(v, 3) for k2, v in ema.items()},
        qualidade=metricas_qualidade(xm, final, sr),
        percepcao=mascara.estatisticas(final),
        segundos=round(time.time() - t0, 1),
        passo_escolhido=escolhido,
    )
    if melhor is None:
        if cfg.margem_db is not None:
            # com teto de ruído o critério interno (90% dos tokens no alvo, em todo recorte) quase nunca
            # fecha, mas a proteção real após o upload sim: quem decide é a avaliação final
            log("  usando o último passo (normal com teto de ruído; a avaliação final mede a proteção real)")
        else:
            log("  [aviso] o escudo não atingiu 90% de sucesso em todos os modelos; usando o último passo. "
                "Tente --forca forte ou mais --passos.")
    return res
