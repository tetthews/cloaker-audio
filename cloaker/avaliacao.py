"""Prova: transcreve o original e o protegido (também depois de simular o upload)
com vários transcritores, inclusive modelos que o escudo nunca viu, e mede quanto
do que foi falado ainda dá pra recuperar."""

from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import midia

STOP = set(
    "a o e é de da do das dos em no na nos nas um uma uns umas que pra para por com sem se "
    "eu tu ele ela nós vós eles elas você vocês me te lhe nos ao aos à às mais menos muito "
    "the and of to in is it this that".split()
)


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def palavras_conteudo(t: str) -> list[str]:
    return [w for w in normalizar(t).split() if len(w) >= 3 and w not in STOP]


def recuperacao(ref: str, hip: str) -> float:
    """Fração das palavras de conteúdo da fala original que aparecem na transcrição."""
    r = Counter(palavras_conteudo(ref))
    if not r:
        return 0.0
    h = Counter(palavras_conteudo(hip))
    acertos = sum(min(c, h.get(w, 0)) for w, c in r.items())
    return acertos / sum(r.values())


def recuperacao_exclusiva(ref: str, outro: str, hip: str) -> float:
    """Como recuperacao(), mas só com as palavras de `ref` que NÃO estão em `outro`.
    No modo roteiro o texto falso repete de propósito palavras do original; sem isso
    a métrica conta essas palavras comuns como vazamento."""
    o = set(palavras_conteudo(outro))
    r = Counter(w for w in palavras_conteudo(ref) if w not in o)
    if not r:
        return 0.0
    h = Counter(palavras_conteudo(hip))
    return sum(min(c, h.get(w, 0)) for w, c in r.items()) / sum(r.values())


def wer(ref: str, hip: str) -> float:
    import jiwer

    r, h = normalizar(ref), normalizar(hip)
    if not r:
        return 0.0
    if not h:
        return 1.0
    return float(min(1.0, jiwer.wer(r, h)))


def ler_referencia(path: str | Path) -> str:
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".ass":
        linhas = []
        for ln in txt.splitlines():
            if ln.startswith("Dialogue:"):
                partes = ln.split(",", 9)
                if len(partes) == 10:
                    linhas.append(re.sub(r"\{[^}]*\}", "", partes[9]).replace("\\N", " "))
        return " ".join(linhas)
    if p.suffix.lower() in (".srt", ".vtt"):
        out = []
        for ln in txt.splitlines():
            s = ln.strip()
            if not s or s.isdigit() or "-->" in s or s.upper().startswith("WEBVTT"):
                continue
            out.append(re.sub(r"<[^>]+>", "", s))
        return " ".join(out)
    return txt


# ------------------------------------------------------------------ transcritores


@dataclass
class Transcritor:
    nome: str
    rotulo: str
    visto_no_treino: bool
    fn: object = None  # callable(audio16: np.ndarray) -> str
    liberar: object = None


def _faster(nome: str, idioma: str | None, vad: bool, dev: str):
    import torch  # noqa: F401  (carrega as DLLs do CUDA antes do ctranslate2 no Windows)
    from faster_whisper import WhisperModel

    try:
        if dev == "cuda":
            m = WhisperModel(nome, device="cuda", compute_type="float16")
        else:
            m = WhisperModel(nome, device="cpu", compute_type="int8")
    except Exception:
        m = WhisperModel(nome, device="cpu", compute_type="int8")

    def fn(a16: np.ndarray, idioma=idioma, vad=vad) -> str:
        segs, _ = m.transcribe(a16, language=idioma, vad_filter=vad, beam_size=5)
        return " ".join(s.text.strip() for s in segs).strip()

    return m, fn


def _w2v(nome: str, dev: str):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    from .substitutos import ALIAS_W2V

    repo = ALIAS_W2V.get(nome, nome)
    proc = Wav2Vec2Processor.from_pretrained(repo)
    m = Wav2Vec2ForCTC.from_pretrained(repo).to(dev)
    m.train(False)

    @torch.no_grad()
    def fn(a16: np.ndarray) -> str:
        textos = []
        passo = 16000 * 20
        for i in range(0, len(a16), passo):
            pedaco = a16[i: i + passo]
            if len(pedaco) < 1600:
                continue
            ent = proc(pedaco, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
            ids = m(ent).logits.argmax(-1)
            textos.append(proc.batch_decode(ids)[0])
        return " ".join(textos).lower()

    return m, fn


def montar_transcritores(nomes: list[str], vistos: list[str], dev: str) -> list[tuple[str, bool, callable]]:
    """Devolve fábricas preguiçosas: carrega um modelo por vez para caber na GPU."""
    alias_treino = {"turbo": "large-v3-turbo"}
    vistos_norm = {alias_treino.get(v, v) for v in vistos}
    fabricas = []
    for nome in nomes:
        visto = nome in vistos_norm
        fabricas.append((nome, visto))
    return fabricas


@dataclass
class Relatorio:
    referencia: str
    fonte_referencia: str
    linhas: list[dict] = field(default_factory=list)
    segundos: float = 0.0

    def resumo(self) -> dict:
        def media(chave, filtro=lambda l: True):
            v = [l[chave] for l in self.linhas if filtro(l) and l.get(chave) is not None]
            return round(float(np.mean(v)), 3) if v else None

        return {
            "recuperacao_original": media("rec_original"),
            "recuperacao_protegido": media("rec_protegido"),
            "recuperacao_rede_128k": media("rec_rede_128k"),
            "recuperacao_rede_64k": media("rec_rede_64k"),
            "recuperacao_rede_128k_nunca_vistos": media("rec_rede_128k", lambda l: not l["visto_no_treino"]),
            "texto_falso_rede_128k": media("falso_rede_128k"),
            "texto_falso_rede_128k_nunca_vistos": media("falso_rede_128k", lambda l: not l["visto_no_treino"]),
            "pior_caso_rede_128k": max((l["rec_rede_128k"] for l in self.linhas if l.get("rec_rede_128k") is not None), default=None),
        }


def avaliar(
    arquivos: dict[str, Path],
    avaliadores: list[str],
    vistos: list[str],
    dev: str,
    referencia: str | None = None,
    fonte_ref: str = "",
    falso: str | None = None,
    log=print,
    leve: bool = False,
) -> Relatorio:
    """arquivos: rótulo -> caminho (original, protegido, rede_128k, rede_64k)."""
    t0 = time.time()
    audios = {k: midia.carregar_audio(p, sr=16000, canais=1)[0][0] for k, p in arquivos.items()}
    fabricas = montar_transcritores(avaliadores, vistos, dev)
    textos: dict[tuple[str, str], dict[str, str]] = {}
    meta: dict[tuple[str, str], bool] = {}

    for nome, visto in fabricas:
        log(f"  transcrevendo com {nome}{' (visto no treino)' if visto else ' (NUNCA visto pelo escudo)'} ...")
        try:
            if nome.startswith("w2v"):
                m, fn = _w2v(nome, dev)
                configs = [("ctc", fn)]
            else:
                m, fn = _faster(nome, None, False, dev)
                configs = [
                    ("pt+vad", lambda a, fn=fn: fn(a, idioma="pt", vad=True)),
                    ("auto", lambda a, fn=fn: fn(a, idioma=None, vad=False)),
                ][: 1 if leve else 2]
        except Exception as e:
            log(f"  [aviso] {nome} indisponível: {e}")
            continue
        for rot, f in configs:
            chave = (nome, rot)
            textos[chave] = {}
            meta[chave] = visto
            for k, a in audios.items():
                try:
                    textos[chave][k] = f(a)
                except Exception as e:
                    textos[chave][k] = f"[erro: {e}]"
        del m
        try:
            import torch

            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    if not referencia:
        # a melhor referência disponível: maior modelo, pt+vad, no original
        prefer = ["large-v3", "large-v3-turbo", "medium", "small"]
        for p in prefer:
            if (p, "pt+vad") in textos:
                referencia = textos[(p, "pt+vad")].get("original", "")
                fonte_ref = f"transcrição do original por {p}"
                break
        referencia = referencia or next(iter(textos.values()), {}).get("original", "")

    rel = Relatorio(referencia=referencia or "", fonte_referencia=fonte_ref)
    for (nome, rot), tx in textos.items():
        linha = {"transcritor": nome, "config": rot, "visto_no_treino": meta[(nome, rot)], "textos": tx}
        for k, t in tx.items():
            if falso:
                linha[f"rec_{k}"] = round(recuperacao_exclusiva(rel.referencia, falso, t), 3)
            else:
                linha[f"rec_{k}"] = round(recuperacao(rel.referencia, t), 3)
            linha[f"wer_{k}"] = round(wer(rel.referencia, t), 3)
            if falso:
                linha[f"falso_{k}"] = round(recuperacao_exclusiva(falso, rel.referencia, t), 3)
        rel.linhas.append(linha)
    rel.segundos = round(time.time() - t0, 1)
    return rel


def por_trecho(audio16: np.ndarray, trechos: list[dict], modelo: str, dev: str) -> list[dict]:
    """Vazamento trecho a trecho: transcreve com tempo por palavra e mede, em cada trecho,
    quanto das palavras exclusivas da fala original aparece (e quais)."""
    import torch  # noqa: F401
    from faster_whisper import WhisperModel

    try:
        m = WhisperModel(modelo, device=dev, compute_type="float16" if dev == "cuda" else "int8")
    except Exception:
        m = WhisperModel(modelo, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(audio16, language="pt", vad_filter=True, beam_size=5, word_timestamps=True)
    pals = [(w.start, w.end, w.word) for s in segs for w in (s.words or [])]
    del m
    out = []
    for t in trechos:
        dentro = " ".join(w for a, b, w in pals if t["ini"] - 0.3 <= (a + b) / 2 <= t["fim"] + 0.3).strip()
        excl = [w for w in dict.fromkeys(palavras_conteudo(t["original"])) if w not in set(palavras_conteudo(t["falso"]))]
        ouvidas = set(palavras_conteudo(dentro))
        out.append({
            "ini": t["ini"], "fim": t["fim"], "original": t["original"], "falso": t["falso"],
            "protegido": t.get("proteger", True) is not False,
            "transcrito": dentro,
            "vazamento": round(sum(w in ouvidas for w in excl) / len(excl), 3) if excl else 0.0,
            "vazou": [w for w in excl if w in ouvidas],
        })
    return out
