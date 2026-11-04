"""Variações de um vídeo para teste de criativo: pequenas mudanças quase imperceptíveis
(recorte, cor, velocidade, grão, volume) + limpeza total de metadados.

Ordem certa com o escudo: gere as variações PRIMEIRO e proteja cada uma DEPOIS.
Mudar velocidade ou reprocessar o áudio de um vídeo já protegido pode enfraquecer a proteção.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from . import midia

INTENSIDADES = {
    #            recorte%   saturação     brilho        contraste     matiz°  velocidade      grão  volume dB
    "leve": dict(crop=(0.8, 2.0), sat=(0.96, 1.05), bri=(-0.015, 0.015), con=(0.98, 1.03), hue=1.5, vel=(0.985, 1.015), grao=(0, 2), vol=0.5),
    "media": dict(crop=(1.5, 3.5), sat=(0.93, 1.08), bri=(-0.025, 0.025), con=(0.96, 1.05), hue=3.0, vel=(0.97, 1.03), grao=(1, 4), vol=1.0),
    "forte": dict(crop=(3.0, 6.0), sat=(0.88, 1.12), bri=(-0.04, 0.04), con=(0.93, 1.08), hue=5.0, vel=(0.95, 1.05), grao=(2, 6), vol=1.5),
}


@dataclass
class Parametros:
    nome: str
    recorte_pct: float
    desloc_x: float  # -1..1: de onde sai o recorte (esquerda..direita)
    desloc_y: float
    saturacao: float
    brilho: float
    contraste: float
    matiz: float
    velocidade: float
    grao: int
    volume_db: float
    espelhar: bool


def _info_video(path: Path) -> dict:
    r = subprocess.run([midia.ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True)
    txt = r.stderr.decode("utf-8", "replace")
    m = re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", txt)
    fps = re.search(r"(\d+(?:\.\d+)?) fps", txt)
    hdr = bool(re.search(r"(smpte2084|arib-std-b67|bt2020)", txt))
    rot = re.search(r"rotate\s*:\s*(-?\d+)|displaymatrix: rotation of (-?\d+(?:\.\d+)?)", txt)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (1080, 1920)
    if rot:
        ang = abs(float(rot.group(1) or rot.group(2))) % 180
        if 45 < ang < 135:
            w, h = h, w
    return {"w": w, "h": h, "fps": float(fps.group(1)) if fps else 30.0, "hdr": hdr}


def sortear(n: int, intensidade: str, semente: int, espelhar: bool) -> list[Parametros]:
    faixa = INTENSIDADES[intensidade]
    rng = random.Random(semente)
    out = []
    for i in range(n):
        u = lambda a, b: rng.uniform(a, b)  # noqa: E731
        vel = u(*faixa["vel"])
        if abs(vel - 1) < 0.004:  # garante variação real
            vel = 1 + 0.006 * (1 if rng.random() < 0.5 else -1)
        out.append(Parametros(
            nome=f"v{i + 1:02d}",
            recorte_pct=round(u(*faixa["crop"]), 2),
            desloc_x=round(u(-1, 1), 2),
            desloc_y=round(u(-1, 1), 2),
            saturacao=round(u(*faixa["sat"]), 3),
            brilho=round(u(*faixa["bri"]), 3),
            contraste=round(u(*faixa["con"]), 3),
            matiz=round(u(-faixa["hue"], faixa["hue"]), 2),
            velocidade=round(vel, 4),
            grao=rng.randint(*faixa["grao"]),
            volume_db=round(u(-faixa["vol"], faixa["vol"]), 2),
            espelhar=espelhar and rng.random() < 0.5,
        ))
    return out


def _filtros(p: Parametros, info: dict) -> tuple[str, str]:
    w, h = info["w"], info["h"]
    k = 1 - p.recorte_pct / 100
    cw, ch = int(w * k) // 2 * 2, int(h * k) // 2 * 2
    x = int((w - cw) / 2 * (1 + p.desloc_x))
    y = int((h - ch) / 2 * (1 + p.desloc_y))
    v = []
    if info["hdr"]:
        v.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv")
    v += [
        f"crop={cw}:{ch}:{x}:{y}",
        f"scale={w}:{h}:flags=lanczos",
        f"eq=saturation={p.saturacao}:brightness={p.brilho}:contrast={p.contraste}",
        f"hue=h={p.matiz}",
    ]
    if p.espelhar:
        v.append("hflip")
    if p.grao > 0:
        v.append(f"noise=alls={p.grao}:allf=t")
    v += [f"setpts=PTS/{p.velocidade}", "format=yuv420p"]
    a = [f"atempo={p.velocidade}", f"volume={p.volume_db}dB"]
    return ",".join(v), ",".join(a)


def gerar(entrada: Path, pasta: Path, n: int = 5, intensidade: str = "leve", semente: int = 7,
          espelhar: bool = False, crf: int = 18, log=print) -> list[dict]:
    if intensidade not in INTENSIDADES:
        raise ValueError(f"intensidade: {', '.join(INTENSIDADES)}")
    pasta.mkdir(parents=True, exist_ok=True)
    info = _info_video(entrada)
    tem_audio = midia.info(entrada).tem_audio
    resultados = []
    for p in sortear(n, intensidade, semente, espelhar):
        saida = pasta / f"{entrada.stem}_{p.nome}.mp4"
        fv, fa = _filtros(p, info)
        mapas = ["-map", "0:v:0"] + (["-map", "0:a:0"] if tem_audio else [])
        args = [
            midia.ffmpeg(), "-hide_banner", "-nostdin", "-y", "-i", str(entrada),
            "-filter:v", fv, *mapas,
            # limpeza de metadados: global, por stream, capítulos, data e encoder
            "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1",
            "-fflags", "+bitexact", "-flags:v", "+bitexact",
            "-metadata", "creation_time=", "-metadata", "encoder=",
            "-metadata:s:v", "encoder=", "-metadata:s:v", "handler_name=",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-r", f"{info['fps']:.3f}",
            # tira a mensagem SEI em que o x264 grava versão e configurações dentro do vídeo
            "-bsf:v", "filter_units=remove_types=6",
            "-movflags", "+faststart",
        ]
        if tem_audio:
            args += ["-filter:a", fa, "-flags:a", "+bitexact", "-c:a", "aac", "-b:a", "192k"]
        args.append(str(saida))
        log(f"  {p.nome}: recorte {p.recorte_pct}% · sat {p.saturacao} · vel {p.velocidade}x · grão {p.grao}"
            f"{' · espelhado' if p.espelhar else ''}")
        r = subprocess.run(args, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-1200:])
        resultados.append({"arquivo": saida.name, **asdict(p)})
    (pasta / "variacoes.json").write_text(
        json.dumps({"origem": str(entrada), "intensidade": intensidade, "semente": semente, "variacoes": resultados},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return resultados


def metadados_restantes(path: Path) -> list[str]:
    """Lista o que sobrou de metadado (pra conferência)."""
    r = subprocess.run([midia.ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True)
    txt = r.stderr.decode("utf-8", "replace")
    chaves = []
    for ln in txt.splitlines():
        m = re.match(r"\s{4,}(\w[\w.-]*)\s*:\s*(.+)", ln)
        if m and m.group(1) not in ("Duration", "Stream", "Metadata"):
            chaves.append(f"{m.group(1)}={m.group(2).strip()}")
    return chaves
