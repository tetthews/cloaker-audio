"""Relatório HTML autocontido (abre direto no navegador, com players de áudio)."""

from __future__ import annotations

import html
import json
from pathlib import Path


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def _cor(v) -> str:
    if v is None:
        return ""
    if v <= 0.10:
        return "bom"
    if v <= 0.35:
        return "medio"
    return "ruim"


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--tx:#1b1b1b;--mut:#6b6b6b;--ln:#e4e4e0;--bom:#1f8a4c;--medio:#b7791f;--ruim:#c0392b;--ac:#2d5bd7}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--card:#1e1e1e;--tx:#ececec;--mut:#9a9a9a;--ln:#333;--bom:#4cc27a;--medio:#e0a84a;--ruim:#ef6b5b;--ac:#7c9cff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
.mut{color:var(--mut)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--ln);border-radius:10px;padding:14px}.big{font-size:28px;font-weight:650}
.bom{color:var(--bom)}.medio{color:var(--medio)}.ruim{color:var(--ruim)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--ln);border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:8px 10px;border-bottom:1px solid var(--ln);text-align:left;vertical-align:top}th{font-weight:600;background:color-mix(in srgb,var(--card) 92%,var(--tx))}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;border:1px solid var(--ln);color:var(--mut)}
.tag.novo{border-color:var(--ac);color:var(--ac)}
details{background:var(--card);border:1px solid var(--ln);border-radius:10px;padding:10px 14px;margin:8px 0}summary{cursor:pointer;font-weight:600}
.tx{font-size:13.5px;white-space:pre-wrap;margin:6px 0 12px}.audio{display:grid;gap:10px}audio{width:100%}
.scroll{overflow-x:auto}
"""


def gerar(pasta: Path, dados: dict) -> Path:
    r = dados.get("avaliacao") or {}
    res = r.get("resumo", {})
    q = dados.get("qualidade", {})
    pc = dados.get("percepcao", {})
    cfg = dados.get("config", {})
    E = html.escape

    cards = [
        ("Fala recuperável no original", res.get("recuperacao_original"), "quanto a IA pega do vídeo sem proteção"),
        ("Recuperável após upload (128k)", res.get("recuperacao_rede_128k"), "média de todos os transcritores"),
        ("Só modelos nunca vistos", res.get("recuperacao_rede_128k_nunca_vistos"), "teste honesto de transferência"),
        ("Pior caso após upload", res.get("pior_caso_rede_128k"), "o transcritor que mais conseguiu"),
    ]
    if res.get("texto_falso_rede_128k") is not None:
        cards.insert(1, ("Texto falso que a IA escreveu", res.get("texto_falso_rede_128k"),
                         "fração das palavras do roteiro falso na transcrição (maior = melhor)"))
    cards_html = "".join(
        f'<div class="card"><div class="mut">{E(t)}</div><div class="big {"" if i == 0 else _cor(v)}">{_pct(v)}</div>'
        f'<div class="mut" style="font-size:13px">{E(s)}</div></div>'
        for i, (t, v, s) in enumerate(cards)
    )

    linhas = []
    for l in r.get("linhas", []):
        tag = '<span class="tag">visto no treino</span>' if l["visto_no_treino"] else '<span class="tag novo">nunca visto</span>'
        cel = "".join(
            f'<td class="{_cor(l.get(k)) if k != "rec_original" else ""}">{_pct(l.get(k))}</td>'
            for k in ("rec_original", "rec_protegido", "rec_rede_128k", "rec_rede_64k")
        )
        if "falso_rede_128k" in l:
            cel += f"<td>{_pct(l.get('falso_rede_128k'))}</td>"
        linhas.append(f"<tr><td>{E(l['transcritor'])} <span class='mut'>· {E(l['config'])}</span><br>{tag}</td>{cel}</tr>")

    detalhes = []
    for l in r.get("linhas", []):
        partes = "".join(
            f"<div class='mut'>{E(k)}</div><div class='tx'>{E(v) if v else '<i>(vazio)</i>'}</div>"
            for k, v in l["textos"].items()
        )
        detalhes.append(f"<details><summary>{E(l['transcritor'])} · {E(l['config'])}</summary>{partes}</details>")

    audios = "".join(
        f"<div><div class='mut'>{E(rot)}</div><audio controls preload='none' src='{E(arq)}'></audio></div>"
        for rot, arq in dados.get("audios", [])
    )

    corpo = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Relatório do escudo</title><style>{CSS}</style></head>
<body><main>
<h1>Relatório do escudo de áudio</h1>
<div class="mut">{E(dados.get('entrada', ''))} · modo <b>{E(dados.get('modo_pedido') or cfg.get('modo', ''))}</b> · força <b>{E(cfg.get('forca', ''))}</b> · perfil <b>{E(cfg.get('perfil', ''))}</b> · {dados.get('segundos_protecao', 0)} s de otimização</div>
<div class="grid">{cards_html}</div>
<p class="mut">“Recuperável” = fração das palavras de conteúdo da fala original que aparecem na transcrição. Quanto menor, melhor a proteção.
Referência: {E(r.get('fonte_referencia', ''))}.</p>

<h2>Transcritores</h2>
<div class="scroll"><table><tr><th>Transcritor</th><th>Original</th><th>Protegido</th><th>Após upload 128k</th><th>Após upload 64k</th>{'<th>Texto falso (128k)</th>' if any('falso_rede_128k' in l for l in r.get('linhas', [])) else ''}</tr>
{''.join(linhas)}</table></div>

<h2>Qualidade do áudio</h2>
<div class="grid">
<div class="card"><div class="mut">SNR (sinal/ruído)</div><div class="big">{q.get('snr_db', '—')} dB</div><div class="mut" style="font-size:13px">acima de ~30 dB costuma ser imperceptível</div></div>
<div class="card"><div class="mut">SNR segmental</div><div class="big">{q.get('snr_segmental_db', '—')} dB</div></div>
<div class="card"><div class="mut">Pico do ruído</div><div class="big">{q.get('pico_ruido_dbfs', '—')} dBFS</div></div>
<div class="card"><div class="mut">Acima do limiar de mascaramento</div><div class="big">{_pct(pc.get('fracao_acima_limiar'))}</div><div class="mut" style="font-size:13px">excesso médio {pc.get('excesso_medio_db', '—')} dB</div></div>
</div>

<h2>Ouça</h2>
<div class="card audio">{audios}</div>

<h2>Transcrições completas</h2>
{f"<details><summary>Texto falso (roteiro)</summary><div class='tx'>{E(r.get('falso') or '')}</div></details>" if r.get('falso') else ''}
<details><summary>Referência</summary><div class="tx">{E(r.get('referencia', ''))}</div></details>
{''.join(detalhes)}

<h2>Configuração</h2>
<details><summary>Parâmetros usados</summary><div class="tx">{E(json.dumps(cfg, ensure_ascii=False, indent=2))}</div></details>
</main></body></html>"""
    out = pasta / "relatorio.html"
    out.write_text(corpo, encoding="utf-8")
    return out
