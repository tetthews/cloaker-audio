"""Linha de comando.

  python -m cloaker proteger VIDEO [opções]
  python -m cloaker lote PASTA [opções]
  python -m cloaker avaliar ORIGINAL PROTEGIDO [--referencia legenda.ass]
  python -m cloaker diagnostico
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    print(msg, flush=True)


def _perfil_auto() -> str:
    """Com placa NVIDIA o perfil padrão; no Mac (MPS) e na CPU o rápido, que leva ~1/3 do tempo."""
    try:
        import torch

        return "padrao" if torch.cuda.is_available() else "rapido"
    except Exception:
        return "padrao"


def _cfg_de_args(a):
    from .config import Config

    if getattr(a, "perfil", None) is None:
        a.perfil = _perfil_auto()

    return Config.montar(
        a.forca,
        a.perfil,
        modo=a.modo,
        frase=a.frase,
        roteiro=str(Path(a.roteiro).resolve()) if getattr(a, "roteiro", None) else None,
        passos=a.passos,
        modelos=a.modelos.split(",") if a.modelos else None,
        avaliadores=a.avaliadores.split(",") if a.avaliadores else None,
        idiomas=a.idiomas.split(",") if a.idiomas else None,
        dispositivo=a.dispositivo,
        semente=a.semente,
    )


def _arquivos_da_pasta(pasta: Path) -> list[Path]:
    from .midia import EXT_AUDIO, EXT_VIDEO

    return sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXT_VIDEO | EXT_AUDIO and "_protegido" not in p.stem)


def processar(entrada: Path, pasta_saida: Path, cfg, piscina, avaliar: bool, referencia: str | None,
              descarregar_antes: bool = True) -> dict:
    from . import ataque, avaliacao, midia, relatorio

    pasta_saida.mkdir(parents=True, exist_ok=True)
    inf = midia.info(entrada)
    _log(f"\n▶ {entrada.name}  ({inf.duracao:.1f} s, {inf.sr} Hz, {inf.canais} canal(is){', com vídeo' if inf.tem_video else ''})")
    x, sr = midia.carregar_audio(entrada)

    modo_pedido = cfg.modo
    if cfg.modo == "automatico":
        import dataclasses

        from . import roteiro_auto

        piscina.liberar()  # GPU livre pro transcritor
        rp = roteiro_auto.gerar(entrada, pasta_saida / "roteiro_automatico.json", cfg.semente, log=_log)
        cfg = dataclasses.replace(cfg, modo="roteiro", roteiro=str(rp))

    with tempfile.TemporaryDirectory() as td:
        res = ataque.proteger(x, sr, cfg, piscina, log=_log, pasta_tmp=td)
    _log(f"  ✓ otimização em {res.segundos} s · sucesso nos substitutos: {res.sucesso_final}")
    _log(f"  ✓ qualidade: {res.qualidade} · percepção: {res.percepcao}")

    protegido = np.clip(x + res.delta[None], -1, 1).astype(np.float32)
    wav = pasta_saida / "audio_protegido.wav"
    midia.salvar_wav(wav, protegido, sr)
    ext = entrada.suffix.lower() if inf.tem_video else ".m4a"
    saida = pasta_saida / f"{entrada.stem}_protegido{ext}"
    midia.montar_saida(entrada, wav, saida, sr)
    midia.salvar_wav16(pasta_saida / "ruido_x10.wav", np.clip(res.delta * 10, -1, 1), sr)
    np.save(pasta_saida / "perturbacao.npy", res.delta)
    _log(f"  ✓ arquivo protegido: {saida}")

    dados = {
        "entrada": str(entrada),
        "saida": str(saida),
        "modo_pedido": modo_pedido,
        "config": cfg.como_dict(),
        "segundos_protecao": res.segundos,
        "passo_escolhido": res.passo_escolhido,
        "sucesso_substitutos": res.sucesso_final,
        "qualidade": res.qualidade,
        "percepcao": res.percepcao,
        "historico": res.historico,
    }

    # grava já (se a avaliação cair, o histórico do treino não se perde)
    (pasta_saida / "relatorio.json").write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")

    if avaliar:
        if descarregar_antes:
            piscina.descarregar()  # devolve RAM/VRAM antes de carregar os modelos da prova
        dados["avaliacao"], dados["audios"] = _avaliar(entrada, saida, pasta_saida, cfg, referencia)

    (pasta_saida / "relatorio.json").write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    rel = relatorio.gerar(pasta_saida, dados)
    _log(f"  ✓ relatório: {rel}")
    return dados


def _avaliar(original: Path, protegido: Path, pasta: Path, cfg, referencia: str | None):
    from . import avaliacao, midia

    orig_m4a = pasta / "ouvir_original.m4a"
    midia.simular_rede(original, orig_m4a, "256k", 48000)
    r128 = pasta / "rede_128k.m4a"
    r64 = pasta / "rede_64k.m4a"
    midia.simular_rede(protegido, r128, "128k", 44100)
    midia.simular_rede(protegido, r64, "64k", 44100)
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"  # faster-whisper não usa MPS
    ref_txt, fonte, falso = None, "", None
    if getattr(cfg, "roteiro", ""):
        r = json.loads(Path(cfg.roteiro).read_text(encoding="utf-8"))["trechos"]
        ref_txt = " ".join(t["original"] for t in r)
        falso = " ".join(t["falso"] for t in r)
        fonte = "trechos originais do roteiro"
    elif referencia:
        ref_txt = avaliacao.ler_referencia(referencia)
        fonte = f"arquivo {Path(referencia).name}"
    _log("  avaliando (isso transcreve cada versão com vários modelos)...")
    leve = dev != "cuda"
    if leve:
        _log("  (sem placa NVIDIA: prova leve, só original e upload 128k, 1 configuração por modelo)")
    arquivos = {"original": original, "rede_128k": r128} if leve else {
        "original": original, "protegido": protegido, "rede_128k": r128, "rede_64k": r64}
    rel = avaliacao.avaliar(
        arquivos,
        cfg.avaliadores,
        cfg.modelos,
        dev,
        referencia=ref_txt,
        fonte_ref=fonte,
        falso=falso,
        log=_log,
        leve=leve,
    )
    resumo = rel.resumo()
    _log(f"  ✓ avaliação em {rel.segundos} s · {resumo}")
    audios = [
        ("Original", orig_m4a.name),
        ("Protegido (arquivo final)", protegido.name),
        ("Protegido após simular upload 128k", r128.name),
        ("Só o ruído, 10x mais alto (pra você ouvir o que foi adicionado)", "ruido_x10.wav"),
    ]
    return {
        "falso": falso,
        "referencia": rel.referencia,
        "fonte_referencia": rel.fonte_referencia,
        "linhas": rel.linhas,
        "resumo": resumo,
        "segundos": rel.segundos,
    }, audios


def cmd_proteger(a) -> int:
    from . import ataque, dispositivo

    cfg = _cfg_de_args(a)
    dev = dispositivo.escolher(cfg.dispositivo)
    entrada = Path(a.entrada).expanduser().resolve()
    if not entrada.exists():
        _log(f"arquivo não encontrado: {entrada}")
        return 2
    pasta = Path(a.saida).expanduser().resolve() if a.saida else entrada.parent / f"{entrada.stem}_protegido"
    _log(f"dispositivo: {dispositivo.descrever(dev)}")
    _log(f"modo {cfg.modo} · força {cfg.forca} · perfil {cfg.perfil} · substitutos {cfg.modelos} · {cfg.passos} passos")
    piscina = ataque.carregar_substitutos(cfg, dev, _log)
    try:
        processar(entrada, pasta, cfg, piscina, not a.sem_avaliacao, a.referencia)
    finally:
        piscina.liberar()
    return 0


def cmd_lote(a) -> int:
    from . import ataque, dispositivo

    cfg = _cfg_de_args(a)
    dev = dispositivo.escolher(cfg.dispositivo)
    pasta = Path(a.pasta).expanduser().resolve()
    arquivos = _arquivos_da_pasta(pasta)
    if not arquivos:
        _log("nenhum vídeo/áudio na pasta")
        return 2
    destino = Path(a.saida).expanduser().resolve() if a.saida else pasta / "protegidos"
    _log(f"{len(arquivos)} arquivo(s) · dispositivo: {dispositivo.descrever(dev)}")
    piscina = ataque.carregar_substitutos(cfg, dev, _log)  # carrega uma vez só
    falhas = []
    try:
        for arq in arquivos:
            try:
                processar(arq, destino / arq.stem, cfg, piscina, a.avaliar, None, descarregar_antes=False)
                final = destino / arq.stem / f"{arq.stem}_protegido{arq.suffix.lower()}"
                if final.exists():
                    shutil.copy2(final, destino / final.name)
            except Exception as e:  # segue para o próximo
                falhas.append((arq.name, str(e)))
                _log(f"  ✗ {arq.name}: {e}")
    finally:
        piscina.liberar()
    _log(f"\nconcluído: {len(arquivos) - len(falhas)} ok, {len(falhas)} com erro · saída em {destino}")
    return 1 if falhas else 0


def cmd_avaliar(a) -> int:
    from .config import Config
    from . import relatorio

    cfg = Config.montar("media", a.perfil)
    if a.avaliadores:
        cfg.avaliadores = a.avaliadores.split(",")
    original = Path(a.original).resolve()
    protegido = Path(a.protegido).resolve()
    # quais modelos o escudo viu: --treinados, ou o relatorio.json ao lado do arquivo
    cfg.modelos = []
    if a.treinados:
        cfg.modelos = a.treinados.split(",")
    else:
        rj = protegido.parent / "relatorio.json"
        if rj.exists():
            try:
                cfg_ant = json.loads(rj.read_text(encoding="utf-8"))["config"]
                cfg.modelos = cfg_ant["modelos"]
                if cfg_ant.get("modo") == "roteiro" and cfg_ant.get("roteiro") and not a.referencia:
                    cfg.roteiro = cfg_ant["roteiro"]  # mede também o texto falso
            except Exception:
                pass
    pasta = Path(a.saida).resolve() if a.saida else protegido.parent / "avaliacao"
    pasta.mkdir(parents=True, exist_ok=True)
    av, audios = _avaliar(original, protegido, pasta, cfg, a.referencia)
    # ruido_x10 não existe aqui
    audios = [x for x in audios if x[1] != "ruido_x10.wav"]
    dados = {"entrada": str(protegido), "config": {"perfil": a.perfil, "avaliadores": cfg.avaliadores}, "avaliacao": av, "audios": audios}
    (pasta / "relatorio.json").write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"relatório: {relatorio.gerar(pasta, dados)}")
    return 0


def cmd_variacoes(a) -> int:
    """Gera N variações quase imperceptíveis (e, se pedido, protege cada uma depois)."""
    from . import variacoes

    entrada = Path(a.entrada).expanduser().resolve()
    pasta = Path(a.saida).expanduser().resolve() if a.saida else entrada.parent / f"{entrada.stem}_variacoes"
    _log(f"gerando {a.n} variação(ões) [{a.intensidade}] de {entrada.name}")
    res = variacoes.gerar(entrada, pasta, a.n, a.intensidade, a.semente, a.espelhar, log=_log)
    sobra = variacoes.metadados_restantes(pasta / res[0]["arquivo"])
    _log(f"  metadados que sobraram na 1ª variação: {sobra or 'nenhum'}")
    if not a.proteger:
        _log(f"\npronto: {pasta}")
        return 0

    from . import ataque, dispositivo

    cfg = _cfg_de_args(a)
    dev = dispositivo.escolher(cfg.dispositivo)
    piscina = ataque.carregar_substitutos(cfg, dev, _log)
    roteiro_base = json.loads(Path(cfg.roteiro).read_text(encoding="utf-8")) if cfg.modo == "roteiro" else None
    try:
        for v in res:
            arq = pasta / v["arquivo"]
            if roteiro_base:  # a velocidade mudou: os tempos dos trechos também
                r = json.loads(json.dumps(roteiro_base))
                for t in r["trechos"]:
                    t["ini"] = round(t["ini"] / v["velocidade"], 2)
                    t["fim"] = round(t["fim"] / v["velocidade"], 2)
                rp = pasta / f"{arq.stem}_roteiro.json"
                rp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
                cfg.roteiro = str(rp)
            processar(arq, pasta / f"{arq.stem}_protegido", cfg, piscina, False, None, descarregar_antes=False)
    finally:
        piscina.liberar()
    _log(f"\npronto: {pasta}")
    return 0


def cmd_roteiro(a) -> int:
    """Transcreve o vídeo em trechos com tempo e grava o JSON pro texto falso ser preenchido."""
    import torch  # noqa: F401  (DLLs do CUDA antes do ctranslate2)
    from faster_whisper import WhisperModel

    from . import midia

    entrada = Path(a.entrada).expanduser().resolve()
    saida = Path(a.saida).resolve() if a.saida else entrada.with_name(entrada.stem + "_roteiro.json")
    from . import roteiro_auto

    trechos = roteiro_auto.transcrever_trechos(entrada, a.modelo)
    aviso = None
    if a.copy:
        copy = Path(a.copy).expanduser().read_text(encoding="utf-8")
        trechos, aviso = roteiro_auto.distribuir_copy(trechos, copy)
    dados = {
        "video": str(entrada),
        "instrucao": "Preencha 'falso' em cada trecho: mesma estrutura e tamanho parecido, com outras informações.",
        "trechos": trechos,
    }
    saida.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    for t in trechos:
        _log(f"[{t['ini']:6.2f}–{t['fim']:6.2f}] FALA:  {t['original']}")
        if t["falso"]:
            _log(f"                 FALSO: {t['falso']}")
    if aviso:
        _log("")
        _log(aviso)
    _log(f"\nroteiro: {saida}")
    return 0


def cmd_baixar(a) -> int:
    """Baixa de uma vez os modelos do perfil (treino e prova), pra o 1º uso não esperar download."""
    from .config import PERFIS
    from .substitutos import ALIAS_W2V

    p = PERFIS[a.perfil or _perfil_auto()]
    falhas = 0
    for nome in p["modelos"]:
        try:
            if nome.startswith("w2v"):
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                Wav2Vec2Processor.from_pretrained(ALIAS_W2V[nome]); Wav2Vec2ForCTC.from_pretrained(ALIAS_W2V[nome])
            else:
                import whisper

                root = str(Path.home() / ".cache" / "whisper")
                whisper._download(whisper._MODELS[nome], root, False)
            _log(f"  ok: {nome} (treino)")
        except Exception as e:
            falhas += 1
            _log(f"  falhou {nome}: {e}")
    for nome in p["avaliadores"]:
        try:
            if nome.startswith("w2v"):
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                Wav2Vec2Processor.from_pretrained(ALIAS_W2V[nome]); Wav2Vec2ForCTC.from_pretrained(ALIAS_W2V[nome])
            else:
                from faster_whisper.utils import download_model

                download_model(nome)
            _log(f"  ok: {nome} (prova)")
        except Exception as e:
            falhas += 1
            _log(f"  falhou {nome}: {e}")
    return 1 if falhas else 0


def cmd_diagnostico(a) -> int:
    import platform

    import torch

    from . import __version__, dispositivo, midia

    dev = dispositivo.escolher("auto")
    _log(f"cloaker {__version__} · Python {platform.python_version()} · {platform.system()} {platform.machine()}")
    _log(f"torch {torch.__version__} · dispositivo: {dispositivo.descrever(dev)}")
    _log(f"ffmpeg: {midia.ffmpeg()}")
    _log(f"codecs para robustez: {[c[0] + '@' + c[2] for c in midia.codecs_validos()]}")
    for mod in ("whisper", "faster_whisper", "transformers", "torchaudio"):
        try:
            m = __import__(mod)
            _log(f"{mod}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            _log(f"{mod}: AUSENTE ({e})")
    return 0


def _opcoes_comuns(p: argparse.ArgumentParser) -> None:
    from .config import FORCAS, MODOS, PERFIS

    p.add_argument("--modo", choices=MODOS, default="automatico",
                   help="automatico (padrão): transcreve e troca cada trecho por um texto falso neutro, sozinho · "
                        "silenciar: a IA transcreve nada · embaralhar: a IA transcreve lixo sem relação com a fala "
                        "(mais fácil de conseguir) · trocar: a IA transcreve a --frase no lugar da sua fala · "
                        "roteiro: a IA transcreve o texto falso de cada trecho (--roteiro)")
    p.add_argument("--frase", default=None, help="frase isca do modo trocar")
    p.add_argument("--roteiro", default=None, help="modo roteiro: JSON gerado pelo comando `roteiro`, com o texto falso preenchido")
    p.add_argument("--forca", choices=list(FORCAS), default="media", help="quanto ruído é permitido (padrão: media)")
    p.add_argument("--perfil", choices=list(PERFIS), default=None,
                   help="esforço: rapido, padrao, maximo (padrão: padrao com placa NVIDIA, rapido no Mac/CPU)")
    p.add_argument("--passos", type=int, default=None, help="sobrescreve o nº de passos do perfil")
    p.add_argument("--modelos", default=None, help="substitutos separados por vírgula (ex.: small,medium,turbo)")
    p.add_argument("--avaliadores", default=None, help="transcritores da prova (ex.: large-v3,distil-large-v3)")
    p.add_argument("--idiomas", default=None, help="idiomas forçados no treino (padrão: pt)")
    p.add_argument("--dispositivo", default="auto", help="auto, cuda, mps ou cpu")
    p.add_argument("--semente", type=int, default=1234)
    p.add_argument("--saida", default=None, help="pasta de saída")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cloaker", description="Escudo de áudio contra transcrição por IA.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("proteger", help="protege um vídeo/áudio")
    p.add_argument("entrada")
    _opcoes_comuns(p)
    p.add_argument("--referencia", default=None, help="legenda/texto do que foi falado (.ass, .srt, .txt) para medir melhor")
    p.add_argument("--sem-avaliacao", action="store_true", help="pula a prova final (mais rápido)")
    p.set_defaults(fn=cmd_proteger)

    p = sub.add_parser("lote", help="protege todos os vídeos de uma pasta")
    p.add_argument("pasta")
    _opcoes_comuns(p)
    p.add_argument("--avaliar", action="store_true", help="roda a prova em cada arquivo (lento)")
    p.set_defaults(fn=cmd_lote)

    p = sub.add_parser("avaliar", help="mede a proteção de um arquivo já protegido")
    p.add_argument("original")
    p.add_argument("protegido")
    p.add_argument("--referencia", default=None)
    p.add_argument("--perfil", default="padrao")
    p.add_argument("--avaliadores", default=None)
    p.add_argument("--treinados", default=None, help="modelos contra os quais o arquivo foi protegido")
    p.add_argument("--saida", default=None)
    p.set_defaults(fn=cmd_avaliar)

    p = sub.add_parser("variacoes", help="gera variações quase imperceptíveis pra teste de criativo")
    p.add_argument("entrada")
    p.add_argument("-n", type=int, default=5, help="quantas variações (padrão 5)")
    p.add_argument("--intensidade", choices=["leve", "media", "forte"], default="leve")
    p.add_argument("--espelhar", action="store_true", help="permite espelhar (cuidado com texto na tela)")
    p.add_argument("--proteger", action="store_true", help="protege cada variação depois de gerar")
    _opcoes_comuns(p)
    p.set_defaults(fn=cmd_variacoes, semente=7)

    p = sub.add_parser("roteiro", help="divide a fala em trechos com tempo (para o modo roteiro)")
    p.add_argument("entrada")
    p.add_argument("--modelo", default="large-v3-turbo")
    p.add_argument("--copy", default=None,
                   help="arquivo .txt com a copy: distribui as palavras EXATAS dela pelos trechos (sem reescrever)")
    p.add_argument("--saida", default=None)
    p.set_defaults(fn=cmd_roteiro)

    p = sub.add_parser("baixar", help="baixa os modelos do perfil de uma vez")
    p.add_argument("--perfil", default=None)
    p.set_defaults(fn=cmd_baixar)

    p = sub.add_parser("diagnostico", help="confere GPU, ffmpeg e dependências")
    p.set_defaults(fn=cmd_diagnostico)

    a = ap.parse_args(argv)
    if getattr(a, "frase", "x") is None:
        from .config import FRASE_PADRAO

        a.frase = FRASE_PADRAO
    t0 = time.time()
    rc = a.fn(a)
    _log(f"tempo total: {time.time() - t0:.0f} s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
