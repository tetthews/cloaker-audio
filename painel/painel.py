#!/usr/bin/env python3
"""Painel local do Cloaker de áudio.

Fica numa página no navegador (http://127.0.0.1:8765). Só usa a biblioteca padrão do
Python e chama o próprio comando `python -m cloaker` do programa instalado. Tudo que o
painel cria vai para `painel-trabalhos/` na raiz (fora do git).

Uso (ou pelos atalhos "3 - Abrir o painel" da raiz):
    .venv/bin/python painel/painel.py                 (Mac)
    .venv\\Scripts\\python.exe painel\\painel.py         (Windows)
    opções: --pasta CAMINHO_DO_CLOAKER --porta 8765 --sem-abrir
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

AQUI = Path(__file__).resolve().parent
WINDOWS = platform.system() == "Windows"
EXT_OK = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".m4a", ".mp3", ".wav"}
LOCK = threading.RLock()
JOBS: dict[str, dict] = {}
FILA: list[tuple[str, str]] = []  # (id do trabalho, fase)
ATUAL: dict = {"id": None, "proc": None, "pid": None}
PASTA: Path = AQUI
TRABALHOS: Path = AQUI


# ----------------------------------------------------------------------------- pasta do Cloaker

def achar_pasta(explicita: str | None) -> Path:
    candidatos = []
    if explicita:
        candidatos.append(Path(explicita).expanduser())
    if os.environ.get("CLOAKER_DIR"):
        candidatos.append(Path(os.environ["CLOAKER_DIR"]).expanduser())
    candidatos += [AQUI, AQUI.parent, Path.home() / "Documents" / "cloaker-audio", Path("C:/cloaker-audio")]
    for c in candidatos:
        if (c / "cloaker").is_dir() and (c / "cloaker" / "cli.py").exists():
            return c.resolve()
    sys.exit("Não achei a pasta do Cloaker. Rode de novo com: --pasta CAMINHO_DA_PASTA")


def python_do_cloaker() -> Path | None:
    p = PASTA / (".venv/Scripts/python.exe" if WINDOWS else ".venv/bin/python")
    return p if p.exists() else None


def ambiente() -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    expat = PASTA / ".expat"
    if expat.exists():  # correção do Python do Homebrew no macOS 26, igual ao cloaker.sh
        env["DYLD_LIBRARY_PATH"] = expat.read_text().strip() + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    return env


# ----------------------------------------------------------------------------- trabalhos

def limpar_nome(n: str) -> str:
    n = Path(n).name
    base, ext = os.path.splitext(n)
    base = re.sub(r"[^\w\-. ]+", "_", base, flags=re.UNICODE).strip(" ._") or "video"
    return base[:80] + ext.lower()


def pasta_job(jid: str) -> Path:
    if not re.fullmatch(r"[\w\-]+", jid):
        raise KeyError(jid)
    p = (TRABALHOS / jid).resolve()
    if TRABALHOS.resolve() not in p.parents:
        raise KeyError(jid)
    return p


def salvar(job: dict) -> None:
    with LOCK:
        (pasta_job(job["id"]) / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def carregar_existentes() -> None:
    TRABALHOS.mkdir(exist_ok=True)
    for d in sorted(TRABALHOS.iterdir()):
        f = d / "job.json"
        if f.exists():
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            JOBS[j["id"]] = j
    # o painel foi fechado no meio: a proteção roda num processo à parte e pode ter seguido sozinha
    for j in sorted(JOBS.values(), key=lambda x: x.get("criado", 0)):
        if j.get("estado") == "rodando":
            if ATUAL["id"] is None and pid_vivo(j.get("pid")):
                ATUAL.update(id=j["id"], pid=j["pid"])
                threading.Thread(target=acompanhar_orfao, args=(j["id"],), daemon=True).start()
            else:
                concluir(j, j.get("fase"), None)
        elif j.get("estado") == "fila":
            FILA.append((j["id"], j.get("fase") or "proteger"))


def pasta_saida(job: dict) -> Path:
    return pasta_job(job["id"]) / f"{Path(job['arquivo']).stem}_protegido"


def resumo_resultado(job: dict) -> dict | None:
    saida = pasta_saida(job)
    rj = saida / "relatorio.json"
    if not rj.exists():
        return None
    try:
        d = json.loads(rj.read_text(encoding="utf-8"))
    except Exception:
        return None
    arq = Path(d.get("saida", ""))
    return {
        "arquivo": arq.name if arq.name else None,
        "existe": (saida / arq.name).exists() if arq.name else False,
        "relatorio_html": (saida / "relatorio.html").exists(),
        "segundos_protecao": d.get("segundos_protecao"),
        "qualidade": d.get("qualidade"),
        "percepcao": d.get("percepcao"),
        "sucesso_substitutos": d.get("sucesso_substitutos"),
        "avaliacao": (d.get("avaliacao") or {}).get("resumo"),
        "modo": d.get("modo_pedido"),
        "perfil": (d.get("config") or {}).get("perfil"),
        "forca": (d.get("config") or {}).get("forca"),
    }


def visao(job: dict, completo: bool = False) -> dict:
    v = {k: job.get(k) for k in ("id", "nome", "arquivo", "estado", "fase", "criado", "inicio", "fim", "opcoes", "erro", "tem_roteiro")}
    v["posicao_fila"] = next((i + 1 for i, (jid, _) in enumerate(FILA) if jid == job["id"]), None)
    v["resultado"] = resumo_resultado(job) if job.get("estado") == "ok" else None
    if completo:
        v["pasta"] = str(pasta_job(job["id"]))
    return v


# ----------------------------------------------------------------------------- execução

def montar_comando(job: dict, fase: str) -> list[str]:
    py = python_do_cloaker()
    base = [str(py), "-m", "cloaker"]
    pj = pasta_job(job["id"])
    video = str(pj / job["arquivo"])
    o = job.get("opcoes") or {}
    if fase == "preparar":
        cmd = base + ["roteiro", video, "--saida", str(pj / "roteiro.json")]
        if (pj / "copy.txt").exists():
            cmd += ["--copy", str(pj / "copy.txt")]
        return cmd
    cmd = base + ["proteger", video, "--modo", o.get("modo", "automatico"), "--forca", o.get("forca", "media")]
    if o.get("perfil") and o["perfil"] != "auto":
        cmd += ["--perfil", o["perfil"]]
    if o.get("modo") == "roteiro":
        cmd += ["--roteiro", str(pj / "roteiro_final.json")]
    if o.get("sem_avaliacao"):
        cmd.append("--sem-avaliacao")
    return cmd


def rodar(jid: str, fase: str) -> None:
    job = JOBS[jid]
    pj = pasta_job(jid)
    log = open(pj / "painel.log", "a", encoding="utf-8", errors="replace")
    with LOCK:
        job.update(estado="rodando", fase=fase, inicio=time.time(), erro=None)
        ATUAL["id"] = jid
        salvar(job)
    log.write(f"\n===== {time.strftime('%H:%M:%S')} · {'preparando o roteiro' if fase == 'preparar' else 'protegendo'} =====\n")
    log.flush()
    rc = -1
    try:
        cmd = montar_comando(job, fase)
        # processo à parte (grupo e sessão próprios): fechar a janela do painel não derruba a proteção,
        # e o cancelar consegue parar o grupo inteiro. No Windows: novo grupo + sem janela de console.
        kw = {"creationflags": 0x00000200 | 0x08000000} if WINDOWS else {"start_new_session": True}
        proc = subprocess.Popen(cmd, cwd=PASTA, env=ambiente(), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
        with LOCK:
            ATUAL.update(proc=proc, pid=proc.pid)
            job["pid"] = proc.pid
            salvar(job)
        rc = proc.wait()
    except Exception as e:  # noqa: BLE001
        log.write(f"[painel] erro ao iniciar: {e}\n")
    finally:
        ATUAL.update(id=None, proc=None, pid=None)
        log.close()
    concluir(job, fase, rc)


def concluir(job: dict, fase: str | None, rc: int | None) -> None:
    """Fecha o trabalho olhando o que ficou na pasta. rc None: processo que o painel reaberto não iniciou."""
    pj = pasta_job(job["id"])
    with LOCK:
        job["fim"] = time.time()
        job.pop("pid", None)
        if job.get("cancelado"):
            job.update(estado="cancelado", erro=None)
        elif rc not in (0, None):
            job.update(estado="erro", erro=f"O programa terminou com erro (código {rc}). Veja o registro abaixo.")
        elif fase == "preparar":
            ok = (pj / "roteiro.json").exists()
            job.update(estado="aguardando_texto" if ok else "erro", tem_roteiro=ok,
                       erro=None if ok else "A transcrição parou antes de terminar. Envie o vídeo de novo.")
        else:
            ok = (pasta_saida(job) / "relatorio.html").exists()  # é o último arquivo que o programa grava
            falha = ("Terminou, mas não gerou o resultado. Veja o registro." if rc == 0 else
                     "A proteção parou antes de terminar (o computador desligou ou o processo foi fechado). Envie o vídeo de novo.")
            job.update(estado="ok" if ok else "erro", erro=None if ok else falha)
        salvar(job)


def pid_vivo(pid) -> bool:
    """O processo ainda existe e é o do Cloaker (o número pode ter sido reaproveitado por outro programa)."""
    if not pid:
        return False
    try:
        if WINDOWS:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"],
                                 capture_output=True, text=True, creationflags=0x08000000).stdout
            return f'"{int(pid)}"' in out and "python" in out.lower()
        out = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True, text=True).stdout
        return "cloaker" in out
    except Exception:
        return False


def acompanhar_orfao(jid: str) -> None:
    """Proteção que seguiu rodando enquanto o painel estava fechado: espera ela acabar e fecha o trabalho."""
    job = JOBS[jid]
    while pid_vivo(job.get("pid")):
        time.sleep(5)
    with LOCK:
        ATUAL.update(id=None, proc=None, pid=None)
    concluir(job, job.get("fase"), None)


def trabalhador() -> None:
    while True:
        item = None
        with LOCK:
            if FILA and ATUAL["id"] is None:
                item = FILA.pop(0)
        if item and item[0] in JOBS:
            rodar(*item)
        else:
            time.sleep(0.6)


def enfileirar(jid: str, fase: str) -> None:
    with LOCK:
        job = JOBS[jid]
        job.update(estado="fila", fase=fase, erro=None, cancelado=False)
        FILA.append((jid, fase))
        salvar(job)


def matar_atual() -> None:
    proc, pid = ATUAL.get("proc"), ATUAL.get("pid")
    if proc is not None and proc.poll() is not None:
        return
    if proc is None and not pid_vivo(pid):
        return
    try:
        if WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            import signal

            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        if proc is not None:
            proc.terminate()


# ----------------------------------------------------------------------------- HTTP

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silencioso
        pass

    # util ---------------------------------------------------------------
    def json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def corpo_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}

    def host_ok(self) -> bool:
        h = (self.headers.get("Host") or "").split(":")[0]
        return h in ("127.0.0.1", "localhost")

    def arquivo(self, caminho: Path, baixar_como: str | None = None):
        if not caminho.is_file():
            return self.json({"erro": "arquivo não encontrado"}, 404)
        tam = caminho.stat().st_size
        ini, fim, code = 0, tam - 1, 200
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            if m.group(1):
                ini = int(m.group(1))
                fim = int(m.group(2)) if m.group(2) else tam - 1
            elif m.group(2):
                ini = max(0, tam - int(m.group(2)))
            fim = min(fim, tam - 1)
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", mimetypes.guess_type(str(caminho))[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(fim - ini + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {ini}-{fim}/{tam}")
        if baixar_como:
            self.send_header("Content-Disposition", f'attachment; filename="{baixar_como}"')
        self.end_headers()
        with open(caminho, "rb") as f:
            f.seek(ini)
            resta = fim - ini + 1
            try:
                while resta > 0:
                    pedaco = f.read(min(1 << 20, resta))
                    if not pedaco:
                        break
                    self.wfile.write(pedaco)
                    resta -= len(pedaco)
            except (BrokenPipeError, ConnectionResetError):
                pass

    # GET ----------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        if not self.host_ok():
            return self.json({"erro": "host inválido"}, 403)
        u = urlparse(self.path)
        p = unquote(u.path)
        q = parse_qs(u.query)
        try:
            if p == "/":
                return self.arquivo(AQUI / "index.html")
            if p == "/api/estado":
                with LOCK:
                    jobs = sorted(JOBS.values(), key=lambda j: j.get("criado", 0), reverse=True)
                    return self.json({
                        "pasta": str(PASTA),
                        "instalado": python_do_cloaker() is not None,
                        "sistema": platform.system(),
                        "em_execucao": ATUAL["id"],
                        "trabalhos": [visao(j) for j in jobs],
                    })
            m = re.fullmatch(r"/api/jobs/([\w\-]+)", p)
            if m:
                j = JOBS[m.group(1)]
                lf = pasta_job(j["id"]) / "painel.log"
                de = int(q.get("de", ["0"])[0])
                txt, novo = "", de
                if lf.exists():
                    with open(lf, "rb") as f:
                        f.seek(de)
                        bruto = f.read()
                        novo = de + len(bruto)
                        txt = bruto.decode("utf-8", errors="replace")
                v = visao(j, True)
                v.update(log=txt, log_pos=novo)
                return self.json(v)
            m = re.fullmatch(r"/api/jobs/([\w\-]+)/roteiro", p)
            if m:
                j = JOBS[m.group(1)]
                f = pasta_job(j["id"]) / "roteiro.json"
                return self.json(json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"trechos": []})
            m = re.fullmatch(r"/arquivos/([\w\-]+)/(.+)", p)
            if m:
                j = JOBS[m.group(1)]
                base = pasta_job(j["id"]).resolve()
                alvo = (base / m.group(2)).resolve()
                if base not in alvo.parents:
                    return self.json({"erro": "caminho inválido"}, 403)
                return self.arquivo(alvo, alvo.name if q.get("baixar") else None)
        except KeyError:
            return self.json({"erro": "não encontrado"}, 404)
        return self.json({"erro": "não encontrado"}, 404)

    # POST ---------------------------------------------------------------
    def do_POST(self):  # noqa: N802
        if not self.host_ok():
            return self.json({"erro": "host inválido"}, 403)
        u = urlparse(self.path)
        p = u.path
        try:
            if p == "/api/enviar":  # corpo cru = o arquivo; nome no cabeçalho
                nome = limpar_nome(unquote(self.headers.get("X-Nome", "video.mp4")))
                if Path(nome).suffix.lower() not in EXT_OK:
                    return self.json({"erro": f"formato não aceito ({Path(nome).suffix}). Use: " + ", ".join(sorted(EXT_OK))}, 400)
                if python_do_cloaker() is None:
                    return self.json({"erro": "O Cloaker não está instalado nesta pasta (falta a pasta .venv). Rode o passo 2 de instalação."}, 400)
                jid = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"\W+", "", Path(nome).stem)[:20]
                pj = TRABALHOS / jid
                pj.mkdir(parents=True)
                n = int(self.headers.get("Content-Length") or 0)
                with open(pj / nome, "wb") as f:
                    resta = n
                    while resta > 0:
                        b = self.rfile.read(min(1 << 20, resta))
                        if not b:
                            break
                        f.write(b)
                        resta -= len(b)
                job = {"id": jid, "nome": Path(nome).stem, "arquivo": nome, "estado": "enviado", "fase": None, "criado": time.time(), "opcoes": {}}
                with LOCK:
                    JOBS[jid] = job
                    salvar(job)
                return self.json({"id": jid})

            if p == "/api/diagnostico":
                py = python_do_cloaker()
                if not py:
                    return self.json({"ok": False, "texto": "O Cloaker não está instalado nesta pasta (falta a pasta .venv)."})
                if ATUAL["id"]:
                    return self.json({"ok": False, "texto": "Tem um vídeo sendo protegido agora. Espere terminar para rodar o diagnóstico."})
                try:
                    r = subprocess.run([str(py), "-m", "cloaker", "diagnostico"], cwd=PASTA, env=ambiente(), capture_output=True, timeout=180)
                    return self.json({"ok": r.returncode == 0, "texto": (r.stdout + r.stderr).decode("utf-8", errors="replace")})
                except subprocess.TimeoutExpired:
                    return self.json({"ok": False, "texto": "O diagnóstico demorou demais (mais de 3 minutos)."})

            m = re.fullmatch(r"/api/jobs/([\w\-]+)/(iniciar|proteger|cancelar|excluir|abrir)", p)
            if not m:
                return self.json({"erro": "não encontrado"}, 404)
            jid, acao = m.groups()
            job = JOBS[jid]
            pj = pasta_job(jid)
            d = self.corpo_json()

            if acao == "iniciar":
                o = {
                    "modo": d.get("modo") if d.get("modo") in ("automatico", "roteiro") else "automatico",
                    "forca": d.get("forca") if d.get("forca") in ("leve", "media", "forte", "maxima") else "media",
                    "perfil": d.get("perfil") if d.get("perfil") in ("auto", "expresso", "rapido", "padrao", "maximo") else "auto",
                    "sem_avaliacao": bool(d.get("sem_avaliacao")),
                }
                job["opcoes"] = o
                if o["modo"] == "roteiro":
                    copy = (d.get("copy") or "").strip()
                    if copy:
                        (pj / "copy.txt").write_text(copy, encoding="utf-8")
                    elif (pj / "copy.txt").exists():
                        (pj / "copy.txt").unlink()
                    enfileirar(jid, "preparar")
                else:
                    enfileirar(jid, "proteger")
                return self.json({"ok": True})

            if acao == "proteger":  # modo roteiro: texto falso já conferido na tela
                trechos = d.get("trechos") or []
                if not trechos or not any(t.get("proteger", True) for t in trechos):
                    return self.json({"erro": "Marque pelo menos um trecho para proteger."}, 400)
                # texto falso maior que a fala: a IA lê metade (teste de 22/11/2026: 17% contra 32% da copy)
                prot = [t for t in trechos if t.get("proteger", True)]
                letras = lambda x: len(" ".join((x or "").split()))
                fala = sum(letras(t.get("original")) for t in prot)
                falso = sum(letras(t.get("falso")) for t in prot)
                limite = int(fala * 1.2)
                if falso > limite:
                    return self.json({"erro": f"O texto falso tem {falso} letras e a fala tem {fala}. Corte frases inteiras, "
                                              f"umas {-(-(falso - limite) // 6)} palavras, até ficar com no máximo {limite} letras."}, 400)
                base = json.loads((pj / "roteiro.json").read_text(encoding="utf-8"))
                base["trechos"] = trechos
                (pj / "roteiro_final.json").write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
                o = job.get("opcoes") or {}
                for k in ("forca", "perfil"):
                    if d.get(k):
                        o[k] = d[k]
                if "sem_avaliacao" in d:
                    o["sem_avaliacao"] = bool(d["sem_avaliacao"])
                o["modo"] = "roteiro"
                job["opcoes"] = o
                enfileirar(jid, "proteger")
                return self.json({"ok": True})

            if acao == "cancelar":
                with LOCK:
                    if ATUAL["id"] == jid:
                        job["cancelado"] = True
                        matar_atual()
                    else:
                        FILA[:] = [x for x in FILA if x[0] != jid]
                        job.update(estado="cancelado")
                        salvar(job)
                return self.json({"ok": True})

            if acao == "excluir":
                if ATUAL["id"] == jid:
                    return self.json({"erro": "Cancele antes de excluir."}, 400)
                with LOCK:
                    FILA[:] = [x for x in FILA if x[0] != jid]
                    JOBS.pop(jid, None)
                shutil.rmtree(pj, ignore_errors=True)
                return self.json({"ok": True})

            if acao == "abrir":
                alvo = str(pasta_saida(job) if pasta_saida(job).exists() else pj)
                if WINDOWS:
                    os.startfile(alvo)  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["open" if platform.system() == "Darwin" else "xdg-open", alvo])
                return self.json({"ok": True})
        except KeyError:
            return self.json({"erro": "não encontrado"}, 404)
        except Exception as e:  # noqa: BLE001
            return self.json({"erro": str(e)}, 500)
        return self.json({"erro": "não encontrado"}, 404)


def main() -> None:
    global PASTA, TRABALHOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=None, help="pasta do Cloaker (onde está o cloaker.sh)")
    ap.add_argument("--porta", type=int, default=8765)
    ap.add_argument("--sem-abrir", action="store_true")
    a = ap.parse_args()
    PASTA = achar_pasta(a.pasta)
    TRABALHOS = PASTA / "painel-trabalhos"
    carregar_existentes()
    threading.Thread(target=trabalhador, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.porta), H)
    url = f"http://127.0.0.1:{a.porta}"
    print(f"Painel do Cloaker no ar: {url}\nPasta do Cloaker: {PASTA}\nFeche esta janela para desligar o painel. Uma proteção em andamento continua.", flush=True)
    if not a.sem_abrir:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        pass  # fechar o painel não para a proteção: ela segue e o painel retoma o acompanhamento ao reabrir


if __name__ == "__main__":
    main()
