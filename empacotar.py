"""Gera dist/cloaker-audio.zip: o pacote pronto pra Windows e Mac (o que está no git).

Feito com zipfile para gravar a permissão de execução (+x) dos .command/.sh com final de
linha LF (o Finder respeita ao descompactar; zip feito pelo Windows perde isso) e CRLF
nos .bat.

    python empacotar.py
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
NOME = "cloaker-audio"
EXEC = {".command", ".sh"}
TEXTO = {".py", ".txt", ".md", ".sh", ".command", ".ps1", ".bat", ".gitignore", ".gitattributes"}
FORA = {"empacotar.py"}


def arquivos() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", check=True)
    return [l for l in out.stdout.splitlines() if l and not l.startswith("tests/") and l not in FORA]


def main() -> Path:
    dist = RAIZ / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"{NOME}.zip"
    lista = arquivos()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in lista:
            p = RAIZ / rel
            dados = p.read_bytes()
            if p.suffix in TEXTO or p.name in TEXTO:
                dados = dados.replace(b"\r\n", b"\n")
                if p.suffix == ".bat":
                    dados = dados.replace(b"\n", b"\r\n")
            info = zipfile.ZipInfo(f"{NOME}/{rel}")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # unix: o Finder lê as permissões abaixo
            modo = 0o755 if p.suffix in EXEC else 0o644
            info.external_attr = (0o100000 | modo) << 16
            z.writestr(info, dados)
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB, {len(lista)} arquivos)")
    return out


if __name__ == "__main__":
    main()
