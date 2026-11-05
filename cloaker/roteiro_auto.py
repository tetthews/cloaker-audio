"""Roteiro automático: transcreve a fala em trechos e preenche um texto falso neutro sozinho.

É o modo padrão. O ataque com alvo (fazer a IA escrever um texto específico, trecho a
trecho e no tempo certo) protege bem mais que só empurrar pra longe do original
(medido: 3% contra 30% de fala recuperável, com o mesmo ruído). Aqui o texto alvo vem de
um banco de frases neutras do dia a dia, cortado no tamanho de cada trecho.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Frases neutras, faladas, sem tema de negócio: viram o "texto falso" do modo automático.
BANCO = """
Hoje de manhã o tempo amanheceu nublado e eu resolvi sair mais cedo pra caminhar na praça.
A padaria da esquina estava cheia, então eu peguei só um café e voltei pra casa.
Minha vizinha plantou uns vasos de manjericão na janela e o cheiro sobe até aqui.
No fim de semana a gente foi visitar a minha tia, que mora numa cidade pequena do interior.
Lá a estrada é de terra e quando chove precisa ir devagar por causa do barro.
Ela fez um bolo de fubá com erva doce e a gente ficou conversando na varanda até tarde.
O cachorro dela é bem velhinho, passa o dia inteiro deitado no sol perto do portão.
Na volta paramos num posto pra abastecer e comprar água, porque o calor estava forte.
Eu sempre esqueço de levar guarda-chuva e acabo chegando molhado nos lugares.
Ontem arrumei o armário da cozinha e achei umas panelas que nem lembrava que tinha.
O meu sobrinho começou a aprender violão e toca a mesma música o dia inteiro.
A feira de domingo tem uma barraca de pastel que sempre forma fila antes das nove.
Eu gosto de ir cedo porque as frutas estão mais bonitas e o movimento é menor.
Comprei laranja, banana, um mamão e umas folhas de alface pra semana.
Depois passei na farmácia pra pegar um remédio e aproveitei pra medir a pressão.
A rua lá de casa está em obra e o barulho das máquinas começa às sete da manhã.
Dizem que vão trocar todo o asfalto e colocar uma calçada nova até o fim do mês.
Enquanto isso a gente estaciona o carro na rua de trás e vem andando.
Na escola das crianças vai ter uma festa junina com quadrilha e barraca de pipoca.
Cada família ficou de levar um prato, e eu fiquei de levar canjica.
A receita é da minha avó, leva leite, amendoim, canela e bastante paciência no fogo.
De noite eu gosto de ler um pouco antes de dormir, mas quase sempre pego no sono rápido.
O livro que estou lendo conta a história de um faroleiro que vive sozinho numa ilha.
Ele anota o tempo todos os dias num caderno e conversa com as gaivotas.
Semana que vem eu preciso levar o carro na oficina porque a luz do painel acendeu.
O mecânico disse que provavelmente é só um sensor, mas é melhor olhar logo.
Meu primo mudou de apartamento e pediu ajuda pra carregar as caixas no sábado.
O prédio novo não tem elevador, então foi um exercício e tanto subir até o quarto andar.
No final a gente pediu uma pizza e comeu sentado no chão da sala, no meio das caixas.
O jardim do condomínio ganhou uns bancos novos e agora tem gente lendo lá de tarde.
Os passarinhos aparecem cedo pra comer as sementes que o porteiro deixa no muro.
Eu tento acordar no mesmo horário todo dia, mas no domingo sempre durmo um pouco mais.
A chuva de ontem derrubou umas folhas grandes da mangueira lá no quintal.
Passei a tarde varrendo e separando os galhos pra colocar na calçada.
A minha mãe ligou pra contar que a novela acabou e que ela não gostou do final.
Ela disse que o casal principal merecia uma despedida mais bonita.
Na praça tem um senhor que vende algodão doce colorido e sempre conta uma piada.
As crianças fazem fila e ele faz cada algodão de uma cor diferente.
Ano que vem eu quero tirar uns dias de folga pra conhecer uma praia do sul.
Falaram que a água lá é bem gelada, mas a paisagem compensa a viagem.
""".strip().split("\n")


def transcrever_trechos(entrada: str | Path, modelo: str = "large-v3-turbo") -> list[dict]:
    """Trechos com tempo (faster-whisper). Libera a GPU no fim."""
    import gc

    import torch
    from faster_whisper import WhisperModel

    from . import midia

    a16 = midia.carregar_audio(entrada, sr=16000, canais=1)[0][0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = WhisperModel(modelo, device=dev, compute_type="float16" if dev == "cuda" else "int8")
    segs, _ = m.transcribe(a16, language="pt", beam_size=5, vad_filter=False, condition_on_previous_text=False)
    trechos = [{"ini": round(s.start, 2), "fim": round(s.end, 2), "original": s.text.strip(), "falso": ""} for s in segs]
    del m
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return [t for t in trechos if t["original"]]


def preencher_falso(trechos: list[dict], semente: int = 0) -> list[dict]:
    """Para cada trecho, um pedaço do banco com o mesmo nº de caracteres (aprox.) da fala original."""
    rng = random.Random(semente)
    frases = BANCO[:]
    rng.shuffle(frases)
    palavras = " ".join(frases).split()
    i = 0
    for t in trechos:
        alvo = max(8, len(t["original"]))
        pedaco: list[str] = []
        while len(" ".join(pedaco)) < alvo * 0.95:
            pedaco.append(palavras[i % len(palavras)])
            i += 1
        texto = " ".join(pedaco)
        t["falso"] = texto[0].upper() + texto[1:]
    return trechos


def distribuir_copy(trechos: list[dict], copy: str) -> tuple[list[dict], str | None]:
    """Espalha as palavras EXATAS da copy pelos trechos, em ordem, proporcional ao tamanho de
    cada fala (sem reescrever nada). Devolve os trechos e um aviso se a copy não couber bem."""
    palavras = copy.split()
    if not palavras or not trechos:
        return trechos, "copy vazia"
    tam_fala = [max(1, len(t["original"])) for t in trechos]
    total_fala = sum(tam_fala)
    total_copy = len(" ".join(palavras))
    razao = total_copy / total_fala
    # fronteiras acumuladas em caracteres da copy
    alvo, acum = [], 0
    for n in tam_fala:
        acum += n
        alvo.append(acum * razao)
    i, pos = 0, 0
    for k, t in enumerate(trechos):
        pedaco = []
        restantes = len(trechos) - k - 1
        while i < len(palavras) and (pos < alvo[k] or not pedaco) and len(palavras) - i > restantes:
            pedaco.append(palavras[i])
            pos += len(palavras[i]) + 1
            i += 1
        if k == len(trechos) - 1:
            pedaco += palavras[i:]
        t["falso"] = " ".join(pedaco)
    aviso = None
    if razao > 1.2:
        aviso = (f"ATENÇÃO: a copy tem {total_copy} caracteres e a fala {total_fala} ({razao:.1f}x). "
                 f"Cada trecho vai ter mais palavras do que cabe no tempo dele, e a IA lê menos da copy. "
                 f"Recomendado: encurtar a copy para ~{int(total_fala * 1.1)} caracteres (cortando frases "
                 f"inteiras) e gerar de novo.")
    elif razao < 0.7:
        aviso = (f"ATENÇÃO: a copy é bem mais curta que a fala ({total_copy} contra {total_fala} caracteres). "
                 f"Sobra tempo de fala sem texto: recomendado alongar a copy para ~{total_fala} caracteres.")
    return trechos, aviso


def _radicais(texto: str) -> set[str]:
    from .avaliacao import palavras_conteudo

    return {w[:5] if len(w) > 5 else w for w in palavras_conteudo(texto)}


def priorizar(trechos: list[dict], limiar: float = 0.6) -> tuple[list[dict], dict]:
    """Foco: compara a fala de cada trecho com o texto falso dele.

    - trecho onde os dois já dizem quase a mesma coisa (>= `limiar` das palavras da fala
      aparecem no falso, por radical): não precisa de ruído. O alvo vira a própria fala, e o
      ruído ali é zerado (menos trabalho e menos ruído).
    - nos demais, o peso cresce com a diferença e com números na fala (valores, preços,
      faixas salariais costumam ser o que mais interessa esconder).
    """
    import re

    n_pulados = 0
    for t in trechos:
        ro, rf = _radicais(t["original"]), _radicais(t["falso"])
        sim = len(ro & rf) / len(ro) if ro else 1.0
        tem_numero = bool(re.search(r"\d|\bmil\b|\breais\b|\bR\$", t["original"], re.I))
        t["similaridade"] = round(sim, 2)
        escolhido = t.get("proteger")  # o usuário pode marcar à mão: true/false no JSON
        if escolhido is False or (escolhido is None and sim >= limiar and not tem_numero):
            t["ativo"] = False
            t["peso"] = 0.0
            t["falso_pedido"] = t["falso"]
            t["falso"] = t["original"]
            n_pulados += 1
        else:
            t["ativo"] = True
            t["peso"] = round(1.0 + (1.0 - sim) + (0.5 if tem_numero else 0.0), 2)
    seg_ativos = sum(t["fim"] - t["ini"] for t in trechos if t["ativo"])
    seg_total = sum(t["fim"] - t["ini"] for t in trechos) or 1.0
    return trechos, {"ativos": len(trechos) - n_pulados, "pulados": n_pulados,
                     "fracao_tempo_ativa": round(seg_ativos / seg_total, 2)}


def gerar(entrada: str | Path, saida: str | Path, semente: int = 0, log=print) -> Path:
    log("  roteiro automático: transcrevendo a fala em trechos...")
    trechos = preencher_falso(transcrever_trechos(entrada), semente)
    dados = {"video": str(entrada), "automatico": True, "trechos": trechos}
    saida = Path(saida)
    saida.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  roteiro automático: {len(trechos)} trechos → {saida.name}")
    return saida
