"""Modelos substitutos diferenciáveis: o escudo é treinado contra eles.

- Família Whisper (openai-whisper): é o que quase todo transcritor usa por baixo
  (TurboScribe, faster-whisper, apps de legenda, scripts com ChatGPT/Claude).
- wav2vec2 (CTC) em português, opcional: arquitetura diferente, ajuda a transferência.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

SR_ASR = 16000
N_AMOSTRAS = 30 * SR_ASR  # janela de 30 s do Whisper
N_QUADROS = 3000

ALIAS_W2V = {
    "w2v-pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "w2v-coraa": "Edresson/wav2vec2-large-xlsr-coraa-portuguese",
}


def _congelar(m: torch.nn.Module) -> torch.nn.Module:
    m.train(False)  # modo inferência (desliga dropout)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@dataclass
class Saida:
    perda: torch.Tensor
    sucesso: torch.Tensor  # (B,) bool: o modelo já "obedece" nesse recorte
    valor: float | None = None  # valor real da perda quando 'perda' é substituta


class Substituto:
    nome: str
    tamanho_gb: float

    def para(self, dev: torch.device) -> None: ...
    def perda(self, lote16: torch.Tensor, comprimentos: list[int], kappa: float = 0.02, peso_ns: float = 0.3) -> Saida: ...


# --------------------------------------------------------------------------- Whisper


class WhisperSubstituto(Substituto):
    def __init__(self, nome: str, cfg, dtype: torch.dtype):
        import whisper
        from whisper.tokenizer import get_tokenizer

        self.nome = nome
        self.dtype = dtype
        m = whisper.load_model(nome, device="cpu")
        # buffer esparso quebra no MPS; denso é inofensivo
        try:
            m.register_buffer("alignment_heads", m.alignment_heads.to_dense(), persistent=False)
        except Exception:
            pass
        if dtype == torch.float16:
            m = m.half()
            for mod in m.modules():
                if isinstance(mod, torch.nn.LayerNorm):
                    mod.float()  # whisper faz LayerNorm em fp32
        self.m = _congelar(m)
        self.n_mels = m.dims.n_mels
        self.tamanho_gb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1024**3
        self.dev = torch.device("cpu")
        self._filtros = None

        # monta as sequências alvo (teacher forcing)
        self.variantes: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        tok0 = get_tokenizer(m.is_multilingual, num_languages=m.num_languages, language="pt", task="transcribe")
        self.no_speech = tok0.no_speech
        self.modo = cfg.modo
        self.tok = tok0
        self.prefixo_pt = list(tok0.sot_sequence) + [tok0.no_timestamps]
        self.ref_palavras: set[str] = set()  # preenchido pelo ataque (transcrição do original)
        self.peso_eot_embaralhar = 0.3
        self.tau = 5.0  # embaralhar: para de empurrar quando a perplexidade já é alta
        for lang in cfg.idiomas:
            tok = get_tokenizer(m.is_multilingual, num_languages=m.num_languages, language=lang, task="transcribe")
            sot_seq = list(tok.sot_sequence)
            texto: list[int] = [] if cfg.modo in ("silenciar", "embaralhar") else tok.encode(" " + cfg.frase.strip())
            # modo sem timestamps
            self._add(sot_seq + [tok.no_timestamps], texto + [tok.eot])
            # modo com timestamps (transcribe() padrão do whisper e do faster-whisper)
            ts0 = tok.timestamp_begin
            if cfg.modo in ("silenciar", "embaralhar"):
                alvo_ts = [ts0, tok.eot]
            else:
                alvo_ts = [ts0] + texto + [tok.timestamp_begin + 150, tok.eot]  # <|3.00|>
            self._add(sot_seq, alvo_ts)

    def _add(self, prefixo: list[int], alvo: list[int]) -> None:
        seq = prefixo + alvo
        entrada = torch.tensor(seq[:-1], dtype=torch.long)
        rotulos = torch.tensor(alvo, dtype=torch.long)
        self.variantes.append((entrada, rotulos, len(prefixo) - 1))

    def para(self, dev: torch.device) -> None:
        if self.dev == dev:
            return
        self.m.to(dev)
        self.dev = dev
        self.variantes = [(e.to(dev), r.to(dev), i) for e, r, i in self.variantes]
        self._filtros = None

    def log_mel(self, lote16: torch.Tensor) -> torch.Tensor:
        """Log-mel idêntico ao do Whisper, mas em lote e diferenciável."""
        from whisper.audio import mel_filters

        if self._filtros is None:
            self._filtros = mel_filters(self.dev, self.n_mels)
            self._janela = torch.hann_window(400, device=self.dev)
        x = lote16.float()
        st = torch.stft(x, 400, 160, window=self._janela, return_complex=True)
        mag = st[..., :-1].abs().pow(2)
        mel = self._filtros @ mag
        log = mel.clamp(min=1e-10).log10()
        topo = log.amax(dim=(-2, -1), keepdim=True)
        log = torch.maximum(log, topo - 8.0)
        return (log + 4.0) / 4.0

    def _encoder(self, mel: torch.Tensor) -> torch.Tensor:
        """Encoder do Whisper com gradient checkpointing por bloco: guarda só a
        entrada de cada bloco e recalcula o resto no backward (cabe em 8 GB)."""
        from torch.utils.checkpoint import checkpoint

        enc = self.m.encoder
        x = F.gelu(enc.conv1(mel))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + enc.positional_embedding).to(x.dtype)
        for bloco in enc.blocks:
            x = checkpoint(bloco, x, use_reentrant=False)
        return enc.ln_post(x)

    def perda(self, lote16, comprimentos, kappa=0.02, peso_ns=0.3) -> Saida:
        mel = self.log_mel(lote16).to(self.dtype)
        feats = self._encoder(mel)
        B = lote16.shape[0]
        total = lote16.new_zeros(())
        ok = torch.ones(B, dtype=torch.bool, device=lote16.device)
        for k, (entrada, rotulos, ini) in enumerate(self.variantes):
            toks = entrada[None].expand(B, -1)
            logits = self.m.logits(toks, feats).float()  # (B, L, V)
            alvo_logits = logits[:, ini:]
            ce = F.cross_entropy(
                alvo_logits.reshape(-1, alvo_logits.shape[-1]), rotulos.repeat(B), reduction="none"
            ).view(B, -1)
            total = total + torch.relu(ce - kappa).mean()
            ok &= (alvo_logits.argmax(-1) == rotulos[None]).all(-1)
            if k == 0 and peso_ns > 0:
                alvo_ns = torch.full((B,), self.no_speech, device=logits.device)
                ns = F.cross_entropy(logits[:, 0], alvo_ns, reduction="none")
                total = total + peso_ns * torch.relu(ns - 0.1).mean()
        total = total / len(self.variantes)
        if self.modo != "embaralhar":
            return Saida(total, ok)
        return self._embaralhar(feats, total, ok)

    def _ts(self, t: float) -> int:
        return self.tok.timestamp_begin + int(min(1500, max(0, round(t / 0.02))))

    def _seq_roteiro(self, alvo: list[dict], chave: str, com_ts: bool) -> tuple[list[int], list[int]]:
        enc = self.tok.encode
        if not com_ts:
            pre = list(self.tok.sot_sequence) + [self.tok.no_timestamps]
            corpo = enc(" " + " ".join(s[chave].strip() for s in alvo))
            return pre, corpo + [self.tok.eot]
        pre = list(self.tok.sot_sequence)
        corpo: list[int] = []
        for s in alvo:
            corpo += [self._ts(s["ini"])] + enc(" " + s[chave].strip()) + [self._ts(s["fim"])]
        return pre, corpo + [self.tok.eot]

    def perda_roteiro(self, lote16, alvos: list[list[dict]], kappa=0.02, peso_longe=0.5, tau=4.0) -> Saida:
        """Puxa a transcrição para o texto falso de cada trecho e, como rede de
        segurança, empurra para longe do texto original (onde a troca não pegar, embaralha)."""
        mel = self.log_mel(lote16).to(self.dtype)
        feats_enc = self._encoder(mel)
        # cada passada do decodificador faz o backward na hora até 'feats' (uma por vez na memória);
        # no fim, uma perda substituta leva o gradiente acumulado pelo encoder.
        feats = feats_enc.detach().requires_grad_(True)
        dev = feats.device
        perdas, acertos = [], []
        n_pass = 4 * len(alvos)
        for b, alvo in enumerate(alvos):
            fb = feats[b: b + 1]
            pb = []
            for com_ts in (False, True):
                pre, y = self._seq_roteiro(alvo, "falso", com_ts)
                ent = torch.tensor(pre + y[:-1], device=dev)[None]
                lg = self.m.logits(ent, fb).float()[0, len(pre) - 1:]
                rot = torch.tensor(y, device=dev)
                ce = F.cross_entropy(lg, rot, reduction="none")
                l1 = torch.relu(ce - kappa).mean()
                acertos.append(float((lg.argmax(-1) == rot).float().mean()))
                (l1 / (2 * len(alvos))).backward()
                pb.append(float(l1.detach()))
                del lg, ce, l1
                # rede de segurança: longe do original
                pre_o, y_o = self._seq_roteiro(alvo, "original", com_ts)
                ent_o = torch.tensor(pre_o + y_o[:-1], device=dev)[None]
                lg_o = self.m.logits(ent_o, fb).float()[0, len(pre_o) - 1:-1]  # sem o eot
                ce_o = F.cross_entropy(lg_o, torch.tensor(y_o[:-1], device=dev), reduction="mean")
                l2 = peso_longe * torch.relu(tau - ce_o)
                (l2 / (2 * len(alvos))).backward()
                pb.append(float(l2.detach()))
                del lg_o, ce_o, l2
            perdas.append(sum(pb) / 2)
        acc = torch.tensor(acertos, device=dev).view(len(alvos), 2).mean(1)
        # perda substituta: d/d(feats_enc) == gradiente acumulado em feats
        substituta = (feats_enc.float() * feats.grad.float()).sum()
        return Saida(substituta, acc >= 0.9, valor=sum(perdas) / len(perdas))

    def transcrever(self, audio16) -> str:
        """Transcrição do original (referência para medir 'embaralhou?')."""
        import whisper

        r = whisper.transcribe(self.m, audio16, language="pt", fp16=self.dtype == torch.float16,
                               condition_on_previous_text=False, verbose=None)
        return r.get("text", "")

    def _embaralhar(self, feats: torch.Tensor, perda_eot: torch.Tensor, ok_eot: torch.Tensor) -> Saida:
        """Ataque não-direcionado: vê o que o modelo transcreve agora e empurra pra longe
        disso (até a perplexidade passar de tau). Converge bem mais rápido que silenciar."""
        from whisper.decoding import DecodingOptions, decode

        from .avaliacao import palavras_conteudo

        comp = getattr(self, "compartilhado", None)
        if comp is None or getattr(self, "decodifica", False) or "res" not in comp:
            with torch.no_grad():
                opts = DecodingOptions(task="transcribe", language="pt", without_timestamps=True,
                                       fp16=self.dtype == torch.float16, sample_len=96)
                res = decode(self.m, feats.detach(), opts)
            if not isinstance(res, list):
                res = [res]
            if comp is not None:
                comp["res"] = res
        else:
            res = comp["res"]  # leitura do modelo de decodificador rápido, no mesmo recorte
        perdas, oks = [], []
        for b, r in enumerate(res):
            y = [t for t in r.tokens if t < self.tok.eot][:96]
            pal = palavras_conteudo(r.text)
            # sucesso: o que sai quase não tem palavras da fala original
            if not pal:
                oks.append(True)
            else:
                prec = sum(1 for w in pal if w in self.ref_palavras) / len(pal) if self.ref_palavras else 1.0
                oks.append(prec < 0.3)
            if not y:
                continue
            ent = torch.tensor(self.prefixo_pt + y, device=feats.device)[None]
            lg = self.m.logits(ent[:, :-1], feats[b: b + 1]).float()[0, len(self.prefixo_pt) - 1:]
            ce = F.cross_entropy(lg, torch.tensor(y, device=feats.device), reduction="mean")
            perdas.append(torch.relu(self.tau - ce))
        nao_dir = torch.stack(perdas).mean() if perdas else feats.new_zeros(()).float()
        ok = torch.tensor(oks, device=feats.device) | ok_eot
        return Saida(nao_dir + self.peso_eot_embaralhar * perda_eot, ok)


# --------------------------------------------------------------------------- wav2vec2


class W2VSubstituto(Substituto):
    """CTC: 'silenciar' empurra todo quadro para o token vazio (blank)."""

    def __init__(self, nome: str, cfg, dtype: torch.dtype):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        repo = ALIAS_W2V.get(nome, nome)
        self.nome = nome
        self.proc = Wav2Vec2Processor.from_pretrained(repo)
        self.m = _congelar(Wav2Vec2ForCTC.from_pretrained(repo))
        self.blank = self.m.config.pad_token_id
        self.modo = cfg.modo
        texto = cfg.frase.strip().upper() if "coraa" not in repo.lower() else cfg.frase.strip().lower()
        ids = self.proc.tokenizer(texto).input_ids
        self.alvo = torch.tensor([i for i in ids if i != self.blank], dtype=torch.long)
        self.tamanho_gb = sum(p.numel() * p.element_size() for p in self.m.parameters()) / 1024**3
        self.dev = torch.device("cpu")

    def para(self, dev: torch.device) -> None:
        if self.dev == dev:
            return
        self.m.to(dev)
        self.alvo = self.alvo.to(dev)
        self.dev = dev

    def perda(self, lote16, comprimentos, kappa=0.02, peso_ns=0.0) -> Saida:
        perdas, oks = [], []
        for b, n in enumerate(comprimentos):
            x = lote16[b, :n].float()
            x = (x - x.mean()) / (x.std() + 1e-7)
            logits = self.m(x[None]).logits[0].float()  # (T, V)
            lp = logits.log_softmax(-1)
            if self.modo == "silenciar":
                perdas.append(torch.relu(-lp[:, self.blank] - kappa).mean())
                oks.append((logits.argmax(-1) == self.blank).all())
            else:
                T = lp.shape[0]
                l = F.ctc_loss(lp[:, None], self.alvo[None], torch.tensor([T]), torch.tensor([len(self.alvo)]),
                               blank=self.blank, reduction="mean", zero_infinity=True)
                perdas.append(l)
                pred = torch.unique_consecutive(logits.argmax(-1))
                pred = pred[pred != self.blank]
                igual = pred.numel() == self.alvo.numel() and bool((pred == self.alvo).all())
                oks.append(torch.tensor(igual, device=lote16.device))
        return Saida(torch.stack(perdas).mean(), torch.stack(oks).to(lote16.device))


def criar(nome: str, cfg, dtype: torch.dtype) -> Substituto:
    if nome.startswith("w2v") or "wav2vec" in nome:
        return W2VSubstituto(nome, cfg, dtype)
    return WhisperSubstituto(nome, cfg, dtype)


class Piscina:
    """Mantém na GPU só o que cabe; o resto espera na RAM (troca LRU)."""

    def __init__(self, modelos: list[Substituto], dev: torch.device, orcamento_gb: float):
        self.modelos = modelos
        self.dev = dev
        self.orcamento = orcamento_gb
        self.residentes: list[Substituto] = []

    @property
    def cabem_todos(self) -> bool:
        return sum(m.tamanho_gb for m in self.modelos) <= self.orcamento

    def garantir(self, m: Substituto) -> None:
        from . import dispositivo

        if m in self.residentes:
            self.residentes.remove(m)
            self.residentes.append(m)
            return
        while self.residentes and sum(r.tamanho_gb for r in self.residentes) + m.tamanho_gb > self.orcamento:
            velho = self.residentes.pop(0)
            velho.para(torch.device("cpu"))
            dispositivo.limpar(self.dev)
        m.para(self.dev)
        self.residentes.append(m)

    def liberar(self) -> None:
        from . import dispositivo

        for r in self.residentes:
            r.para(torch.device("cpu"))
        self.residentes.clear()
        dispositivo.limpar(self.dev)

    def descarregar(self) -> None:
        """Apaga os modelos de vez (GPU e RAM)."""
        from . import dispositivo

        self.residentes.clear()
        for m in self.modelos:
            if hasattr(m, "m"):
                del m.m
        self.modelos.clear()
        dispositivo.limpar(self.dev)
