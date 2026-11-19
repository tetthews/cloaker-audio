"""Motor MLX (opcional) para os substitutos Whisper.

Faz o trabalho pesado (log-mel, encoder, decodificador e o gradiente da perda em
relação ao áudio) com o MLX da Apple, em vez do PyTorch MPS. A interface é a mesma da
`WhisperSubstituto`: o ataque.py não muda. A ponte com o torch é uma perda substituta:
`(folha * g).sum()`, cujo `.backward()` deposita exatamente `g` (o gradiente calculado
pelo MLX) em `folha.grad`.

Escolha do motor: variável de ambiente CLOAKER_MOTOR=mlx (padrão: torch).
Suporta os modos roteiro/automático (`perda_roteiro`) e silenciar/trocar (`perda`).
O modo embaralhar não é suportado: `criar()` usa a versão torch nele.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from .substitutos import N_AMOSTRAS, Saida, Substituto

_DIRETORIO_PESOS = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"

# --------------------------------------------------------------------------- pesos


def _caminho_pesos(nome: str) -> str:
    """Mesmo arquivo .pt que o openai-whisper usa (baixa se ainda não estiver no cache)."""
    import whisper

    if os.path.isfile(nome):
        return nome
    url = whisper._MODELS.get(nome)
    if url is None:
        raise RuntimeError(f"modelo Whisper desconhecido: {nome}")
    arq = _DIRETORIO_PESOS / os.path.basename(url)
    if arq.is_file():
        return str(arq)
    whisper._download(url, str(_DIRETORIO_PESOS), False)
    return str(arq)


def carregar_pesos(nome: str, dtype: mx.Dtype) -> tuple[dict, dict]:
    """Lê o .pt do openai-whisper e devolve (dims, pesos em MLX).

    Conversão: nomes de camada ficam iguais (blocks.N.attn.query.weight etc.);
    convoluções vão de (saída, entrada, k) do torch para (saída, k, entrada) do MLX;
    LayerNorm fica em fp32 (o Whisper faz a conta em fp32), o resto em `dtype`."""
    ck = torch.load(_caminho_pesos(nome), map_location="cpu", weights_only=True)
    dims = ck["dims"]
    pesos: dict[str, mx.array] = {}
    for k, v in ck["model_state_dict"].items():
        if k.startswith("decoder.mask") or k.endswith("alignment_heads"):
            continue
        a = v.float().numpy()
        if k.endswith(("conv1.weight", "conv2.weight")):
            a = np.ascontiguousarray(a.transpose(0, 2, 1))
        ln = (".attn_ln." in k or ".cross_attn_ln." in k or ".mlp_ln." in k
              or k.startswith(("encoder.ln_post.", "decoder.ln.")))
        pesos[k] = mx.array(a).astype(mx.float32 if ln else dtype)
    if "encoder.positional_embedding" not in pesos:
        pesos["encoder.positional_embedding"] = mx.array(_sinusoides(dims["n_audio_ctx"], dims["n_audio_state"])).astype(dtype)
    return dims, pesos


def _sinusoides(comprimento: int, canais: int, max_scale: float = 10000.0) -> np.ndarray:
    inc = np.log(max_scale) / (canais // 2 - 1)
    inv = np.exp(-inc * np.arange(canais // 2))
    t = np.arange(comprimento)[:, None] * inv[None, :]
    return np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- arquitetura


def _ln(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    return mx.fast.layer_norm(x.astype(mx.float32), w, b, 1e-5).astype(x.dtype)


def _lin(x: mx.array, p: dict, pre: str, bias: bool = True) -> mx.array:
    y = x @ p[pre + ".weight"].T
    return y + p[pre + ".bias"] if bias else y


class _Whisper:
    """Whisper (encoder + decodificador) em MLX, só o que o ataque usa."""

    def __init__(self, dims: dict, pesos: dict):
        self.dims = type("Dims", (), dims)()  # acesso por atributo, como no openai-whisper
        self.p = pesos
        self.n_head_a = dims["n_audio_head"]
        self.n_head_t = dims["n_text_head"]

    def _atencao(self, q, k, v, n_head, mask=None):
        B, Lq, D = q.shape
        Lk = k.shape[1]
        hd = D // n_head
        q = q.reshape(B, Lq, n_head, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, Lk, n_head, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, Lk, n_head, hd).transpose(0, 2, 1, 3)
        if Lq <= 8 or mask is not None:
            # sequências curtas do decodificador: o kernel rápido do MLX não tem gradiente
            # para Lq pequeno, e aqui a conta manual custa quase nada (softmax em fp32, como o Whisper)
            w = (q * hd ** -0.25) @ (k * hd ** -0.25).transpose(0, 1, 3, 2)
            w = w.astype(mx.float32)
            if mask is not None:
                w = w + mask.astype(mx.float32)
            o = mx.softmax(w, axis=-1).astype(v.dtype) @ v
        else:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask=mask)
        return o.transpose(0, 2, 1, 3).reshape(B, Lq, D)

    def _bloco_enc(self, i: int, x: mx.array) -> mx.array:
        """x chega em fp32 (resíduo em precisão cheia); as contas pesadas vão em `dt`."""
        p = self.p
        dt = p[f"encoder.blocks.{i}.attn.query.weight"].dtype
        pre = f"encoder.blocks.{i}"
        h = _ln(x, p[pre + ".attn_ln.weight"], p[pre + ".attn_ln.bias"]).astype(dt)
        q = _lin(h, p, pre + ".attn.query")
        k = _lin(h, p, pre + ".attn.key", bias=False)
        v = _lin(h, p, pre + ".attn.value")
        x = x + _lin(self._atencao(q, k, v, self.n_head_a), p, pre + ".attn.out").astype(mx.float32)
        h = _ln(x, p[pre + ".mlp_ln.weight"], p[pre + ".mlp_ln.bias"]).astype(dt)
        h = _lin(_gelu(_lin(h, p, pre + ".mlp.0")), p, pre + ".mlp.2")
        return x + h.astype(mx.float32)

    def encoder(self, mel: mx.array) -> mx.array:
        """mel: (B, n_mels, 3000) -> (B, 1500, d). Um checkpoint por bloco (menos memória)."""
        p = self.p
        x = mx.transpose(mel, (0, 2, 1))  # (B, 3000, n_mels): formato do MLX
        x = _gelu(mx.conv1d(x, p["encoder.conv1.weight"], stride=1, padding=1) + p["encoder.conv1.bias"])
        x = _gelu(mx.conv1d(x, p["encoder.conv2.weight"], stride=2, padding=1) + p["encoder.conv2.bias"])
        x = (x + p["encoder.positional_embedding"]).astype(mx.float32)
        for i in range(self.dims.n_audio_layer):
            x = mx.checkpoint(lambda t, i=i: self._bloco_enc(i, t))(x)
        return _ln(x, p["encoder.ln_post.weight"], p["encoder.ln_post.bias"]).astype(p["encoder.conv1.weight"].dtype)

    def kv_cruzado(self, feats: mx.array) -> list[tuple[mx.array, mx.array]]:
        """Chaves e valores da atenção cruzada, uma vez por recorte (reaproveitados nas passadas)."""
        p = self.p
        return [(_lin(feats, p, f"decoder.blocks.{i}.cross_attn.key", bias=False),
                 _lin(feats, p, f"decoder.blocks.{i}.cross_attn.value"))
                for i in range(self.dims.n_text_layer)]

    def logits(self, toks: mx.array, kv: list[tuple[mx.array, mx.array]]) -> mx.array:
        """toks: (B, L) int; kv de feats com mesmo B (ou B=1, que broadcast). -> (B, L, V) em fp32.
        O resíduo fica em fp32; as multiplicações de matriz vão no tipo dos pesos."""
        p = self.p
        dt = p["decoder.token_embedding.weight"].dtype
        L = toks.shape[1]
        x = (p["decoder.token_embedding.weight"][toks] + p["decoder.positional_embedding"][:L]).astype(mx.float32)
        mask = mx.where(mx.arange(L)[:, None] >= mx.arange(L)[None, :], 0.0, -mx.inf).astype(mx.float32)
        for i in range(self.dims.n_text_layer):
            pre = f"decoder.blocks.{i}"
            h = _ln(x, p[pre + ".attn_ln.weight"], p[pre + ".attn_ln.bias"]).astype(dt)
            q = _lin(h, p, pre + ".attn.query")
            k = _lin(h, p, pre + ".attn.key", bias=False)
            v = _lin(h, p, pre + ".attn.value")
            x = x + _lin(self._atencao(q, k, v, self.n_head_t, mask), p, pre + ".attn.out").astype(mx.float32)
            h = _ln(x, p[pre + ".cross_attn_ln.weight"], p[pre + ".cross_attn_ln.bias"]).astype(dt)
            q = _lin(h, p, pre + ".cross_attn.query")
            ck, cv = kv[i]
            x = x + _lin(self._atencao(q, ck, cv, self.n_head_t), p, pre + ".cross_attn.out").astype(mx.float32)
            h = _ln(x, p[pre + ".mlp_ln.weight"], p[pre + ".mlp_ln.bias"]).astype(dt)
            x = x + _lin(_gelu(_lin(h, p, pre + ".mlp.0")), p, pre + ".mlp.2").astype(mx.float32)
        x = _ln(x, p["decoder.ln.weight"], p["decoder.ln.bias"]).astype(dt)
        return (x @ p["decoder.token_embedding.weight"].T).astype(mx.float32)


def _gelu(x: mx.array) -> mx.array:
    # F.gelu do torch (versão exata, com erf)
    return x * 0.5 * (1.0 + mx.erf(x * (2.0 ** -0.5)))


# --------------------------------------------------------------------------- log-mel


def _tabelas_stft() -> tuple[np.ndarray, np.ndarray]:
    """DFT como multiplicação de matrizes (janela de Hann periódica de 400 já embutida):
    equivale a torch.stft(n_fft=400) e é exata e diferenciável em fp32."""
    n = np.arange(400)
    janela = 0.5 - 0.5 * np.cos(2 * np.pi * n / 400)  # torch.hann_window(400), periódica
    f = np.arange(201)
    ang = 2 * np.pi * n[:, None] * f[None, :] / 400
    return (janela[:, None] * np.cos(ang)).astype(np.float32), (-janela[:, None] * np.sin(ang)).astype(np.float32)


# --------------------------------------------------------------------------- substituto


class MLXWhisperSubstituto(Substituto):
    def __init__(self, nome: str, cfg, dtype: torch.dtype):
        import whisper
        from whisper.audio import mel_filters
        from whisper.tokenizer import get_tokenizer

        self.nome = nome
        self.dtype = dtype
        self._mxdt = mx.float16 if dtype == torch.float16 else mx.float32
        if cfg.modo == "embaralhar":
            raise RuntimeError("o motor MLX não suporta o modo embaralhar")
        dims, pesos = carregar_pesos(nome, self._mxdt)
        self.m = _Whisper(dims, pesos)
        self.n_mels = dims["n_mels"]
        self.tamanho_gb = sum(v.nbytes for v in pesos.values()) / 1024**3
        self.modo = cfg.modo
        self.dev = torch.device("cpu")  # só para a interface; o MLX decide sozinho onde roda
        self.ref_palavras: set[str] = set()

        self._filtros = mx.array(mel_filters(torch.device("cpu"), self.n_mels).numpy())  # (n_mels, 201)
        cos_, sin_ = _tabelas_stft()
        self._dft_cos, self._dft_sin = mx.array(cos_), mx.array(sin_)

        multi = dims["n_vocab"] >= 51865
        n_lang = dims["n_vocab"] - 51765 - int(multi)
        tok0 = get_tokenizer(multi, num_languages=n_lang, language="pt", task="transcribe")
        self.tok = tok0
        self.no_speech = tok0.no_speech
        self.variantes: list[tuple[list[int], list[int], int]] = []
        for lang in cfg.idiomas:
            tok = get_tokenizer(multi, num_languages=n_lang, language=lang, task="transcribe")
            sot_seq = list(tok.sot_sequence)
            texto: list[int] = [] if cfg.modo in ("silenciar", "embaralhar") else tok.encode(" " + cfg.frase.strip())
            self._add(sot_seq + [tok.no_timestamps], texto + [tok.eot])
            ts0 = tok.timestamp_begin
            if cfg.modo in ("silenciar", "embaralhar"):
                alvo_ts = [ts0, tok.eot]
            else:
                alvo_ts = [ts0] + texto + [tok.timestamp_begin + 150, tok.eot]
            self._add(sot_seq, alvo_ts)

    def _add(self, prefixo: list[int], alvo: list[int]) -> None:
        seq = prefixo + alvo
        self.variantes.append((seq[:-1], alvo, len(prefixo) - 1))

    def para(self, dev: torch.device) -> None:
        """No MLX a memória é unificada: não há o que mover."""
        return

    # ----- áudio -> log-mel (idêntico ao do torch)

    def log_mel(self, x: mx.array) -> mx.array:
        """x: (B, N) float32 a 16 kHz -> (B, n_mels, N/160) com a normalização do Whisper."""
        N = x.shape[1]
        # center=True, pad_mode="reflect": 200 amostras de cada lado
        esq = mx.arange(200, 0, -1)
        dir_ = mx.arange(N - 2, N - 202, -1)
        xp = mx.concatenate([x[:, esq], x, x[:, dir_]], axis=1)
        n_q = N // 160  # 3001 quadros menos o último, que o Whisper descarta
        idx = mx.arange(n_q)[:, None] * 160 + mx.arange(400)[None, :]
        quadros = xp[:, idx]  # (B, n_q, 400)
        re = quadros @ self._dft_cos
        im = quadros @ self._dft_sin
        mag = (re * re + im * im).transpose(0, 2, 1)  # (B, 201, n_q)
        mel = self._filtros @ mag
        log = mx.log10(mx.maximum(mel, 1e-10))
        topo = mx.max(log, axis=(-2, -1), keepdims=True)
        log = mx.maximum(log, topo - 8.0)
        return (log + 4.0) / 4.0

    # ----- perdas

    @staticmethod
    def _ce(lg: mx.array, rot: mx.array) -> mx.array:
        """cross entropy por posição; lg (L, V) em fp32, rot (L,)."""
        lg = lg.astype(mx.float32)
        return mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, rot[:, None], axis=-1)[:, 0]

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

    def _preparar(self, lote16: torch.Tensor) -> mx.array:
        return mx.array(lote16.detach().float().cpu().numpy())

    def _saida(self, folha: torch.Tensor, g: mx.array, sucesso, valor: float) -> Saida:
        """Perda substituta: seu backward deposita exatamente g (gradiente do MLX) em folha.grad."""
        mx.eval(g)
        gt = torch.from_numpy(np.array(g, copy=False).astype(np.float32)).to(folha.device)
        subst = (folha.float() * gt).sum()
        ok = torch.tensor([bool(s) for s in sucesso], dtype=torch.bool, device=folha.device)
        return Saida(subst, ok, valor=valor)

    def perda_roteiro(self, lote16, alvos: list[list[dict]], kappa=0.02, peso_longe=0.5, tau=4.0) -> Saida:
        """Mesma perda da versão torch: puxa para o texto falso de cada trecho e, como rede de
        segurança, empurra para longe do original. Tudo (valor e gradiente no áudio) no MLX."""
        x = self._preparar(lote16)
        nb = len(alvos)
        # sequências (inteiros, fora do grafo): por recorte, 2 variantes x (falso, original)
        seqs = []
        for alvo in alvos:
            por = []
            for com_ts in (False, True):
                pre, y = self._seq_roteiro(alvo, "falso", com_ts)
                pre_o, y_o = self._seq_roteiro(alvo, "original", com_ts)
                por.append((mx.array(pre + y[:-1])[None], mx.array(y), len(pre) - 1,
                            mx.array(pre_o + y_o[:-1])[None], mx.array(y_o[:-1]), len(pre_o) - 1))
            seqs.append(por)
        info: dict = {}

        def f(audio):
            feats = self.m.encoder(self.log_mel(audio).astype(self._mxdt))
            total = mx.array(0.0)
            acertos, valores = [], []
            for b in range(nb):
                kv = self.m.kv_cruzado(feats[b: b + 1])
                pb = []
                for ent, rot, ini, ent_o, rot_o, ini_o in seqs[b]:
                    lg = self.m.logits(ent, kv)[0, ini:]
                    ce = self._ce(lg, rot)
                    l1 = mx.mean(mx.maximum(ce - kappa, 0.0))
                    acertos.append(mx.mean((mx.argmax(lg, axis=-1) == rot).astype(mx.float32)))
                    lg_o = self.m.logits(ent_o, kv)[0, ini_o:-1]
                    ce_o = mx.mean(self._ce(lg_o, rot_o))
                    l2 = peso_longe * mx.maximum(tau - ce_o, 0.0)
                    pb += [l1, l2]
                    total = total + (l1 + l2) / (2 * nb)
                valores.append(sum(pb) / 2)
            info["acertos"] = mx.stack(acertos)
            info["valor"] = mx.mean(mx.stack(valores))
            return total

        total, g = mx.value_and_grad(f)(x)
        mx.eval(total, g, info["acertos"], info["valor"])
        acc = np.array(info["acertos"]).reshape(nb, 2).mean(1)
        return self._saida(lote16, g, acc >= 0.9, float(info["valor"]))

    def perda(self, lote16, comprimentos, kappa=0.02, peso_ns=0.3) -> Saida:
        """Modos silenciar/trocar: mesma perda da versão torch (teacher forcing em cada variante)."""
        x = self._preparar(lote16)
        B = x.shape[0]
        info: dict = {}

        def f(audio):
            feats = self.m.encoder(self.log_mel(audio).astype(self._mxdt))
            kv = self.m.kv_cruzado(feats)
            total = mx.array(0.0)
            ok = mx.ones((B,), dtype=mx.bool_)
            for k, (entrada, rotulos, ini) in enumerate(self.variantes):
                toks = mx.broadcast_to(mx.array(entrada)[None], (B, len(entrada)))
                rot = mx.array(rotulos)
                logits = self.m.logits(toks, kv).astype(mx.float32)  # (B, L, V)
                alvo_lg = logits[:, ini:]
                ce = mx.logsumexp(alvo_lg, axis=-1) - mx.take_along_axis(
                    alvo_lg, mx.broadcast_to(rot[None, :, None], (B, rot.shape[0], 1)), axis=-1)[..., 0]
                total = total + mx.mean(mx.maximum(ce - kappa, 0.0))
                ok = ok & mx.all(mx.argmax(alvo_lg, axis=-1) == rot[None], axis=-1)
                if k == 0 and peso_ns > 0:
                    l0 = logits[:, 0]
                    ns = mx.logsumexp(l0, axis=-1) - l0[:, self.no_speech]
                    total = total + peso_ns * mx.mean(mx.maximum(ns - 0.1, 0.0))
            total = total / len(self.variantes)
            info["ok"] = ok
            return total

        total, g = mx.value_and_grad(f)(x)
        mx.eval(total, g, info["ok"])
        return self._saida(lote16, g, np.array(info["ok"]), float(total))
