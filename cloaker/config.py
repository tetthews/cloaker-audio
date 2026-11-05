"""Presets de força (quanto ruído é permitido) e de perfil (quanto esforço gastar)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

# eps: limite absoluto da perturbação por amostra (escala -1..1).
# offset_db: quanto abaixo do limiar de mascaramento o ruído deve ficar (maior = mais discreto).
# alpha0: peso inicial da perda perceptual (é ajustado sozinho durante a otimização).
# margem_db: teto DURO do ruído em relação ao limiar de mascaramento (projeção a cada passo).
#   maior = mais discreto. None = sem teto duro (só a penalidade perceptual).
FORCAS: dict[str, dict] = {
    "leve": dict(eps=0.006, offset_db=12.0, alpha0=0.10, margem_db=6.0),
    "media": dict(eps=0.012, offset_db=8.0, alpha0=0.05, margem_db=0.0),
    "forte": dict(eps=0.025, offset_db=4.0, alpha0=0.02, margem_db=-6.0),
    "maxima": dict(eps=0.05, offset_db=0.0, alpha0=0.01, margem_db=None),
}

# modelos: substitutos atacados (o escudo é treinado contra eles).
# avaliadores: modelos usados na prova final. Os que não estão em "modelos" são
# o teste honesto de transferência: o escudo nunca os viu.
PERFIS: dict[str, dict] = {
    "rapido": dict(
        modelos=["base", "small", "turbo"],
        passos=400,
        lote=1,
        avaliadores=["large-v3-turbo", "large-v3"],
    ),
    "padrao": dict(
        modelos=["small", "medium", "turbo"],
        passos=800,
        lote=1,
        avaliadores=["small", "medium", "large-v3-turbo", "large-v3"],
    ),
    "maximo": dict(
        modelos=["small", "medium", "turbo", "large-v3", "w2v-pt"],
        passos=1600,
        lote=2,
        avaliadores=[
            "small",
            "medium",
            "large-v3-turbo",
            "large-v3",
            "w2v-coraa",
        ],
    ),
}

MODOS = ("automatico", "roteiro", "embaralhar", "silenciar", "trocar")

FRASE_PADRAO = "Este conteúdo é protegido e não pode ser copiado."


@dataclass
class Config:
    modo: str = "automatico"
    frase: str = FRASE_PADRAO
    roteiro: str = ""  # modo roteiro: JSON com os trechos e o texto falso de cada um
    forca: str = "media"
    perfil: str = "padrao"
    modelos: list[str] = field(default_factory=list)
    avaliadores: list[str] = field(default_factory=list)
    passos: int = 0
    lote: int = 0
    idiomas: list[str] = field(default_factory=lambda: ["pt"])
    eps: float = 0.0
    offset_db: float = 0.0
    alpha0: float = 0.0
    margem_db: float | None = None
    folga_inicial_db: float = 12.0  # o teto começa mais solto e aperta até margem_db
    # robustez
    prob_codec: float = 0.7
    codec_a_cada: int = 8
    prob_vad: float = 0.3
    prob_44k: float = 0.4
    prob_ruido: float = 0.3
    ganho_db: tuple[float, float] = (-6.0, 3.0)
    peso_nospeech: float = 0.05
    lr_rel: float = 0.2  # passo do Adam em fração do eps
    ensemble: str = "soma"  # soma (todos por passo) ou alterna
    kappa: float = 0.02  # para de empurrar um token quando p >= ~0.98
    semente: int = 1234
    dispositivo: str = "auto"
    memoria_modelos_gb: float = 0.0  # 0 = automático

    @classmethod
    def montar(cls, forca: str, perfil: str, **sobrescritas) -> "Config":
        if forca not in FORCAS:
            raise ValueError(f"força inválida: {forca} (use {', '.join(FORCAS)})")
        if perfil not in PERFIS:
            raise ValueError(f"perfil inválido: {perfil} (use {', '.join(PERFIS)})")
        base = dict(FORCAS[forca])
        p = PERFIS[perfil]
        cfg = cls(
            forca=forca,
            perfil=perfil,
            modelos=list(p["modelos"]),
            avaliadores=list(p["avaliadores"]),
            passos=p["passos"],
            lote=p["lote"],
            **base,
        )
        for k, v in sobrescritas.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 0:
                continue
            if not hasattr(cfg, k):
                raise ValueError(f"opção desconhecida: {k}")
            setattr(cfg, k, v)
        if cfg.modo not in MODOS:
            raise ValueError(f"modo inválido: {cfg.modo} (use {', '.join(MODOS)})")
        return cfg

    def como_dict(self) -> dict:
        return asdict(self)
