"""Testes rápidos (sem baixar modelos): métricas, alinhamento de codec e máscara."""

import numpy as np
import torch

from cloaker import avaliacao, midia
from cloaker.psico import Mascara, metricas_qualidade


def test_recuperacao_e_normalizacao():
    ref = "Já conhece a Peixaria Central? Vem comigo que eu te mostro a loja."
    assert avaliacao.recuperacao(ref, ref) == 1.0
    assert avaliacao.recuperacao(ref, "") == 0.0
    assert 0 < avaliacao.recuperacao(ref, "conhece peixaria") < 1
    assert avaliacao.normalizar("Ação, Peixe!") == "acao peixe"


def test_ler_referencia_ass(tmp_path):
    p = tmp_path / "x.ass"
    p.write_text("[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,IG,,0,0,0,,{\\an5}Olá mundo\n", encoding="utf-8")
    assert avaliacao.ler_referencia(p) == "Olá mundo"


def test_alinhamento_codec():
    sr = 48000
    t = np.arange(sr * 3) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 1.3 * t) > 0)).astype(np.float32)
    atrasado = np.concatenate([np.zeros(1024, np.float32), x])
    al = midia._alinhar(x, atrasado)
    assert len(al) == len(x)
    assert np.abs(al - x).max() < 1e-5


def test_aac_ida_e_volta():
    codecs = midia.codecs_validos()
    if not codecs:
        return
    sr = 48000
    t = np.arange(sr * 2) / sr
    x = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    y = midia.ida_e_volta_codec(x, sr, codecs[0])
    assert y.shape == x.shape
    erro = np.mean((y - x) ** 2) / np.mean(x**2)
    assert erro < 0.05


def test_mascara_e_qualidade():
    sr = 16000
    x = torch.randn(sr * 2) * 0.1
    m = Mascara(x, sr, offset_db=8.0)
    d = torch.zeros_like(x)
    assert float(m.perda(d)) == 0.0
    d2 = torch.randn_like(x) * 0.5
    assert float(m.perda(d2)) > 0
    q = metricas_qualidade(x, d2 * 0.01, sr)
    assert q["snr_db"] > 20


def test_recuperacao_exclusiva():
    orig = "você ganha três mil reais trabalhando sozinha"
    falso = "você compra peixe azul descascado sozinha"
    assert avaliacao.recuperacao_exclusiva(orig, falso, falso) == 0.0
    assert avaliacao.recuperacao_exclusiva(falso, orig, falso) == 1.0
