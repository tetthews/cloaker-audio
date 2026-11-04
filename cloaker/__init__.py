"""Cloaker de áudio: protege a fala de vídeos contra transcrição automática por IA.

Gera uma perturbação adversarial quase inaudível, otimizada contra um conjunto de
modelos de reconhecimento de fala (família Whisper e, opcionalmente, wav2vec2), e
robusta à recompressão das redes sociais.
"""

__version__ = "0.2.0"
