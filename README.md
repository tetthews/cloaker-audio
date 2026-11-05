# Cloaker de áudio

> Escudo contra a cópia dos seus Reels por IA de transcrição. Windows e Mac.

Protege a fala dos seus vídeos contra **cópia em massa por transcrição automática**.
O programa adiciona ao áudio um ruído calculado (perturbação adversarial) que fica
escondido embaixo da sua voz. Pessoas ouvem o vídeo normalmente, mas transcritores de IA
da família Whisper (TurboScribe, faster-whisper, apps de legenda, scripts com
ChatGPT/Claude) passam a escrever **outro texto** no lugar do que você falou.

> É a mesma ideia do Glaze/Nightshade (imagens), aplicada a voz.

---

## Começo rápido (para quem vai usar)

### 1. Baixar

Baixe o pacote em **[Releases → cloaker-audio.zip](https://github.com/tetthews/cloaker-audio/releases/latest)**
(ou no botão verde **Code → Download ZIP**) e descompacte numa pasta fixa,
por exemplo `Documentos/cloaker-audio`. Não mude a pasta de lugar depois de instalar.

### 2. Verificar o computador

| Windows | Mac |
|---|---|
| dois cliques em **`1 - Verificar o computador (Windows).bat`** | botão direito em **`1 - Verificar o computador (Mac).command` → Abrir** |

Ele confere sistema, memória, disco, placa de vídeo, Python e internet e diz se dá pra
instalar. Não instala nada.

**O que precisa:**

| | Mínimo | Recomendado |
|---|---|---|
| Windows | 64 bits, 8 GB de RAM, 12 GB livres | placa **NVIDIA** com 6 GB+ (ex.: RTX 3060/4060), 16 GB de RAM, 20 GB livres |
| Mac | macOS 13, 8 GB de RAM, 12 GB livres | chip **M1/M2/M3/M4**, 16 GB de RAM |
| Tempo por vídeo de 1 min | horas (sem placa de vídeo) | ~25 min (NVIDIA, perfil padrão) · ~1 h a 1h30 (Mac M, perfil rápido, estimado) |

> **No Mac o programa usa sozinho o perfil `rapido`** (menos modelos e passos) e uma prova
> mais curta. Medido num MacBook Air M4 de 16 GB com o perfil padrão: **3h15** por vídeo de
> 1 min, por isso o rápido virou o padrão no Mac. Protege um pouco menos (no teste, 5% a 15%
> da fala recuperável contra 3% no padrão). Feche o navegador e outros programas pesados,
> deixe na tomada e com a tampa aberta.

### 3. Instalar

| Windows | Mac |
|---|---|
| dois cliques em **`2 - Instalar no Windows.bat`** | botão direito em **`2 - Instalar no Mac.command` → Abrir** |

São 6 passos guiados, de 15 a 40 minutos: verificação → Python → motor de IA (PyTorch) →
bibliotecas → modelos de IA (~5 GB) → atalho. No fim aparece **"Proteger Video"** na
Área de Trabalho (Windows) ou na Mesa (Mac). Não feche a janela até aparecer **INSTALADO**.

> **Mac bloqueou?** Na primeira vez o macOS não deixa abrir com duplo clique: use botão
> direito → Abrir. Se ainda bloquear: Ajustes do Sistema → Privacidade e Segurança →
> "Abrir Mesmo Assim".
>
> **Windows mostrou "O Windows protegeu o computador"?** Clique em "Mais informações" →
> "Executar assim mesmo".

### 4. Usar

1. **Arraste o vídeo** em cima do atalho **"Proteger Video"**
   (ou de `Proteger video (arraste aqui).bat`; no Mac, do app da Mesa ou de `proteger.command`).
2. Espere terminar. A janela mostra o progresso.
3. Ao lado do vídeo aparece a pasta **`NOME_protegido`**. Lá dentro:
   - **`NOME_protegido.mp4`** (ou `.mov`): **é esse que você posta**
   - `relatorio.html`: abre no navegador e mostra o que a IA consegue ler do original e do protegido

Não precisa escolher nada: o modo padrão (`automatico`) transcreve a sua fala e treina o
escudo pra IA escrever frases neutras do dia a dia no lugar dela.

### Instalar com o Claude Code (prompt pronto)

Se você usa o [Claude Code](https://claude.com/claude-code), cole isto nele:

```text
Instale pra mim o Cloaker de áudio: https://github.com/tetthews/cloaker-audio

1. Clone o repositório numa pasta fixa (Documentos/cloaker-audio no Mac,
   C:\cloaker-audio no Windows).
2. Rode a verificação do computador (no Windows:
   powershell -ExecutionPolicy Bypass -File scripts\verificar-windows.ps1;
   no Mac: bash scripts/verificar-mac.sh) e me explique o resultado.
   Se aparecer algum item FALTA, pare e me diga como resolver.
3. Instale seguindo o README. No Windows:
   powershell -ExecutionPolicy Bypass -File scripts\instalar-windows.ps1
   No Mac: siga os passos de "2 - Instalar no Mac.command" (Homebrew,
   python@3.12, ffmpeg, scripts/instalar-libs.sh, cloaker baixar).
   Me avise o que está fazendo em cada passo.
4. No fim rode "cloaker diagnostico" e me ensine a proteger um vídeo.

Depois disso, quando eu pedir "protege esse vídeo", rode o modo automatico.
Quando eu mandar uma copy para a IA ler no lugar da fala, use o modo roteiro
do README com --copy (palavras exatas da minha copy, sem reescrever).
Quando eu pedir "nesse vídeo quero que a IA leia um texto sobre X", gere os
trechos e escreva o texto falso de cada um sobre X (mesmo tamanho e
estrutura de frase).
Nos dois casos: me mostre a tabela FALA x FALSO trecho a trecho e só rode a
proteção depois do meu ok. Se a copy for longa demais, me diga quais frases
sugere cortar; não corte nem reescreva sem me perguntar.
```

---

## Como funciona (em português simples)

Os concorrentes baixam o seu Reels e jogam numa IA de transcrição pra copiar o que você
fala. O programa adiciona ao áudio um **ruído calculado** que fica escondido embaixo da
sua voz: as pessoas ouvem normalmente, mas a IA "ouve" outra coisa e escreve um texto sem
relação com o seu.

O ruído é treinado contra várias IAs ao mesmo tempo, passando pela compressão do
Instagram a cada passo do treino, pra sobreviver ao upload. No teste de referência (vídeo
de 62 s, após simular o upload), a IA lia **97%** das palavras importantes do original e
só **3%** do protegido, inclusive numa IA que o escudo nunca viu no treino.

---

## Comandos (terminal)

Windows: `cloaker.bat COMANDO` · Mac: `./cloaker.sh COMANDO` (dentro da pasta).

| O quê | Comando |
|---|---|
| Proteger um vídeo | `cloaker proteger video.mp4` |
| Pasta inteira | `cloaker lote PASTA` |
| Texto falso escolhido | `cloaker roteiro video.mp4 --copy copy.txt` e depois `cloaker proteger video.mp4 --modo roteiro --roteiro video_roteiro.json` |
| Variações de criativo | `cloaker variacoes video.mp4 -n 5 --proteger` |
| Só medir um protegido | `cloaker avaliar original.mp4 protegido.mp4` |
| Baixar os modelos | `cloaker baixar` |
| Checar a instalação | `cloaker diagnostico` |

O resultado sai numa pasta `NOME_protegido/` ao lado do vídeo:

- `NOME_protegido.mp4`: **é esse que você posta** (o vídeo não é recomprimido, só o áudio troca)
- `relatorio.html`: o que a IA consegue tirar antes e depois, com players de áudio
- `ruido_x10.wav`: só o ruído adicionado, 10× mais alto, pra você saber o que foi colocado
- `relatorio.json`, `audio_protegido.wav`, `perturbacao.npy`, `roteiro_automatico.json`

### Opções principais

| Opção | Valores | O que faz |
|---|---|---|
| `--modo` | `automatico` (padrão) · `roteiro` · `embaralhar` · `silenciar` · `trocar` | ver "Modos" |
| `--forca` | `leve` · `media` (padrão) · `forte` · `maxima` | teto do ruído. `media` = discreto (aprovado de ouvido); `leve` ainda mais discreto e protege menos; `forte`/`maxima` audíveis |
| `--perfil` | `rapido` · `padrao` · `maximo` | quantos modelos e passos. Sem a opção: `padrao` com NVIDIA, `rapido` no Mac/CPU. `maximo` inclui large-v3 e wav2vec2 |
| `--sem-avaliacao` | | pula a prova final (uns 5 min mais rápido) |
| `--referencia` | `.ass` `.srt` `.txt` | o texto real do vídeo, pra medir com precisão |
| `--frase` | texto | frase isca do modo `trocar` |

## Modos

| Modo | O que a IA do copiador recebe | Resultado medido (vídeo de 62 s, após upload 128k) |
|---|---|---|
| `automatico` (**padrão**) | frases neutras do dia a dia no lugar de cada trecho; o texto falso é gerado sozinho | mesma técnica do `roteiro` |
| `roteiro` | **o texto falso que você escolher**, trecho a trecho; onde não pegar, embaralha | fala recuperável 97% → **3%** (IA nunca vista: 3%); ~28% das palavras vêm do texto falso |
| `embaralhar` | lixo sem relação com a fala | 98% → 30% com o ruído discreto (mais fraco) |
| `silenciar` | nada (ou fragmentos) | não fecha nos modelos grandes |
| `trocar` | uma frase fixa | experimental |

### Modo roteiro (você escolhe o texto falso)

**Com a sua copy pronta** (as palavras exatas, sem reescrever):

1. Salve a copy num `copy.txt` e rode `cloaker roteiro video.mp4 --copy copy.txt`.
   Ele divide a fala em trechos, espalha a copy por eles e mostra FALA × FALSO de cada trecho.
2. **Confira a tabela antes de proteger.** Se aparecer "ATENÇÃO: a copy tem ... (1.3x)", a copy
   é mais longa que a fala: corte frases inteiras até o tamanho sugerido e rode o passo 1 de novo.
   Copy longa demais = mais palavras por segundo = a IA lê menos dela.
3. `cloaker proteger video.mp4 --modo roteiro --roteiro video_roteiro.json`

**Sem copy pronta:** `cloaker roteiro video.mp4` e preencha o campo `falso` de cada trecho no
`video_roteiro.json` (mesma estrutura de frase, tamanho parecido, outras informações), ou peça
ao Claude: "nesse vídeo quero que a IA leia um texto sobre X".

> **Regra para quem opera (pessoa ou Claude):** mostre o texto falso trecho a trecho e espere o
> ok antes de proteger. Nunca reescreva a copy do cliente sem mostrar. Cada rodada leva de
> 20 min (NVIDIA) a mais de 1 h (Mac).

Expectativa realista: a IA não lê a copy inteira palavra por palavra. Nos testes, de 28% a 36%
das palavras transcritas vêm do texto falso; o resto sai embaralhado. O que importa é que a
fala original some (3% a 15% recuperável).

Quanto mais o texto falso "soa" parecido com o verdadeiro (troca números, nomes, passos, oferta),
mais fácil a IA cair. Onde a troca não pega, o treino ainda empurra a IA pra longe do original.

## Variações para teste de criativo

```
cloaker variacoes video.mp4 -n 5 --intensidade leve   # só as variações
cloaker variacoes video.mp4 -n 5 --proteger           # variações já protegidas
```

Cada variação recebe mudanças pequenas e sorteadas (recorte 1–2%, deslocamento do enquadramento,
saturação, brilho, contraste, matiz, velocidade ±1,5%, grão, volume) e **limpeza total de metadados**
(data de criação, aparelho, localização, encoder e a assinatura SEI do x264). Os parâmetros de cada
uma ficam em `variacoes.json`. `--espelhar` permite espelhar (cuidado com texto na tela).

**Ordem certa:** variações primeiro, proteção depois (`--proteger` já faz assim). Mudar velocidade
ou reprocessar o áudio de um vídeo já protegido enfraquece a proteção.

## Como funciona (técnico)

1. **Ataque em conjunto.** O ruído é otimizado contra vários modelos ao mesmo tempo
   (Whisper small, medium, large-v3-turbo; no `maximo` também large-v3 e wav2vec2 PT).
   Nos modos `automatico`/`roteiro` o alvo é cada trecho virar o texto falso, no tempo
   certo (com timestamps), e ao mesmo tempo se afastar do original.
2. **Robustez ao mundo real (EOT).** Em cada passo o áudio passa por um codec real
   (AAC 64–160k, MP3, Opus/WhatsApp, via ffmpeg), reamostragem 48k→44,1k→16k,
   ganho aleatório, ruído e recortes. Assim o ruído sobrevive ao upload e ao download.
3. **Esconder do ouvido.** Um modelo psicoacústico (mascaramento na escala Bark +
   limiar absoluto de audição) calcula quanto ruído cabe embaixo de cada som em cada
   instante. A cada passo o ruído é **cortado** onde passa desse teto (que começa folgado
   e aperta até o fim) e tudo acima de 8 kHz é zerado (o Whisper não escuta lá; o ouvido sim).
4. **Prova honesta.** O relatório transcreve o original e o protegido, também depois de
   simular o upload (AAC 128k e 64k), com modelos que o escudo viu e com um que ele
   **nunca viu** (large-v3). A coluna "nunca visto" estima o efeito num copiador que usa
   outra ferramenta.

## Limites (leia)

- **Não existe proteção 100%.** Isso encarece a cópia em massa; não impede quem senta e ouve.
- **Funciona melhor contra a família Whisper**, que é a base da maioria das ferramentas.
  Transcritores de outras empresas (Google/Gemini, CapCut, AssemblyAI) podem não ser afetados.
- **Legenda queimada na tela vaza tudo**: um OCR lê o texto direto da imagem. Poste o vídeo
  protegido **sem** legenda escrita da fala, ou só com palavras-chave.
- **Legenda automática do Instagram** também usa IA, então pode piorar ou sumir no vídeo
  protegido, e isso pode afetar busca e alcance. Teste com alguns vídeos antes de aplicar em tudo.
- O escudo é de cada vídeo: precisa rodar em cada um (por isso existe o `lote`).
- Na força `media` ~97% do ruído fica abaixo do limiar de mascaramento. O AAC do Instagram
  também descarta o que o ouvido não percebe; por isso a prova sempre mede após simular o upload.
- Outros programas usando a placa de vídeo ou muita memória (navegador com muitas abas, jogos)
  deixam a proteção lenta. Feche-os enquanto protege.

## Estrutura

```
1 - Verificar o computador (Windows|Mac)   confere se o computador roda
2 - Instalar no Windows|Mac                 instalador passo a passo
Proteger video (arraste aqui).bat           uso no Windows · proteger.command no Mac
cloaker.bat / cloaker.sh                    linha de comando
scripts/                                    verificação e instaladores
cloaker/
  cli.py            comandos
  config.py         presets de força e perfil
  roteiro_auto.py   modo automatico: trechos + texto falso neutro
  ataque.py         loop de otimização (Adam + L∞ + teto psicoacústico)
  substitutos.py    Whisper/wav2vec2 diferenciáveis, alvos, troca de modelos na GPU
  transformacoes.py codec real com gradiente (BPDA), reamostragem, recortes
  psico.py          modelo psicoacústico e métricas de qualidade
  avaliacao.py      transcrição com faster-whisper e métricas de vazamento
  relatorio.py      relatório HTML
  midia.py          ffmpeg (Windows, Mac e Linux)
  variacoes.py      variações de criativo (crop, cor, velocidade, metadados)
empacotar.py        gera dist/cloaker-audio.zip (preserva permissão de execução pro Mac)
```
