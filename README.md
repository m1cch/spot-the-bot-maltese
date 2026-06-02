# Spot the bot — Maltese

Курсовая работа: «Естественный язык как целое». Целевой язык — **мальтийский (mt)**. 
Лаборатория проекта: В. А. Громов, ВШЭ.

## Пайплайн

1. **Сбор корпуса** мальтийского текста (`corpus_raw/`):
   - Wikipedia mt (быстрый старт)
   - Korpus Malti v4 / MLRS — академический корпус UM
   - Bloomlibrary, archive.org, libgen — литература
   - На выходе: один `.txt` на текст, имя файла = название.

2. **Препроцессинг** (`scripts/preprocess.py` → `corpus_clean/`):
   - Лемматизация (Stanza maltese / UDPipe)
   - Замена местоимений (`PRON1`), числительных (`ORDINAL1`), имён собственных (`PERSON1`)
   - На выходе: единый файл, строка = очищенный текст.

3. **SVD-словарь** (`scripts/build_svd.py` → `embeddings/svd/`):
   - `TfidfVectorizer` с `token_pattern` из мальтийского алфавита.
   - SVD-разложение, ранг `k=1024`.
   - Словарь `{слово: вектор[1024]}` → `np.save`.

4. **Альтернативные эмбеддинги** (`embeddings/fasttext/`, `embeddings/word2vec/`):
   - fastText `cc.mt.300.bin` (готовый).
   - gensim CBOW / skipgram на нашем корпусе.

5. **Бот-корпус** (`scripts/train_lstm.py`, `corpus_bot/`):
   - LSTM по `TextGen.ipynb` (А. Ахметов).
   - Сгенерировать столько же текстов, сколько у человека.

6. **Датасет n-грамм** (`scripts/build_dataset.py` → `datasets/`):
   - Скользящее окно по обоим корпусам, конкатенация векторов слов.
   - Вектора размера `n*m`.

7. **Кластеризация Уишарта** (`scripts/wishart_compare.py` → `results/clustering/`):
   - Wishart на облаках человек/бот.
   - Метрики качества кластеризации из главы 23 Aggarwal & Reddy.

8. **Топология (опц.)** (`scripts/topology.py` → `results/topology/`):
   - persistent homology (ripser / giotto-tda).
   - Betti, persistence diagrams.

9. **Текст** (`paper/`).

## Структура

```
corpus_raw/         # сырые .txt по источникам
corpus_clean/       # лемматизированные тексты
corpus_bot/         # сгенерированные LSTM тексты
embeddings/         # svd / fasttext / word2vec
datasets/           # n-граммные датасеты
models/             # обученные модели (LSTM)
results/            # графики, метрики
scripts/            # рабочие скрипты
results/            # метрики (json) и графики
paper/              # сама курсовая (tex + pdf)
```

## Данные

Сырой и очищенный корпуса (~1,7 ГБ), SVD-словарь, веса LSTM и датасеты
$n$-грамм в репозиторий не включены из-за размера. Все скрипты в `scripts/`
содержат пути и параметры, достаточные для воспроизведения пайплайна с нуля;
источники корпуса перечислены в разделе «Пайплайн» выше. Готовые метрики и
графики, на которые опирается текст работы, лежат в `results/`.

## Литература

- Bellegarda — Latent Semantic Mapping (SVD основа)
- Aggarwal & Reddy — Data Clustering, гл. 23 (метрики)
- Gromov & Migrina (2020) — Language as Self-organized Critical System
- Gromov & Dang (2023) — Semantic and sentiment trajectories
- Селезнев, Коган, Данг — ВКР Spot the bot
- Xiaojin Zhu — Persistent Homology for NLP
- Edelsbrunner, Harer — Computational Topology
