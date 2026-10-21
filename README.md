# whisper-server

Сервер распознавания речи с OpenAI-совместимым API. Внутри
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) или
[GigaAM-v3](https://github.com/salute-developers/GigaAM) (только русский); работает
на CPU и на GPU, запускается через `uv run` или Docker.

- `/v1/audio/transcriptions` и `/v1/audio/translations` в формате OpenAI, плюс
  простой `/transcribe`;
- ответы `json`, `text`, `srt`, `vtt`, `verbose_json` со словами и метками времени;
- записи до 2 часов, раздельное распознавание каналов стерео, подсказки лексики;
- модель грузится по первому запросу и выгружается по простою, при нехватке
  видеопамяти работа продолжается на CPU;
- очередь FIFO, параллельная обработка нескольких записей, токен по желанию.

Почему всё устроено именно так и на каких замерах это держится:
[docs/README.md](docs/README.md).

## Варианты

Движок и устройство выбираются при установке, один вариант за раз:

| Вариант | Движок | Устройство | uv | Docker-образ |
| --- | --- | --- | --- | --- |
| `cpu` | faster-whisper | CPU | `uv sync` | `ghcr.io/the80hz/whisper-api:latest` |
| `gpu` | faster-whisper | NVIDIA, CUDA 12 | `uv sync --extra gpu` | `ghcr.io/the80hz/whisper-api:gpu` |
| `gigaam` | GigaAM-v3 | NVIDIA | `uv sync --extra gigaam` | `ghcr.io/the80hz/whisper-api:gigaam` |
| `gigaam-cpu` | GigaAM-v3 | CPU | `uv sync --extra gigaam-cpu` | `ghcr.io/the80hz/whisper-api:gigaam-cpu` |

У каждого образа есть и тег, закреплённый на коммит: `sha-<коммит>`,
`gpu-sha-<коммит>`, `gigaam-sha-<коммит>`, `gigaam-cpu-sha-<коммит>`. Для
развёртывания лучше он: если `pull` оборвётся посреди слоя, плавающий тег молча
оставит старый локальный образ. `cpu`, `gpu` и `gigaam-cpu` собираются для
`linux/amd64` и `linux/arm64`, `gigaam` только для `linux/amd64`. Пакеты
публичные, логин не нужен.

## Быстрый старт

```bash
cp sample.env .env
```

### uv

Python 3.13 и [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # или --extra gpu, --extra gigaam, --extra gigaam-cpu
uv run whisper-server            # порт из PORT, по умолчанию 3373
```

GigaAM включается переменной `ENGINE=gigaam` (в `.env` или в окружении). Веса
скачиваются при первом старте: Whisper в `~/.cache/huggingface`, GigaAM в
`~/.cache/gigaam`. Пока качаются веса GigaAM, сервис не отвечает на HTTP.

То же через Makefile: `make setup` (или `setup-gpu`, `setup-gigaam`,
`setup-gigaam-cpu`), затем `make run`.

### Docker

```bash
docker compose up -d                                              # CPU
docker compose -f compose.yml -f compose.gpu.yml up -d            # GPU
docker compose --profile gigaam up -d                             # + GigaAM на CPU, порт 3375
docker compose -f compose.yml -f compose.gpu.yml --profile gigaam up -d   # + GigaAM на GPU
```

Compose берёт готовые образы из GHCR; `--build` соберёт их локально. Без compose:

```bash
docker build --build-arg VARIANT=gigaam-cpu -t whisper-api:gigaam-cpu .
docker run --env-file .env -e ENGINE=gigaam -p 3373:3373 \
  -v ./model-cache/gigaam:/root/.cache/gigaam whisper-api:gigaam-cpu
```

Для GPU нужен NVIDIA Container Toolkit и флаг `--gpus all`. Если CUDA падает с
ошибкой про `libcudnn_ops.so`, образ не тот: нужен `:gpu`, а не `:latest`.

### Проверка

```bash
curl -s localhost:3373/health
curl -s localhost:3373/v1/audio/transcriptions \
  -F file=@audio.wav -F language=ru -F response_format=text
```

## API

### `POST /v1/audio/transcriptions`

Поля `multipart/form-data` в духе OpenAI:

| Поле | Смысл |
| --- | --- |
| `file` | аудио или видео |
| `model` | принимается для совместимости; работает модель из настроек |
| `language` | код языка, например `ru` или `en` |
| `prompt` | начальная подсказка |
| `hotwords` | термины через запятую или с новой строки (имена, препараты, услуги), уходят в `hotwords` faster-whisper |
| `channels` | `mix` (по умолчанию) или `split`, см. [Длинные записи и стерео](#длинные-записи-и-стерео) |
| `timestamp_granularities[]` | `word` — слова внутри каждого сегмента |
| `response_format` | `json`, `text`, `srt`, `vtt`, `verbose_json` |
| `temperature` | температура декодирования, по умолчанию `0` |
| `timeout_seconds` | таймаут этого запроса на стороне сервера |

`/v1/audio/translations` устроен так же (без `hotwords`, `channels` и
`timestamp_granularities[]`) и переводит на английский задачей Whisper
`translate`.

Ответ `verbose_json`:

```json
{"task": "transcribe", "language": "ru", "duration": 123.4, "text": "...",
 "model": "large-v3-turbo", "channels": 2, "channel_mode": "split",
 "segments": [{"id": 0, "start": 0.0, "end": 2.1, "text": "...", "avg_logprob": -0.2,
               "compression_ratio": 1.1, "no_speech_prob": 0.01, "channel": 0,
               "words": [{"start": 0.0, "end": 0.4, "word": "...", "probability": 0.9}]}],
 "words": [{"start": 0.0, "end": 0.4, "word": "...", "probability": 0.9}]}
```

- `model` — модель, которая реально работала (после перехода на CPU это
  `CPU_FALLBACK_MODEL`).
- `channels` — число каналов файла, `channel_mode` повторяет запрос; `channel`
  есть у каждого сегмента при `split`.
- `words` внутри сегментов только с `timestamp_granularities[]=word`. Общий список
  `words` наверху ответа остаётся для совместимости и есть в любом `verbose_json`.

### `POST /transcribe`

Поле `file` и параметры запроса `task` (`transcribe` или `translate`),
`language`, `word_timestamps`, `timeout_seconds`. В ответе `text` на верхнем
уровне. Этим путём пользуется удалённый STT-клиент бота: Telegram `voice.ogg`
в `file`, `?language=ru`.

### `GET /health`

Состояние сервиса и модели. Спящую модель не будит, `?wake=true` будит
принудительно.

| Поле | Смысл |
| --- | --- |
| `model_state` | `loaded` или `sleeping` |
| `model`, `device`, `compute_type` | что использовала последняя загрузка, а не что написано в настройках |
| `cpu_fallback` | `true`, если модель на CPU после нехватки видеопамяти |
| `model_idle_seconds`, `model_unload_seconds` | сколько осталось до выгрузки |
| `transcribe_workers` | сколько записей обрабатывается одновременно |

Если задан `API_TOKEN`, пути распознавания требуют
`Authorization: Bearer <токен>`; `/health` открыт.

## Длинные записи и стерео

faster-whisper сам режет звук на окна по 30 с, поэтому двухчасовой файл не нужно
делить вручную. Ограничения:

- **Загрузка.** Тело запроса пишется на диск, не в память. Запрос с
  `Content-Length` больше `MAX_UPLOAD_MB` получает 413 до приёма тела; размер
  проверяется ещё раз при копировании. Временного места нужно примерно вдвое
  больше размера файла.
- **Длительность.** Запись длиннее `MAX_AUDIO_SECONDS` получает 413.
- **Таймаут.** Ожидание результата —
  `max(DEFAULT_TIMEOUT_SECONDS, длительность * TIMEOUT_PER_AUDIO_SECOND)`, время в
  очереди входит. Явный `timeout_seconds` важнее. Если длительность не прочиталась
  из заголовка, действует `DEFAULT_TIMEOUT_SECONDS`.
- **Память.** Файл декодируется целиком в 16 кГц float32: около 460 МБ на 2 часа
  моно. `channels=split` держит все каналы сразу, двухчасовое стерео — около 1 ГБ.

`channels=split` распознаёт каналы стерео по отдельности, и говорящий известен по
каналу без диаризации:

- у каждого сегмента есть `channel` (с нуля), сегменты всех каналов слиты в один
  список по `start`, `id` — позиция в нём;
- моно — это один канал (`channel: 0`), а не ошибка; файл с тремя и больше
  каналами делится на столько же прогонов;
- без `language` язык определяется по каналу 0 и используется для остальных;
- каналы идут по очереди, так что `split` занимает примерно суммарную длительность
  каналов.

Реальный двухчасовой файл на 500 МБ на GPU не проверялся.

## GigaAM

`ENGINE=gigaam` отдаёт тот же API, но распознаёт GigaAM-v3 (`v3_e2e_rnnt`). Очередь,
воркеры, ограничения, раздельные каналы и все форматы ответа работают так же.

Отличия:

- Модель берёт не больше 25 с за раз, поэтому запись режется Silero VAD на куски
  по паузам, куски идут через модель пачками (`GIGAAM_BATCH_SIZE`), каждый кусок
  делится на фразы по паузам между словами. Музыка ожидания не поглощает соседнюю
  речь, чем грешит Whisper.
- Детектор речи смотрит на запись дважды: как есть и выровненной по громкости
  (`GIGAAM_QUIET_SPEECH`). Из второго прохода берутся только слова там, где первый
  видел тишину. Так не пропадает тихий голос после громкого звука; цена — примерно
  вдвое больше работы модели.
- Текстового входа нет: `hotwords`, `prompt`, `temperature` и `language`
  принимаются и игнорируются. Редкие фамилии и названия препаратов выходят хуже,
  чем у Whisper с подсказками. Английский и перевод не поддерживаются.
- У слов нет `probability`, у сегментов нет `avg_logprob`.
- Тире диалога, которое модель пишет отдельным токеном, отбрасывается.
- При нехватке видеопамяти та же модель перегружается на CPU, меньшей нет.

Декодирование — цикл на Python, поэтому больше нескольких `TRANSCRIBE_WORKERS`
потоки ждут друг друга. Для большей пропускной способности — несколько
контейнеров или `uvicorn --workers N`.

## Засыпание модели и переход на CPU

Модель не держится в памяти между запросами: после `MODEL_UNLOAD_SECONDS` простоя
она выгружается и освобождает видеопамять, следующий запрос грузит её заново. На
общей карте это даёт соседям пользоваться ей, пока распознавать нечего.

Пробуждение может попасть на момент, когда карту занял кто-то другой. При
`CUDA_OOM_FALLBACK_CPU=true` нехватка видеопамяти не роняет запрос: модель
грузится на CPU, распознавание продолжается. CTranslate2 выделяет память лениво,
поэтому перехват работает и при загрузке, и посреди распознавания. Переход
действует на одну загрузку, следующее пробуждение снова пробует GPU.

Запуск на CPU — это не GPU с другим флагом:

- **Модель.** `WHISPER_MODEL` подобрана под GPU, на CPU она слишком медленная.
  Переход грузит `CPU_FALLBACK_MODEL` (`small`), если она не пуста.
- **Тип вычислений.** GPU-типов `int8_float16` и `float16` у CTranslate2 на CPU
  нет. Любой запуск на CPU берёт `CPU_FALLBACK_COMPUTE_TYPE` и при необходимости
  понижает его до поддерживаемого. Это касается и явного `DEVICE=cpu`, и
  `DEVICE=auto` без GPU; модель в этих случаях остаётся `WHISPER_MODEL`.

Замеры, по которым выбраны значения: [docs/research.md](docs/research.md).

## Параллельная обработка

По умолчанию записи обрабатываются по одной. `TRANSCRIBE_WORKERS=N` даёт N
записям идти одновременно на одной загруженной модели, остальные ждут в очереди
(`QUEUE_MAX_SIZE`). Каждая запись в работе занимает свою долю видеопамяти и ядро
CPU на декодирование звука, поэтому число поднимают на простаивающей карте, а не
на занятой.

## Настройки

Переменные окружения, образец — [sample.env](sample.env).

| Переменная | По умолчанию | Смысл |
| --- | --- | --- |
| `PORT` | `3373` | порт HTTP |
| `ENGINE` | `whisper` | `whisper` или `gigaam` |
| `WHISPER_MODEL` | `large-v3-turbo` | модель faster-whisper |
| `GIGAAM_MODEL` | `v3_e2e_rnnt` | модель GigaAM |
| `GIGAAM_DOWNLOAD_ROOT` | пусто | каталог весов GigaAM; пусто — `~/.cache/gigaam` |
| `GIGAAM_BATCH_SIZE` | `16` | кусков одной записи в одной пачке GigaAM |
| `GIGAAM_QUIET_SPEECH` | `true` | второй проход детектора по выровненной громкости |
| `DEVICE` | `auto` | `auto`, `cpu` или `cuda` |
| `COMPUTE_TYPE` | `int8` | тип вычислений faster-whisper: `int8`, `int8_float16`, `float16` |
| `CPU_THREADS` | `0` | потоки CPU faster-whisper; `0` — авто |
| `QUEUE_MAX_SIZE` | `8` | ожидающих задач, дальше 429 |
| `TRANSCRIBE_WORKERS` | `1` | записей одновременно |
| `DEFAULT_TIMEOUT_SECONDS` | `180` | минимальный таймаут запроса |
| `TIMEOUT_PER_AUDIO_SECOND` | `1.0` | таймаут на секунду аудио |
| `MAX_UPLOAD_MB` | `500` | предел размера файла |
| `MAX_AUDIO_SECONDS` | `7500` | предел длительности записи |
| `MODEL_UNLOAD_SECONDS` | `60` | выгрузка после простоя; `0` — не выгружать |
| `HEALTH_WAKE_MODEL` | `false` | разрешить `/health` будить модель |
| `CUDA_OOM_FALLBACK_CPU` | `true` | переход на CPU при нехватке видеопамяти |
| `CPU_FALLBACK_MODEL` | `small` | модель для этого перехода; пусто — `WHISPER_MODEL` |
| `CPU_FALLBACK_COMPUTE_TYPE` | `int8` | тип вычислений на CPU |
| `VAD_FILTER` | `true` | VAD faster-whisper перед распознаванием |
| `VAD_THRESHOLD` | `0.5` | порог вероятности речи Silero VAD |
| `VAD_MIN_SILENCE_DURATION_MS` | `500` | тишина, которая разделяет фрагменты речи |
| `VAD_SPEECH_PAD_MS` | `200` | запас вокруг найденной речи |
| `CONDITION_ON_PREVIOUS_TEXT` | `false` | подавать предыдущее окно в следующее; выключено против зацикливания |
| `REPETITION_PENALTY` | `1.1` | штраф за уже сгенерированные токены |
| `NO_REPEAT_NGRAM_SIZE` | `3` | запрет повторов n-грамм этой длины; `0` — без запрета |
| `COMPRESSION_RATIO_THRESHOLD` | `2.2` | слишком сжимаемый (повторяющийся) текст считается неудачей |
| `LOG_PROB_THRESHOLD` | `-1.0` | повтор декодирования при низкой средней вероятности |
| `NO_SPEECH_THRESHOLD` | `0.6` | порог «окно — тишина» |
| `TEMPERATURE_FALLBACK` | `true` | повтор с повышенной температурой при провале порогов |
| `HALLUCINATION_SILENCE_THRESHOLD` | `1.0` | пропуск тишины вокруг подозрительных галлюцинаций; `0` — выключить |
| `LOG_LEVEL` | `INFO` | уровень логов |
| `LOG_FILE` | `logs/whisper.log` | файл логов, каталог создаётся сам |
| `API_TOKEN` | пусто | bearer-токен для путей распознавания |
| `API_KEY` | пусто | старое имя `API_TOKEN` для совместимости; `API_TOKEN` важнее |

## Структура

```text
src/whisper_server/
├── server.py          API, очередь, жизненный цикл модели, faster-whisper
├── gigaam_engine.py   движок GigaAM: VAD-нарезка, пачки, второй проход
├── chunking.py        нарезка записи на куски по паузам
├── audio.py           чтение и декодирование аудио
└── config.py          настройки из окружения
scripts/               замеры скорости на CPU и GPU
docs/                  история, решения и замеры
```

## Разработка

```bash
uv sync
uv run ruff format --check . && uv run ruff check . && uv run pytest
```

CI (`.github/workflows/docker-publish.yml`) собирает все четыре образа на pull
request и публикует их в GHCR при push в `main`.
