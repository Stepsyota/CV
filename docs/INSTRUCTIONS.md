# Lab 3 — инструкция по запуску

Полное руководство: окружение, данные, обучение на ПК / ноутбуке с GPU, Git, Colab, отчёт.

Краткая шпаргалка команд — в [README.md](../README.md).

---

## 1. Что нужно

- **Windows 10/11**, PowerShell
- **Python ≥ 3.11**, менеджер пакетов [uv](https://docs.astral.sh/uv/) (`uv` в PATH)
- **Датасет CIFAR-10 (Kaggle)** — папка `cifar-10` в корне проекта (не в Git)
- Для ускорения: **NVIDIA GPU** + актуальный драйвер (на CPU обучение долгое)

Модель: MLP `3072 → 256 → 128 → 64 → 10` (~828k параметров, лимит 1M).  
Три эксперимента: аугментации таблицы **№2**, **№6**, **№7** (`configs/aug_variant_*.yaml`).

---

## 2. Установка окружения

```powershell
cd C:\Users\steps\Projects\CV
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync
```

Проверка импорта:

```powershell
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

На машине с **RTX** должно быть `cuda True`. Если `False` — см. [раздел 9](#9-nvidia-gpu-rtx).

---

## 3. Данные

### Train (Kaggle)

В корне репозитория:

```
cifar-10/
  trainLabels.csv
  train/train/*.png    # или train/*.png
```

Распакуйте `train.7z` из Kaggle. Папка `cifar-10/test/` **без меток** для лабы **не используется** — в Colab/Drive её заливать не нужно (очень большой объём).

### Test (с метками)

Официальный test CIFAR-10 (10 000 изображений) скачивается через `torchvision` при флаге `--download-test` в каталог `data/torchvision_cifar10/` (в `.gitignore`).

### Переопределение пути к train

```powershell
$env:CIFAR10_DATA_ROOT = "D:\datasets\cifar-10"
uv run python scripts/train.py --config configs/aug_variant_2.yaml --download-test
```

### Режим `torchvision` (без Kaggle на диске)

Train и test берутся из официального CIFAR-10 (~170 MB, один раз скачивается в `data/torchvision_cifar10/`). Split train/val: 45k/5k, `seed=42`, как у Kaggle.

```powershell
uv run python scripts/train.py --config configs/aug_variant_2.yaml --data-source torchvision --download-test
```

Удобно для **Colab**, когда не хотите заливать датасет на Drive. На основном ПК с Kaggle-папкой обычно оставляют `data_source: kaggle` в `configs/base.yaml`.

---

## 4. Быстрая проверка (smoke test)

1–2 эпохи, без MLflow:

```powershell
uv run python scripts/train.py --config configs/aug_variant_2.yaml --no-mlflow --download-test --epochs 1
```

Ожидаемо: вывод блоков Architecture / Hyperparameters / Train & Test, файл `checkpoints/aug_table_2/best.pt`.

---

## 5. Полное обучение

### Один вариант

```powershell
uv run python scripts/train.py --config configs/aug_variant_2.yaml --download-test
```

Варианты: `aug_variant_2.yaml`, `aug_variant_6.yaml`, `aug_variant_7.yaml`.

Параметры по умолчанию в `configs/base.yaml` (например `epochs: 80`, early stopping).

### Все три варианта + демо + отчёт

```powershell
.\scripts\run_all_experiments.ps1
```

Скрипт последовательно обучает три конфига, запускает `demo_preprocessing.py` и `generate_report.py`.

### Полезные флаги `scripts/train.py`

| Флаг | Назначение |
|------|------------|
| `--config PATH` | Конфиг эксперимента (мержится с `base.yaml`) |
| `--download-test` | Скачать test CIFAR-10, если нет |
| `--no-mlflow` | Не логировать в MLflow |
| `--epochs N` | Переопределить число эпох |

---

## 6. MLflow (опционально)

В отдельном терминале:

```powershell
cd C:\Users\steps\Projects\CV
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv run mlflow server --host 127.0.0.1 --port 5000
```

Перед обучением:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
uv run python scripts/train.py --config configs/aug_variant_2.yaml --download-test
```

UI: http://127.0.0.1:5000. Каталог `mlruns/` не коммитится.

На ноутбуке для длинных прогонов удобнее `--no-mlflow`; метрики остаются в `checkpoints/*/history.json`.

---

## 7. Отчёт

После обучения (и при наличии чекпоинтов/графиков):

```powershell
uv run python scripts/generate_report.py
```

Результат: `reports/lab3_report.md` (и фигуры в `artifacts/`, `reports/figures/`).

Демо аугментаций отдельно:

```powershell
uv run python scripts/demo_preprocessing.py
```

Картинки: `artifacts/preprocess_demo/`.

---

## 8. Работа с ноутбуком (RTX) и Git

**Идея:** в Git только код и конфиги; **данные и чекпоинты** переносятся отдельно.

### Первый раз на основном ПК

```powershell
cd C:\Users\steps\Projects\CV
git init
git add -A
git commit -m "Initial commit: lab3 CIFAR-10 MLP"
git remote add origin <URL-репозитория>
git push -u origin master
```

> Пока **нет ни одного коммита**, команды вроде `git restore --staged` дают `fatal: could not resolve HEAD`. Сначала сделайте `git commit`.

### На ноутбуке с GPU

```powershell
git clone <URL> C:\Projects\CV
cd C:\Projects\CV
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync
```

Скопируйте папку **`cifar-10`** (USB / сеть / архив) в корень клона.  
Перед каждой сессией: `git pull`.

Обучение:

```powershell
.\scripts\run_all_experiments.ps1
```

### Вернуть результаты на основной ПК

Скопируйте с ноутбука (не через Git):

- `checkpoints/aug_table_2/`, `aug_table_6/`, `aug_table_7/`
- при необходимости `artifacts/`

Пример zip в PowerShell:

```powershell
Compress-Archive -Path checkpoints, artifacts -DestinationPath $env:USERPROFILE\Downloads\lab3-results.zip
```

На ПК распакуйте в корень проекта и выполните `uv run python scripts/generate_report.py`.

### Что не коммитить

См. `.gitignore`: `.venv/`, `cifar-10/`, `data/`, `checkpoints/`, `artifacts/`, `mlruns/`, `*.pdf`, архивы `*.7z`, `*.tar.gz`.

---

## 9. NVIDIA GPU (RTX)

1. Установите свежий драйвер NVIDIA.
2. После `uv sync` проверьте CUDA (см. раздел 2).
3. Если `cuda False`, установите PyTorch с CUDA под вашу версию: https://pytorch.org/get-started/locally/  
   Дальше подстройте зависимость в `pyproject.toml` / команду `uv pip install` по инструкции PyTorch (часто отдельный index `cu124` и т.п.).

Обучение автоматически использует `cuda`, если `torch.cuda.is_available()`.

---

## 10. Google Colab

Пошаговая настройка: **[COLAB_SETUP.md](COLAB_SETUP.md)** (чеклист, все ячейки, скачивание результатов на ПК).

Кратко: `USE_TORCHVISION_DATA = True` в ноутбуке — датасет в Colab, на Drive только zip с кодом; test с метками через `torchvision`, Kaggle `cifar-10/test/` не нужен.

---

## 11. Структура проекта

```
configs/           # base.yaml + варианты аугментаций
scripts/           # train.py, generate_report.py, run_all_experiments.ps1
src/lab3/          # данные, модель, обучение, MLflow
checkpoints/       # best.pt, history (после обучения)
artifacts/         # графики, демо
reports/           # lab3_report.md
notebooks/         # colab_lab3.ipynb
docs/              # эта инструкция
```

---

## 12. Частые проблемы

| Симптом | Что делать |
|---------|------------|
| `could not resolve HEAD` | Сделать первый `git commit` |
| `Не найдены PNG` / нет `trainLabels.csv` | Проверить распаковку `cifar-10`, путь `CIFAR10_DATA_ROOT` |
| Очень долго одна эпоха | CPU; использовать RTX или Colab |
| `cuda False` на ноутбуке с NVIDIA | Переустановить torch с CUDA |
| OpenCV / «шляпа» от препода | В проекте нет прямого `cv2`; headless OpenCV — зависимость Albumentations |

---

## 13. Контакты с методичкой

- Консольный вывод форматируется под требования лабы (`src/lab3/logging_fmt.py`).
- Лимит параметров проверяется в `src/lab3/model.py` (`max_params` в конфиге).
