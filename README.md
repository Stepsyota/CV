# CV Lab 3 — CIFAR-10 MLP-Mixer

Три эксперимента с аугментациями (таблица **№2**, **№6**, **№7**), MLP-Mixer (только `nn.Linear`) ≤ 1M параметров, MLflow, чекпоинты и отчёт.

**Подробная инструкция:** [docs/INSTRUCTIONS.md](docs/INSTRUCTIONS.md) (установка, GPU-ноутбук, Git, отчёт, troubleshooting).  
**Colab по шагам:** [docs/COLAB_SETUP.md](docs/COLAB_SETUP.md).  
**Теория (один файл, с нуля):** [docs/LAB3_THEORY.md](docs/LAB3_THEORY.md).

## Окружение

```powershell
cd C:\Users\steps\Projects\CV
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync
uv run python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"
```

На Windows/Linux `uv sync` тянет **CUDA**-сборку torch (`cu128`). Если в версии есть `+cpu` или `cuda False` — на машине нет драйвера NVIDIA / нет дискретной GPU, либо нужен другой индекс CUDA (см. [INSTRUCTIONS.md](docs/INSTRUCTIONS.md) §9).

## Данные

- **Train (Kaggle):** `cifar-10/train/train/*.png` + `trainLabels.csv`
- **Test (с метками):** официальный split CIFAR-10 через `torchvision` в `data/torchvision_cifar10` (при первом запуске с `--download-test`)

Kaggle-папка `cifar-10/test/` без меток для обучения не используется.

## MLflow

Метрики пишутся в `./mlruns` при обучении. UI:

```powershell
.\scripts\mlflow_server.ps1
```

Откройте http://127.0.0.1:5000, эксперимент `lab3-cifar10-mlp`.

## Команды

Демо предобработки:

```powershell
uv run python scripts/demo_preprocessing.py
```

Один эксперимент:

```powershell
uv run python scripts/train.py --config configs/aug_variant_2.yaml --download-test
```

Все три + отчёт:

```powershell
.\scripts\run_all_experiments.ps1
```

Отчёт вручную:

```powershell
uv run python scripts/generate_report.py
```

Результат: `reports/lab3_report.md`, чекпоинты в `checkpoints/aug_table_*`.

## Архитектура по умолчанию

MLP-Mixer: патчи **8×8** (размер 4 px), `dim=128`, `depth=12` (~900k параметров). Классический MLP: `model.type: mlp` в `configs/base.yaml`.

## Заметки

- OpenCV тянется как зависимость **Albumentations** (`opencv-python-headless`); свой код на `cv2` не нужен.
- На CPU одна эпоха занимает заметное время; для отладки: `--epochs 2 --no-mlflow`.
