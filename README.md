# CV Lab 3 — CIFAR-10 MLP

Три эксперимента с аугментациями (таблица **№2**, **№6**, **№7**), MLP ≤ 1M параметров, MLflow, чекпоинты и отчёт.

## Окружение

```powershell
cd C:\Users\steps\Projects\CV
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync
```

## Данные

- **Train (Kaggle):** `cifar-10/train/train/*.png` + `trainLabels.csv`
- **Test (с метками):** официальный split CIFAR-10 через `torchvision` в `data/torchvision_cifar10` (при первом запуске с `--download-test`)

Kaggle-папка `cifar-10/test/` без меток для обучения не используется.

## MLflow

```powershell
uv run mlflow server --host 0.0.0.0 --port 5000
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

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

`3072 – 256 (relu) – 128 (relu) – 64 (relu) – 10` (~828k параметров).

## Google Colab (бесплатный GPU)

1. На ПК соберите архив кода (без `cifar-10` и `.venv`):

   ```powershell
   .\scripts\pack_for_colab.ps1
   ```

   Файл появится в `Downloads\CV-lab3-project.zip`. Опционально: `.\scripts\pack_cifar_for_drive.ps1` — архив датасета.

2. На Google Drive создайте папку `CV-lab3`:

   - `CV-lab3-project.zip` — из шага 1;
   - `cifar-10/` — распакованный Kaggle train (`trainLabels.csv`, `train/train/*.png`).

3. Откройте `notebooks/colab_lab3.ipynb` в Colab (**File → Upload notebook** или перетащите файл). **Runtime → Change runtime type → GPU**.

4. В первой ячейке пути по умолчанию уже указывают на `My Drive/CV-lab3/`. Запустите все ячейки. После каждого варианта аугментации чекпоинты копируются в `Drive/CV-lab3/colab_outputs/`.

5. Скачайте `lab3_download_*.zip` с Drive, распакуйте `checkpoints/` (и `artifacts/` при необходимости) в корень проекта на ПК и соберите отчёт:

   ```powershell
   uv run python scripts/generate_report.py
   ```

Если репозиторий на GitHub — в ноутбуке задайте `GIT_REPO_URL` вместо zip.

## Заметки

- OpenCV тянется как зависимость **Albumentations** (`opencv-python-headless`); свой код на `cv2` не нужен.
- На CPU одна эпоха занимает заметное время; для отладки: `--epochs 2 --no-mlflow`.
