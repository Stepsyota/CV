# Лабораторная работа №3 — CIFAR-10 MLP

Дата отчёта: 2026-10-01 14:28

## Цель
Обучение MLP на CIFAR-10 с динамической предобработкой обучающей выборки, сравнение трёх пар методов из таблицы 1, трекинг в MLflow.

## Ограничения
- Только линейные слои (`nn.Linear`), ≤ 1000000 параметров.
- Фактическое число параметров: **828490**.
- Архитектура: `3072 – 256 (relu) – 128 (relu) – 64 (relu) – 10`

## Варианты предобработки

| ID | Методы | Run |
|---:|---|---|
| 2 | RandomCrop + HorizontalFlip | `aug_table_2` |
| 6 | RandomBrightnessContrast | `aug_table_6` |
| 7 | GaussNoise + Blur | `aug_table_7` |

## Результаты

| Вариант | Best test accuracy |
|---|---:|
| #2 (`aug_table_2`) | 0.5691 |
| #6 (`aug_table_6`) | 0.5520 |
| #7 (`aug_table_7`) | 0.5519 |

**Лучший вариант:** #2 (test accuracy 0.5691).

## Примеры аугментаций

### Таблица #2
![sample_1.png](figures/aug2_sample_1.png)
![sample_2.png](figures/aug2_sample_2.png)
![sample_3.png](figures/aug2_sample_3.png)

### Таблица #6
![sample_1.png](figures/aug6_sample_1.png)
![sample_2.png](figures/aug6_sample_2.png)
![sample_3.png](figures/aug6_sample_3.png)

### Таблица #7
![sample_1.png](figures/aug7_sample_1.png)
![sample_2.png](figures/aug7_sample_2.png)
![sample_3.png](figures/aug7_sample_3.png)

## Графики обучения

### aug_table_2
![curves](figures/curves_aug_table_2.png)

### aug_table_6
![curves](figures/curves_aug_table_6.png)

### aug_table_7
![curves](figures/curves_aug_table_7.png)

## Гиперпараметры (base)

- batch_size: 128
- lr: 0.001
- optimizer: adamw
- epochs (max): 80
- MLflow experiment: `lab3-cifar10-mlp`

## Вывод
Сравните устойчивость вариантов по val/test и выберите лучший по `best_test_accuracy`. Для MLP без свёрток целевые ~95% на CIFAR-10 обычно недостижимы; фиксируйте достигнутый максимум.