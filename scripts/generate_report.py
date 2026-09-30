from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from lab3.augment import describe_augment_variant
from lab3.config import load_config
from lab3.model import MLP, count_trainable_parameters


def _read_summary(checkpoint_root: Path, run_name: str) -> dict | None:
    path = checkpoint_root / run_name / "summary.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lab3 markdown report")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/lab3_report.md"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo = Path(cfg["_repo_root"])
    checkpoint_root = Path(cfg["checkpoint_dir"])
    figures_dir = args.out.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        (2, "configs/aug_variant_2.yaml"),
        (6, "configs/aug_variant_6.yaml"),
        (7, "configs/aug_variant_7.yaml"),
    ]

    model = MLP(
        hidden_dims=cfg["model"]["hidden_dims"],
        activation=cfg["model"].get("activation", "relu"),
        dropout=float(cfg["model"].get("dropout", 0)),
    )
    num_params = count_trainable_parameters(model)

    lines: list[str] = [
        "# Лабораторная работа №3 — CIFAR-10 MLP",
        "",
        f"Дата отчёта: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Цель",
        "Обучение MLP на CIFAR-10 с динамической предобработкой обучающей выборки, "
        "сравнение трёх пар методов из таблицы 1, трекинг в MLflow.",
        "",
        "## Ограничения",
        f"- Только линейные слои (`nn.Linear`), ≤ {cfg.get('max_params', 1_000_000)} параметров.",
        f"- Фактическое число параметров: **{num_params}**.",
        f"- Архитектура: `{model.architecture_string()}`",
        "",
        "## Варианты предобработки",
        "",
        "| ID | Методы | Run |",
        "|---:|---|---|",
    ]

    results: list[tuple[int, str, float | None]] = []
    for aug_id, cfg_rel in variants:
        vcfg = load_config(repo / cfg_rel)
        run_name = vcfg.get("run_name", f"aug_table_{aug_id}")
        summary = _read_summary(checkpoint_root, run_name)
        best = summary["best_test_accuracy"] if summary else None
        results.append((aug_id, run_name, best))
        lines.append(f"| {aug_id} | {describe_augment_variant(aug_id)} | `{run_name}` |")

    lines.extend(["", "## Результаты", "", "| Вариант | Best test accuracy |", "|---|---:|"])
    best_overall = ("—", -1.0)
    for aug_id, run_name, best in results:
        val = f"{best:.4f}" if best is not None else "нет прогона"
        lines.append(f"| #{aug_id} (`{run_name}`) | {val} |")
        if best is not None and best > best_overall[1]:
            best_overall = (f"#{aug_id}", best)

    lines.extend(
        [
            "",
            f"**Лучший вариант:** {best_overall[0]} (test accuracy {best_overall[1]:.4f})."
            if best_overall[1] >= 0
            else "",
            "",
            "## Примеры аугментаций",
            "",
        ]
    )

    demo_root = repo / "artifacts" / "preprocess_demo"
    for aug_id, _, _ in results:
        variant_dir = demo_root / f"variant_{aug_id}"
        if not variant_dir.is_dir():
            lines.append(f"_Нет картинок для варианта {aug_id}. Запустите `scripts/demo_preprocessing.py`._")
            continue
        lines.append(f"### Таблица #{aug_id}")
        for img in sorted(variant_dir.glob("*.png"))[:3]:
            dest = figures_dir / f"aug{aug_id}_{img.name}"
            shutil.copy2(img, dest)
            rel = dest.relative_to(args.out.parent)
            lines.append(f"![{img.name}]({rel.as_posix()})")
        lines.append("")

    lines.extend(["## Графики обучения", ""])
    for aug_id, run_name, _ in results:
        curves = checkpoint_root / run_name / "accuracy_curves.png"
        if curves.is_file():
            dest = figures_dir / f"curves_{run_name}.png"
            shutil.copy2(curves, dest)
            rel = dest.relative_to(args.out.parent)
            lines.append(f"### {run_name}")
            lines.append(f"![curves]({rel.as_posix()})")
            lines.append("")

    lines.extend(
        [
            "## Гиперпараметры (base)",
            "",
            f"- batch_size: {cfg['batch_size']}",
            f"- lr: {cfg['lr']}",
            f"- optimizer: {cfg.get('optimizer')}",
            f"- epochs (max): {cfg['epochs']}",
            f"- MLflow experiment: `{cfg['mlflow']['experiment_name']}`",
            "",
            "## Вывод",
            "Сравните устойчивость вариантов по val/test и выберите лучший по `best_test_accuracy`. "
            "Для MLP без свёрток целевые ~95% на CIFAR-10 обычно недостижимы; фиксируйте достигнутый максимум.",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
