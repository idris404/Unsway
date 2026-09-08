"""Generate deterministic, dependency-free SVG figures from versioned reports."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from html import escape
from pathlib import Path
from typing import Any

from unsway.data.io import write_manifest
from unsway.reporting.config import Phase5Config

COLORS = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "purple": "#7c3aed",
    "gray": "#64748b",
    "grid": "#dbe3ee",
    "ink": "#172033",
    "muted": "#5b6577",
    "paper": "#ffffff",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Report must contain an object: {path}")
    return value


def _svg_shell(width: int, height: int, title: str, body: Iterable[str]) -> str:
    description = f"Unsway research figure: {title}"
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        f'<rect width="{width}" height="{height}" fill="{COLORS["paper"]}"/>',
        "<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}</style>",
        *body,
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    weight: int = 400,
    anchor: str = "start",
    color: str | None = None,
) -> str:
    fill = color or COLORS["ink"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}">{escape(value)}</text>'
    )


def _chart_frame(
    title: str, subtitle: str, width: int, height: int, y_max: float, y_label: str
) -> tuple[list[str], float, float, float, float]:
    left, top, right, bottom = 92.0, 112.0, width - 40.0, height - 82.0
    plot_width, plot_height = right - left, bottom - top
    axis_midpoint = top + plot_height / 2
    body = [
        _text(40, 42, title, size=24, weight=700),
        _text(40, 70, subtitle, color=COLORS["muted"]),
        (
            f'<text x="24" y="{axis_midpoint:.1f}" font-size="14" font-weight="600" '
            f'text-anchor="middle" fill="{COLORS["ink"]}" '
            f'transform="rotate(-90 24 {axis_midpoint:.1f})">{escape(y_label)}</text>'
        ),
    ]
    for step in range(6):
        value = y_max * step / 5
        y = bottom - plot_height * step / 5
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>'
        )
        body.append(
            _text(left - 12, y + 5, f"{value:.0f}%", size=12, anchor="end", color=COLORS["muted"])
        )
    return body, left, top, plot_width, plot_height


def grouped_bar_svg(
    title: str,
    subtitle: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    width: int,
    height: int,
    y_max: float,
    caption: str,
) -> str:
    """Render a grouped percentage bar chart as accessible SVG."""
    if not categories or not series:
        raise ValueError("Grouped bar charts require categories and series")
    if any(len(values) != len(categories) for _, values, _ in series):
        raise ValueError("Every bar series must align with categories")
    body, left, top, plot_width, plot_height = _chart_frame(
        title, subtitle, width, height, y_max, "Rate (%)"
    )
    bottom = top + plot_height
    group_width = plot_width / len(categories)
    bar_width = min(42.0, group_width * 0.72 / len(series))
    for category_index, category in enumerate(categories):
        center = left + group_width * (category_index + 0.5)
        for series_index, (_name, values, color) in enumerate(series):
            value = values[category_index]
            height_px = plot_height * value / y_max
            x = center + (series_index - (len(series) - 1) / 2) * bar_width - bar_width * 0.42
            y = bottom - height_px
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.84:.1f}" '
                f'height="{height_px:.1f}" rx="3" fill="{color}"/>'
            )
            body.append(
                _text(
                    x + bar_width * 0.42,
                    y - 8,
                    f"{value:.1f}%",
                    size=11,
                    weight=600,
                    anchor="middle",
                )
            )
        body.append(
            _text(center, bottom + 28, category, size=12, anchor="middle", color=COLORS["muted"])
        )
    legend_x = width - 40 - sum(95 + len(name) * 3 for name, _, _ in series)
    for name, _, color in series:
        body.append(
            f'<rect x="{legend_x:.1f}" y="88" width="12" height="12" rx="2" fill="{color}"/>'
        )
        body.append(_text(legend_x + 18, 99, name, size=12))
        legend_x += 95 + len(name) * 3
    body.append(_text(40, height - 28, caption, size=11, color=COLORS["muted"]))
    return _svg_shell(width, height, title, body)


def dose_response_svg(validation: dict[str, Any], *, width: int, height: int) -> str:
    """Render the positive-dose validation sweep."""
    strengths = [0.0, 0.5, 1.0, 2.0, 4.0]
    baseline = float(validation["baseline"]["pressured_target_rate"]) * 100
    names = [
        ("sae_4825", "SAE 4825", COLORS["blue"]),
        ("neuron_144", "Neuron 144", COLORS["green"]),
        ("random_control", "Random control", COLORS["gray"]),
    ]
    values: dict[str, list[float]] = {name: [baseline] for name, _, _ in names}
    for condition in validation["conditions"]:
        name = str(condition["intervention"])
        strength = float(condition["strength"])
        if name in values and strength > 0:
            values[name].append(float(condition["metrics"]["pressured_target_rate"]) * 100)
    body, left, top, plot_width, plot_height = _chart_frame(
        "Validation dose-response",
        "Targeted sycophancy under positive activation additions",
        width,
        height,
        30,
        "Targeted sycophancy (%)",
    )
    bottom = top + plot_height
    x_positions = [left + plot_width * i / (len(strengths) - 1) for i in range(len(strengths))]
    for x, strength in zip(x_positions, strengths, strict=True):
        body.append(
            _text(
                x,
                bottom + 28,
                f"+{strength:g}" if strength else "0",
                size=12,
                anchor="middle",
                color=COLORS["muted"],
            )
        )
    body.append(
        _text(
            left + plot_width / 2,
            height - 48,
            "Steering strength (residual-stream L2 norm)",
            size=12,
            weight=600,
            anchor="middle",
        )
    )
    for series_index, (name, label, color) in enumerate(names):
        points = []
        for x, value in zip(x_positions, values[name], strict=True):
            y = bottom - plot_height * value / 30
            points.append(f"{x:.1f},{y:.1f}")
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        body.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        legend_x = left + series_index * 180
        body.append(
            f'<line x1="{legend_x}" y1="91" x2="{legend_x + 24}" y2="91" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        body.append(_text(legend_x + 31, 96, label, size=12))
    body.append(
        _text(
            40,
            height - 22,
            "Source: reports/phase4_validation.json · validation split · positive doses only",
            size=11,
            color=COLORS["muted"],
        )
    )
    return _svg_shell(width, height, "Validation dose-response", body)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_phase5_artifacts(config: Phase5Config) -> dict[str, Any]:
    """Generate all final figures and a compact result manifest."""
    phase2 = _load(config.inputs.phase2_report)
    training = _load(config.inputs.phase3_training_report)
    features = _load(config.inputs.phase3_feature_report)
    validation = _load(config.inputs.phase4_validation_report)
    test = _load(config.inputs.phase4_test_report)
    width, height = config.style.width, config.style.height
    overall = phase2["metrics"]["overall"]
    by_source = phase2["metrics"]["by_source"]
    source_labels = ["AQuA", "Math", "MMLU", "TruthfulQA"]
    source_keys = ["aqua_mc", "math_mc_cot", "mmlu_mc_cot", "truthful_qa_mc"]
    output = config.output.figure_dir
    figures = {
        "behavioral_baseline": output / "behavioral_baseline.svg",
        "baseline_by_source": output / "baseline_by_source.svg",
        "feature_evidence": output / "feature_evidence.svg",
        "steering_dose_response": output / "steering_dose_response.svg",
        "heldout_steering": output / "heldout_steering.svg",
    }
    _write(
        figures["behavioral_baseline"],
        grouped_bar_svg(
            "Behavioral sycophancy baseline",
            "GPT-2 small on 996 initially-correct trials",
            ["Overall"],
            [
                (
                    "Pressured target",
                    [float(overall["pressured_target_rate"]) * 100],
                    COLORS["orange"],
                ),
                ("Neutral target", [float(overall["control_target_rate"]) * 100], COLORS["blue"]),
            ],
            width=width,
            height=height,
            y_max=50,
            caption=(
                "Source: reports/phase2_baseline.json · all splits · pressure effect "
                "+6.02 pp (95% CI +4.47 to +7.58)"
            ),
        ),
    )
    _write(
        figures["baseline_by_source"],
        grouped_bar_svg(
            "Targeted sycophancy by benchmark",
            "Rates use a fixed denominator of initially-correct examples",
            source_labels,
            [
                (
                    "Pressured target",
                    [float(by_source[key]["pressured_target_rate"]) * 100 for key in source_keys],
                    COLORS["orange"],
                ),
                (
                    "Neutral target",
                    [float(by_source[key]["control_target_rate"]) * 100 for key in source_keys],
                    COLORS["blue"],
                ),
            ],
            width=width,
            height=height,
            y_max=50,
            caption=(
                "Source: reports/phase2_baseline.json · AQuA, math, MMLU and TruthfulQA subsets"
            ),
        ),
    )
    feature = features["top_features"][0]
    _write(
        figures["feature_evidence"],
        grouped_bar_svg(
            "Out-of-sample behavioral discrimination",
            "Validation AUROC: SAE feature 4825 versus best raw residual neuron",
            ["SAE 4825", "Neuron 144"],
            [
                (
                    "Oriented AUROC",
                    [
                        float(feature["validation"]["oriented_auroc"]) * 100,
                        float(features["raw_neuron_baseline"]["validation_oriented_auroc"]) * 100,
                    ],
                    COLORS["purple"],
                )
            ],
            width=width,
            height=height,
            y_max=100,
            caption=(
                "Source: reports/phase3_features.json · SAE validation explained variance "
                f"{float(training['history'][-1]['validation']['explained_variance']) * 100:.2f}% "
                "· 3 dead features"
            ),
        ),
    )
    _write(
        figures["steering_dose_response"], dose_response_svg(validation, width=width, height=height)
    )
    test_baseline = test["baseline"]
    steered = test["conditions"][0]
    _write(
        figures["heldout_steering"],
        grouped_bar_svg(
            "Frozen held-out steering result",
            "Neuron 144 at validation-selected strength +4.0",
            ["Baseline", "Steered"],
            [
                (
                    "Targeted sycophancy",
                    [
                        float(test_baseline["pressured_target_rate"]) * 100,
                        float(steered["metrics"]["pressured_target_rate"]) * 100,
                    ],
                    COLORS["green"],
                )
            ],
            width=width,
            height=height,
            y_max=40,
            caption=(
                "Source: reports/phase4_test.json · 144 fixed eligible trials · delta "
                "-0.69 pp (95% CI -2.06 to +0.67)"
            ),
        ),
    )
    summary = {
        "schema_version": 1,
        "conclusion": "causal_effect_inconclusive",
        "behavioral_baseline": {
            "examples": int(overall["total_trials"]),
            "eligible": int(overall["eligible_trials"]),
            "initial_accuracy": float(overall["initial_accuracy"]),
            "pressured_target_rate": float(overall["pressured_target_rate"]),
            "control_target_rate": float(overall["control_target_rate"]),
            "pressure_effect": float(overall["pressure_effect"]),
        },
        "feature_evidence": {
            "sae_feature": 4825,
            "sae_validation_auroc": float(feature["validation"]["oriented_auroc"]),
            "raw_neuron": 144,
            "raw_neuron_validation_auroc": float(
                features["raw_neuron_baseline"]["validation_oriented_auroc"]
            ),
            "sae_validation_explained_variance": float(
                training["history"][-1]["validation"]["explained_variance"]
            ),
        },
        "heldout_intervention": {
            "name": str(steered["intervention"]),
            "strength": float(steered["strength"]),
            "eligible": int(test_baseline["fixed_eligible_trials"]),
            "baseline_rate": float(test_baseline["pressured_target_rate"]),
            "steered_rate": float(steered["metrics"]["pressured_target_rate"]),
            "delta": float(steered["delta"]["pressured_target_rate"]),
            "delta_ci_95": list(steered["delta"]["pressured_target_rate_ci_95"]),
        },
        "figures": {name: str(path) for name, path in figures.items()},
    }
    write_manifest(config.output.summary_report, summary)
    return summary
