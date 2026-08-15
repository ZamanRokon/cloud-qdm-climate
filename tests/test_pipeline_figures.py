from dataclasses import replace
from pathlib import Path

from cloud_qdm import pipeline
from cloud_qdm.config import load_config


def _paper_config():
    path = Path(__file__).parents[1] / "configs" / "example_paper.yml"
    return load_config(path)


def test_model_by_model_false_collects_metrics_without_rendering(monkeypatch) -> None:
    config = _paper_config()
    expected = [{"model": "TEST", "stage": "raw"}]
    rendered = {"called": False}

    monkeypatch.setattr(pipeline, "build_evaluation_rows", lambda *_args, **_kwargs: expected)

    def fail_if_rendered(*_args, **_kwargs):
        rendered["called"] = True
        return [], []

    monkeypatch.setattr(pipeline, "make_evaluation_figures", fail_if_rendered)
    paths, rows = pipeline._evaluation_diagnostics(config, ({}, {}, {}), model="TEST")

    assert paths == []
    assert rows == expected
    assert not rendered["called"]


def test_model_by_model_true_uses_per_model_renderers(monkeypatch) -> None:
    base = _paper_config()
    config = replace(base, figures=replace(base.figures, model_by_model=True))
    evaluation_path = Path("evaluation.png")
    projection_path = Path("projection.png")

    monkeypatch.setattr(
        pipeline,
        "make_evaluation_figures",
        lambda *_args, **_kwargs: ([evaluation_path], [{"stage": "raw"}]),
    )
    monkeypatch.setattr(
        pipeline,
        "make_projection_figures",
        lambda *_args, **_kwargs: ([projection_path], [{"quantile": 0.5}]),
    )

    evaluation_paths, _ = pipeline._evaluation_diagnostics(config, ({}, {}, {}), model="TEST")
    projection_paths, _ = pipeline._projection_diagnostics(
        config,
        {},
        {},
        {},
        {},
        model="TEST",
        scenario="ssp245",
        period="near-term",
    )

    assert evaluation_paths == [evaluation_path]
    assert projection_paths == [projection_path]
