"""Local re-implementation of the ZAC 2026 call/state score (train/dev only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .catalog import Catalog
from .text import token_f1


@dataclass
class SampleScore:
    score: float
    matched: int
    predicted: int
    gold: int
    precision: float
    recall: float
    hash_ok: bool
    empty_gold: bool
    empty_pred: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "matched": self.matched,
            "predicted": self.predicted,
            "gold": self.gold,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "hash_ok": self.hash_ok,
            "empty_gold": self.empty_gold,
            "empty_pred": self.empty_pred,
        }


def _arg_matches(pred_value: Any, gold_value: Any, prop: dict[str, Any]) -> bool:
    if prop.get("x-match") == "fuzzy" and isinstance(pred_value, str) and isinstance(gold_value, str):
        return token_f1(pred_value, gold_value) > 0.6
    return pred_value == gold_value


def calls_match(pred: dict[str, Any], gold: dict[str, Any], catalog: Catalog) -> bool:
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return False
    if pred.get("name") != gold.get("name"):
        return False
    pred_args = pred.get("arguments") if isinstance(pred.get("arguments"), dict) else {}
    gold_args = gold.get("arguments") if isinstance(gold.get("arguments"), dict) else {}
    if set(pred_args) != set(gold_args):
        return False
    tool = catalog.tools.get(str(pred.get("name")))
    if tool is None:
        return pred_args == gold_args
    for key, gold_value in gold_args.items():
        if key not in pred_args:
            return False
        prop = tool.prop(key)
        if not _arg_matches(pred_args[key], gold_value, prop):
            return False
    return True


def _max_matching(pred_calls: list[dict[str, Any]], gold_calls: list[dict[str, Any]], catalog: Catalog) -> int:
    if not pred_calls or not gold_calls:
        return 0
    adj: list[list[int]] = []
    for pred in pred_calls:
        adj.append([j for j, gold in enumerate(gold_calls) if calls_match(pred, gold, catalog)])
    match_gold = [-1] * len(gold_calls)

    def dfs(i: int, seen: list[bool]) -> bool:
        for j in adj[i]:
            if seen[j]:
                continue
            seen[j] = True
            if match_gold[j] == -1 or dfs(match_gold[j], seen):
                match_gold[j] = i
                return True
        return False

    matched = 0
    for i in range(len(pred_calls)):
        if dfs(i, [False] * len(gold_calls)):
            matched += 1
    return matched


def score_prediction(
    pred_calls: list[dict[str, Any]],
    pred_hash: str,
    gold_calls: list[dict[str, Any]],
    target_hash: str,
    catalog: Catalog,
) -> SampleScore:
    hash_ok = bool(pred_hash) and pred_hash == target_hash
    empty_gold = not gold_calls
    empty_pred = not pred_calls
    if not hash_ok:
        return SampleScore(0.0, 0, len(pred_calls), len(gold_calls), 0.0, 0.0, False, empty_gold, empty_pred)
    if empty_gold:
        score = 1.0 if empty_pred else 0.0
        return SampleScore(score, 0, len(pred_calls), 0, 1.0 if empty_pred else 0.0, 1.0 if empty_pred else 0.0, True, True, empty_pred)
    if empty_pred:
        return SampleScore(0.0, 0, 0, len(gold_calls), 0.0, 0.0, True, False, True)
    matched = _max_matching(pred_calls, gold_calls, catalog)
    if matched == 0:
        return SampleScore(0.0, 0, len(pred_calls), len(gold_calls), 0.0, 0.0, True, False, False)
    precision = matched / len(pred_calls)
    recall = matched / len(gold_calls)
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SampleScore(score, matched, len(pred_calls), len(gold_calls), precision, recall, True, False, False)


def aggregate(scores: Iterable[SampleScore]) -> dict[str, Any]:
    scores = list(scores)
    n = len(scores)
    if not n:
        return {"n": 0, "score": 0.0, "score_100": 0.0}
    return {
        "n": n,
        "score": sum(s.score for s in scores) / n,
        "score_100": 100.0 * sum(s.score for s in scores) / n,
        "hash_accuracy": sum(1 for s in scores if s.hash_ok) / n,
        "exact_calls": sum(1 for s in scores if s.matched == s.gold == s.predicted and s.hash_ok) / n,
        "empty_gold": sum(1 for s in scores if s.empty_gold),
        "nonempty_gold": sum(1 for s in scores if not s.empty_gold),
        "mean_predicted_calls": sum(s.predicted for s in scores) / n,
        "mean_gold_calls": sum(s.gold for s in scores) / n,
        "mean_matched_calls": sum(s.matched for s in scores) / n,
    }


def replay_gold(sample: Any, env_factory: Callable[[str], Any]) -> tuple[Any, list[dict[str, Any]], str]:
    """Debug/evaluation helper. The agent itself never creates a second env for a sample."""
    env = env_factory(sample.id)
    for call in sample.gold_calls:
        env.call(call.get("name", ""), call.get("arguments") or {})
    return env, env.get_tool_calls(), env.get_state_hash()


def evaluate_predictions(
    samples: Iterable[Any],
    predictions: dict[str, dict[str, Any]],
    catalog: Catalog,
    env_factory: Callable[[str], Any],
) -> dict[str, Any]:
    scores: list[SampleScore] = []
    details: list[dict[str, Any]] = []
    for sample in samples:
        _, _, target_hash = replay_gold(sample, env_factory)
        pred = predictions.get(sample.id) or {}
        score = score_prediction(
            list(pred.get("tool_calls") or []),
            str(pred.get("state_hash") or ""),
            sample.gold_calls,
            target_hash,
            catalog,
        )
        scores.append(score)
        details.append({"id": sample.id, **score.as_dict()})
    out = aggregate(scores)
    out["details"] = details
    return out
