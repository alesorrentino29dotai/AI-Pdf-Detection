from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "academic_reference.json"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class AcademicCalibrator:
    reference_scores: list[float]
    method: str = "empirical_cdf"

    def adjust(self, raw_ai_score: float) -> float:
        if not self.reference_scores:
            return raw_ai_score
        if self.method == "empirical_cdf":
            cdf = sum(1 for score in self.reference_scores if score <= raw_ai_score) / len(
                self.reference_scores
            )
            return max(0.0, min(1.0, 1.0 - cdf))
        if self.method == "platt":
            weight = self.platt_weight
            bias = self.platt_bias
            logit = _logit(raw_ai_score)
            return _sigmoid(weight * logit + bias)
        return raw_ai_score

    @property
    def platt_weight(self) -> float:
        return float(getattr(self, "_platt_weight", 1.0))

    @property
    def platt_bias(self) -> float:
        return float(getattr(self, "_platt_bias", 0.0))

    @classmethod
    def from_dict(cls, payload: dict) -> AcademicCalibrator:
        calibrator = cls(
            reference_scores=list(payload.get("reference_scores", [])),
            method=str(payload.get("method", "empirical_cdf")),
        )
        if "platt_weight" in payload:
            calibrator._platt_weight = float(payload["platt_weight"])
        if "platt_bias" in payload:
            calibrator._platt_bias = float(payload["platt_bias"])
        return calibrator

    def to_dict(self) -> dict:
        payload = {
            "method": self.method,
            "reference_scores": self.reference_scores,
            "reference_count": len(self.reference_scores),
        }
        if self.method == "platt":
            payload["platt_weight"] = self.platt_weight
            payload["platt_bias"] = self.platt_bias
        return payload


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    clipped = max(min(probability, 1.0 - 1e-6), 1e-6)
    return math.log(clipped / (1.0 - clipped))


def fetch_arxiv_abstracts(
    max_results: int = 200,
    categories: str = "cs.AR OR cat:cs.CR OR cat:cs.DC OR cat:cs.PL",
    before_date: str = "20221031",
    after_date: str = "20180101",
) -> list[str]:
    query = (
        f"({categories}) AND submittedDate:[{after_date}0000 TO {before_date}2359]"
    )
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    abstracts: list[str] = []
    for entry in root.findall("a:entry", ARXIV_NS):
        summary = entry.find("a:summary", ARXIV_NS)
        if summary is None or not summary.text:
            continue
        text = " ".join(summary.text.split())
        if len(text.split()) >= 20:
            abstracts.append(text)
    return abstracts


def load_calibrator(path: Path | None = None) -> AcademicCalibrator | None:
    reference_path = path or DEFAULT_REFERENCE_PATH
    if not reference_path.exists():
        return None
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    return AcademicCalibrator.from_dict(payload)


def save_calibrator(calibrator: AcademicCalibrator, path: Path | None = None) -> Path:
    reference_path = path or DEFAULT_REFERENCE_PATH
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(calibrator.to_dict(), indent=2), encoding="utf-8")
    return reference_path


def build_calibrator_from_scores(reference_scores: list[float]) -> AcademicCalibrator:
    return AcademicCalibrator(reference_scores=reference_scores, method="empirical_cdf")
