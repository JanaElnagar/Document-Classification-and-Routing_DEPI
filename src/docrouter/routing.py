"""Turn class probabilities into a routing decision (FR2, FR3, FR4)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/routing.yaml")


@dataclass
class Router:
    routes: dict
    threshold: float
    review_queue: str = "Manual Review"

    @classmethod
    def from_yaml(cls, path=DEFAULT_CONFIG):
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(routes=cfg["routes"], threshold=float(cfg["threshold"]),
                   review_queue=cfg.get("review_queue", "Manual Review"))

    def route(self, probs: dict, top_k: int = 3) -> dict:
        """probs: {label: probability}. Returns category, confidence, department, top_k."""
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        category, confidence = ranked[0]
        needs_review = confidence < self.threshold
        department = self.review_queue if needs_review else self.routes.get(category, self.review_queue)
        return {
            "category": category,
            "confidence": round(float(confidence), 4),
            "department": department,
            "needs_review": needs_review,
            "top_k": [{"label": lab, "probability": round(float(p), 4)} for lab, p in ranked[:top_k]],
        }
