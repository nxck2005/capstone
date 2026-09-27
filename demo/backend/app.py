"""Offline demo: published validation chart plus fresh CPU inference on approved images.

Frozen checkpoints are bundled separately; chart counts never stand in for an
individual prediction. The final-test split remains sealed and unused.
"""

from __future__ import annotations

import csv
import base64
import hashlib
import io
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from config.params import get  # noqa: E402  (project sources live under src/)

EVIDENCE = ROOT / "results/learned/w10"
CLOSEOUT = EVIDENCE / "w10_continuation_closeout_v11.json"
UNITS = EVIDENCE / "w10_continuation_units_v11.json"
UNIT_MANIFEST = EVIDENCE / "w10_continuation_unit_manifest_v11.json"
PER_IMAGE_MANIFEST = EVIDENCE / "w10_continuation_per_image_manifest_v11.json"
LEARNED_ASSET = "demo/assets/checkpoints/learned-r1-6.pt"
EXAMPLES = "demo/assets/examples"
PRIMARY_CSV = "presentation-results/data/w10_primary_252.csv"
CLASS_NAMES = (
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Evidence:
    """Immutable in-memory projection of committed aggregate-only evidence."""

    def __init__(self, root: Path = ROOT) -> None:
        base = root / "results/learned/w10"
        closeout = _json(base / CLOSEOUT.name)
        published = _json(base / UNITS.name)
        if closeout.get("status") != "COMPLETE" or closeout.get("validation_only") is not True:
            raise RuntimeError("W10 validation closeout is not complete")
        if closeout.get("test") != "SEALED" or closeout.get("test_access") != 0:
            raise RuntimeError("test boundary is not sealed")
        if published.get("test") != "SEALED" or published.get("test_access") != 0:
            raise RuntimeError("published units do not preserve sealed test")
        for name, field in (
            (UNIT_MANIFEST.name, "unit_manifest_sha256"),
            (PER_IMAGE_MANIFEST.name, "per_image_manifest_sha256"),
        ):
            if _sha256(base / name) != closeout[field]:
                raise RuntimeError(f"W10 {name} differs from closeout")
        unit_manifest = _json(base / UNIT_MANIFEST.name)
        per_image_manifest = _json(base / PER_IMAGE_MANIFEST.name)
        units = published["units"]
        if len(units) != closeout["unit_count"] or len(units) != published["unit_count"]:
            raise RuntimeError("W10 unit count differs from closeout")
        if [unit["unit_id"] for unit in units] != [item["unit_id"] for item in closeout["ordinal_provenance"]]:
            raise RuntimeError("W10 unit ordering differs from closeout")
        if unit_manifest.get("unit_count") != len(units) or len(unit_manifest.get("units", [])) != len(units):
            raise RuntimeError("W10 unit manifest count differs")
        if per_image_manifest.get("stream_count") != closeout["stream_count"] or len(per_image_manifest.get("streams", [])) != closeout["stream_count"]:
            raise RuntimeError("W10 scorer stream count differs")
        for unit, bound in zip(units, unit_manifest["units"], strict=True):
            if any(unit.get(field) != bound.get(field) for field in (
                "ordinal", "unit_id", "n_correct", "n_total", "per_image_path", "per_image_sha256", "scorer_variants"
            )):
                raise RuntimeError("W10 published unit differs from bound manifest")
        grid = [float(value) for value in get("channel.test_snr_grid_db")]
        curves: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for index, unit in enumerate(units):
            if unit.get("ordinal") != index or unit.get("split") != "val" or not unit.get("validation_only"):
                raise RuntimeError("W10 unit is not an ordered validation unit")
            if unit.get("test_access") != 0 or unit.get("test") != "SEALED":
                raise RuntimeError("W10 unit crosses the test boundary")
            total, correct = unit["n_total"], unit["n_correct"]
            if total != 1000 or not (0 <= correct <= total) or unit["denominator"] != total:
                raise RuntimeError("invalid W10 aggregate count")
            key = (unit["system"], unit["bw_ratio"])
            curves.setdefault(key, []).append({
                "snr_db": unit["snr_db"],
                "n_correct": correct,
                "n_total": total,
                "accuracy": correct / total,
                "coverage_rate": unit["coverage_rate"],
                "unit_id": unit["unit_id"],
            })
        if len(curves) != 12 or any([float(p["snr_db"]) for p in points] != grid for points in curves.values()):
            raise RuntimeError("W10 scope or SNR grid is incomplete")
        with (root / PRIMARY_CSV).open(newline="", encoding="utf-8") as stream:
            csv_rows = list(csv.DictReader(stream))
        if len(csv_rows) != len(units):
            raise RuntimeError("presentation CSV cardinality differs from W10 units")
        for ordinal, (row, unit) in enumerate(zip(csv_rows, units, strict=True)):
            if (
                int(row["ordinal"]) != ordinal
                or row["unit_id"] != unit["unit_id"]
                or row["system"] != unit["system"]
                or row["bw_ratio"] != unit["bw_ratio"]
                or float(row["snr_db"]) != float(unit["snr_db"])
                or int(row["n_correct"]) != unit["n_correct"]
                or int(row["n_total"]) != unit["n_total"]
                or float(row["accuracy_end_to_end"]) != unit["n_correct"] / unit["n_total"]
                or float(row["coverage_rate"]) != float(unit["coverage_rate"])
                or row["primary_scorer"] != unit["scorer_variants"][0]["classifier_variant"]
                or int(row["decode_failure_count"]) != unit["decode_failure_count"]
                or int(row["infeasible_count"]) != unit["infeasible_count"]
            ):
                raise RuntimeError(f"presentation CSV differs from authenticated W10 unit {ordinal}")
        self.curves = curves
        self.grid = grid
        self.closeout_id = closeout["closeout_id"]
        self.units_sha256 = _sha256(base / UNITS.name)
        self.root = root
        self.learned_checkpoint_id = next(
            unit["binding"]["checkpoint_id"] for unit in units
            if unit["system"] == "learned" and unit["bw_ratio"] == "r_1_6"
        )
        self.live = LiveLearned(root, self.learned_checkpoint_id)
        authority = _json(base / "w10_rehearsal_authorization.json")
        classical_binding = next(
            binding for binding in authority["bindings"]
            if binding["scope"]["system"] == "classical_adaptive" and binding["scope"]["bw_ratio"] == "r_1_6"
        )
        self.classical = LiveClassical(root, classical_binding)

        # The published val manifest is metadata only. Never import the test
        # loader, parse test rows, or expose a caller-controlled file path.
        with (root / "data/manifests/imagenette160.csv").open(newline="", encoding="utf-8") as stream:
            self.images = [
                {"image_id": row["stable_sample_id"], "label_index": int(row["label"])}
                for row in csv.DictReader(stream) if row["split"] == "val"
            ]
        if len(self.images) != 1000 or len({r["image_id"] for r in self.images}) != 1000:
            raise RuntimeError("validation image manifest is incomplete")
        self.image_indices = {row["image_id"]: index for index, row in enumerate(self.images)}
        self.examples: dict[str, dict[str, Any]] = {}
        example_manifest = _json(root / EXAMPLES / "manifest.json")
        if example_manifest.get("dataset") != "imagenette160" or example_manifest.get("source_manifest_sha256") != _sha256(root / "data/manifests/imagenette160.csv"):
            raise RuntimeError("bundled training examples disagree with manifest")
        with (root / "data/manifests/imagenette160.csv").open(newline="", encoding="utf-8") as stream:
            train_labels = {row["stable_sample_id"]: int(row["label"]) for row in csv.DictReader(stream) if row["split"] == "train"}
        from data.identity import stable_sample_id

        for row in example_manifest["images"]:
            slug = row["id"]
            if not isinstance(slug, str) or not slug.replace("-", "").isalpha() or row["file"] != f"{slug}.jpeg":
                raise RuntimeError("bundled training example has an unsafe file name")
            image_id = row["stable_sample_id"]
            path = root / EXAMPLES / row["file"]
            if not path.is_file() or path.is_symlink():
                raise RuntimeError("bundled training example is missing or unsafe")
            source = path.read_bytes()
            if row["split"] != "train" or hashlib.sha256(source).hexdigest() != row["sha256"] or stable_sample_id(source) != image_id or train_labels.get(image_id) != row["class_index"]:
                raise RuntimeError("bundled training example fails source/manifest authentication")
            self.examples[image_id] = {"source": source, "id": slug, "label": row["label"], "label_index": row["class_index"]}
        if len(self.examples) != 4:
            raise RuntimeError("bundled training examples are incomplete")
        self._dataset: Any = None
        self._png_cache: OrderedDict[str, bytes] = OrderedDict()
        self._lock = RLock()

    def image_png(self, image_id: str) -> bytes:
        with self._lock:
            if image_id in self._png_cache:
                self._png_cache.move_to_end(image_id)
                return self._png_cache[image_id]
            product = self.product(image_id)
            from PIL import Image

            output = io.BytesIO()
            Image.fromarray(product.canonical_image, mode="RGB").save(output, format="PNG")
            self._png_cache[image_id] = output.getvalue()
            if len(self._png_cache) > 32:
                self._png_cache.popitem(last=False)
            return self._png_cache[image_id]

    def product(self, image_id: str) -> Any:
        if image_id in self.examples:
            from data.preprocessing import canonicalize_source

            return canonicalize_source(self.examples[image_id]["source"], "imagenette160")
        if image_id not in self.image_indices:
            raise KeyError("unknown validation image")
        with self._lock:
            if self._dataset is None:
                from data.registry import load_dataset

                self._dataset = load_dataset("imagenette160", "val", self.root)
                actual = [self._dataset.source_sample(i).stable_sample_id for i in range(len(self._dataset))]
                if actual != [row["image_id"] for row in self.images]:
                    self._dataset = None
                    raise RuntimeError("dataset differs from the committed validation manifest")
            product, label = self._dataset[self.image_indices[image_id]]
            if product.stable_sample_id != image_id or label != self.images[self.image_indices[image_id]]["label_index"]:
                raise RuntimeError("validation image identity differs")
            return product


class LiveLearned:
    """Read-only W8 checkpoint inference, one frozen r_1_6 seed cell on CPU."""

    def __init__(self, root: Path, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id
        path = root / LEARNED_ASSET
        # Retain the authenticated snapshot in memory: no second file read can
        # race a replacement or a partial transfer between SHA and torch.load.
        self.checkpoint_bytes = path.read_bytes() if path.is_file() and not path.is_symlink() else None
        if self.checkpoint_bytes is not None and hashlib.sha256(self.checkpoint_bytes).hexdigest() != checkpoint_id:
            self.checkpoint_bytes = None
        self.model: Any = None
        self._cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._lock = RLock()

    @property
    def available(self) -> bool:
        return self.checkpoint_bytes is not None

    def _model(self) -> Any:
        import torch

        from models.djscc import build_djscc
        from training.w8_protocol import W8_CHECKPOINT_ROLE, load_w8_config

        if self.checkpoint_bytes is None:
            raise RuntimeError("frozen learned checkpoint is missing or differs from published SHA-256")
        if self.model is None:
            checkpoint = torch.load(io.BytesIO(self.checkpoint_bytes), map_location="cpu", weights_only=True)
            if (
                checkpoint.get("artifact_role") != W8_CHECKPOINT_ROLE
                or checkpoint.get("run_id") != "w8-r_1_6-train0-channel0"
                or checkpoint.get("completed_epoch") != 92
            ):
                raise RuntimeError("learned checkpoint role, cell or selected epoch differs")
            config = load_w8_config("r_1_6", 0, 0)
            model = build_djscc(config, device="cpu")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            self.model = model
        return self.model

    def infer(self, product: Any, snr_db: int) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        from channels.awgn import keyed_complex_noise
        from data.preprocessing import evaluation_input
        from evaluation.w10_evidence import scheduled_noise_id

        key = (product.stable_sample_id, snr_db)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return {**self._cache[key], "cache_hit": True, "inference_ms": 0.0}
            model = self._model()
            start = time.perf_counter()
            noise_id = scheduled_noise_id(
                stable_sample_id=product.stable_sample_id,
                bw_ratio="r_1_6", test_snr_db=snr_db, k=model.k,
            )
            noise = keyed_complex_noise(noise_id, model.k, dtype=torch.complex64, device="cpu")
            inputs = evaluation_input(product).unsqueeze(0)
            with torch.inference_mode():
                output = model(inputs, float(snr_db), unit_noise=noise)
                probs = torch.softmax(output.logits[0], dim=0)
                label = int(probs.argmax().item())
                confidence = float(probs[label].item())
                preview = output.reconstruction[0].detach().cpu().permute(1, 2, 0).numpy()
            if not np.isfinite(preview).all() or not np.isfinite(confidence):
                raise RuntimeError("non-finite learned inference output")
            rgb = np.rint(np.clip(preview, 0, 1) * 255).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
            result = {
                "status": "available", "system": "learned", "label_index": label,
                "confidence": confidence, "confidence_kind": "uncalibrated_softmax_own_task_head",
                "reconstruction_png_data_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                "reconstruction_role": "visualization_only_not_a_reported_metric",
                "checkpoint_sha256": self.checkpoint_id, "noise_id": noise_id,
                "inference_ms": round((time.perf_counter() - start) * 1000, 2), "cache_hit": False,
            }
            self._cache[key] = result
            if len(self._cache) > 64:
                self._cache.popitem(last=False)
            return result


class LiveClassical:
    """Frozen BR-4 adaptive JPEG 2000 + LDPC single-image CPU path."""

    def __init__(self, root: Path, authority_binding: dict[str, Any]) -> None:
        self.root = root
        self.authority_binding = authority_binding
        freeze = _json(root / "results/baseline/g8_f/artifact_classifier_freeze.json")
        self.checkpoint_id = freeze["checkpoint_file_sha256"]
        path = root / "demo/assets/checkpoints/artifact-classifier.pt"
        self.checkpoint_bytes = path.read_bytes() if path.is_file() and not path.is_symlink() else None
        if self.checkpoint_bytes is not None and (
            hashlib.sha256(self.checkpoint_bytes).hexdigest() != self.checkpoint_id
            or len(self.checkpoint_bytes) != freeze["checkpoint_bytes"]
        ):
            self.checkpoint_bytes = None
        try:
            from env import loaded_openjpeg_version

            self.codec_ready = loaded_openjpeg_version(required=True) == str(get("environment.openjpeg"))
        except (ImportError, RuntimeError, OSError, ValueError):
            self.codec_ready = False
        self.classifier: Any = None
        self._cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._lock = RLock()

    @property
    def available(self) -> bool:
        return self.checkpoint_bytes is not None and self.codec_ready

    def _classifier(self) -> Any:
        import torch
        from models.reference_classifier import build_reference_classifier

        if self.classifier is None:
            if self.checkpoint_bytes is None:
                raise RuntimeError("frozen artifact classifier is missing or differs from its published SHA-256")
            checkpoint = torch.load(io.BytesIO(self.checkpoint_bytes), map_location="cpu", weights_only=True)
            if checkpoint.get("completed_epoch") != 17:
                raise RuntimeError("artifact scorer selected epoch differs")
            model = build_reference_classifier("imagenette160", device="cpu")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            self.classifier = model
        return self.classifier

    def infer(self, product: Any, *, label: int, snr_db: int) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        from baseline.classical.pipeline import ChannelIdentity, DELIVERED, run_classical_pipeline
        from baseline.classical.records import score_result
        from baseline.j2k import J2KCodec
        from config.params import get
        from data.preprocessing import codec_input, reconstruction_input
        from evaluation.w10_bindings import binding_for
        from evaluation.w10_classical import (
            _candidate_table, _outage_policy, _selection_map, resolve_classical_configuration,
        )
        from evaluation.w10_evidence import dataset_version, split_manifest_hash

        key = (product.stable_sample_id, snr_db)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return {**self._cache[key], "cache_hit": True, "inference_ms": 0.0}
            start = time.perf_counter()
            current = binding_for(self.root, "classical_adaptive", "r_1_6")
            if current["binding_id"] != self.authority_binding["binding_id"]:
                raise RuntimeError("frozen W10 classical binding drifted")
            binding = current["selection"]
            point = _selection_map(binding)[snr_db]
            candidates = _candidate_table(self.root, expected_sha256=binding["candidate_authority"]["sha256"])
            modulation, ldpc_rate, encode_axis, _ = resolve_classical_configuration(
                binding=binding, point=point, candidates=candidates, codec_kind="jpeg2000", quality=None,
            )
            result = run_classical_pipeline(
                product, dataset="imagenette160",
                k_symbols=int(get("bandwidth.k_symbols.imagenette160.r_1_6")),
                modulation=modulation, ldpc_rate=ldpc_rate, snr_db=float(snr_db),
                codec=J2KCodec(self.root / "demo/backend/.cache/j2k"),
                channel_identity=ChannelIdentity(dataset_version(), split_manifest_hash(), 0),
                encode_axis_px=encode_axis, device="cpu",
            )
            policy = _outage_policy(self.root)
            classifier = self._classifier() if result.verdict == DELIVERED else None
            outcome = score_result(
                result, true_label=label, policy=policy,
                canonical_image=codec_input(product) if result.verdict == DELIVERED else None,
                classifier=classifier, device="cpu",
            )
            image_url = None
            confidence = None
            if result.verdict == DELIVERED:
                if result.decoded_image is None or classifier is None:
                    raise RuntimeError("delivered classical row lacks decoded pixels or scorer")
                with torch.inference_mode():
                    logits = classifier(reconstruction_input(result.decoded_image).unsqueeze(0))[0]
                    probabilities = torch.softmax(logits, dim=0)
                if int(probabilities.argmax()) != outcome.pred_label:
                    raise RuntimeError("classical prediction differs from scored result")
                confidence = float(probabilities[outcome.pred_label].item())
                if not np.isfinite(confidence):
                    raise RuntimeError("non-finite classical classifier output")
                output = io.BytesIO()
                Image.fromarray(result.decoded_image, mode="RGB").save(output, format="PNG")
                image_url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
            record = {
                "status": result.verdict, "predicted_label": CLASS_NAMES[outcome.pred_label],
                "confidence": confidence, "image_url": image_url,
                "detail": (
                    f"Frozen adaptive {modulation} / LDPC {ldpc_rate} / axis {encode_axis}. "
                    + ("BR-13 constant-class outage prediction; no image decoded. " if result.verdict != DELIVERED else "Artifact-finetuned scorer, uncalibrated softmax. ")
                    + f"{round((time.perf_counter() - start) * 1000, 2)} ms compute."
                ),
                "noise_id": result.noise_id, "checkpoint_sha256": self.checkpoint_id if result.verdict == DELIVERED else None,
                "inference_ms": round((time.perf_counter() - start) * 1000, 2), "cache_hit": False,
            }
            self._cache[key] = record
            if len(self._cache) > 64:
                self._cache.popitem(last=False)
            return record


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    image_id: str = Field(min_length=16, max_length=16, pattern=r"^[0-9a-f]{16}$")
    snr_db: int
    ratio: str


def create_app(*, evidence: Evidence | None = None) -> FastAPI:
    evidence = evidence or Evidence()
    app = FastAPI(title="Capstone validation demo", version="1.0.0", docs_url=None, redoc_url=None)

    @app.get("/api/metadata")
    def metadata() -> dict[str, Any]:
        return {
            "dataset": "imagenette160", "split": "val", "test": "SEALED", "test_access": 0,
            "channel": "simulated AWGN", "snr_grid_db": evidence.grid, "default_snr_db": -8,
            "ratios": [
                {"id": ratio, "label": ratio.replace("r_1_", "1/"),
                 "channel_uses": int(get(f"bandwidth.k_symbols.imagenette160.{ratio}"))}
                for ratio in ("r_1_6", "r_1_24")
            ],
            "evidence_label": "W10 v11 published validation · test sealed",
            "systems": [{"system": system, "bw_ratio": ratio} for system, ratio in evidence.curves],
            "image_count": len(evidence.images), "source": "W10 v11 published validation closeout",
            "closeout_id": evidence.closeout_id, "units_sha256": evidence.units_sha256,
            "inference_available": evidence.live.available or evidence.classical.available,
            "inference_reason": None if evidence.live.available and evidence.classical.available else "One or more frozen inference checkpoints are unavailable; each arm reports its own status.",
            "live_inference_arms": [
                {"system": system, "ratio": "r_1_6"}
                for system, available in (("learned", evidence.live.available), ("classical_adaptive", evidence.classical.available))
                if available
            ],
            "classical_inference_available": evidence.classical.available,
            "portable_example_count": len(evidence.examples),
            "portable_examples_split": "train",
        }

    @app.get("/api/chart")
    def chart(ratio: str = Query(default="r_1_6")) -> dict[str, Any]:
        if ratio not in ("r_1_6", "r_1_24"):
            raise HTTPException(status_code=422, detail="ratio outside frozen W10 chart")
        names = {
            "learned": "Semantic DJSCC", "classical_adaptive": "Classical adaptive (artifact scorer)",
            "er9_digital": "Task-aware digital", "classical_fixed_mcs": "Classical fixed MCS",
        }
        names["learned_snr_randomised"] = "Randomized-SNR learned"
        ordered = ["learned", "classical_adaptive", "learned_snr_randomised"]
        colors = {"learned": "#0072B2", "classical_adaptive": "#D55E00", "learned_snr_randomised": "#009E73"}
        series = [
            {"id": system, "label": names[system], "color": colors[system], "points": [
                {"snr_db": p["snr_db"], "accuracy": p["accuracy"], "coverage": p["coverage_rate"]}
                for p in evidence.curves[(system, ratio)]
            ]}
            for system in ordered if (system, ratio) in evidence.curves
        ]
        return {
            "ratio": ratio, "series": series, "split": "val",
            "metric": "top-1 accuracy = n_correct / n_total", "source": evidence.closeout_id,
            "test": "SEALED",
        }

    @app.get("/api/images")
    def images(split: Literal["train"] = "train") -> dict[str, Any]:
        # Portable gallery intentionally uses training images. Frozen aggregate
        # chart remains validation-only; never call train examples held-out.
        return {"split": "train", "images": [
            {"id": image_id, "label": row["label"], "truth_label": row["label"],
             "thumbnail_url": f"/api/examples/{row['id']}", "example_split": "train"}
            for image_id, row in evidence.examples.items()
        ]}

    @app.get("/api/images/{image_id}")
    def image(image_id: str) -> Response:
        if len(image_id) != 16 or any(c not in "0123456789abcdef" for c in image_id):
            raise HTTPException(status_code=404, detail="unknown validation image")
        try:
            png = evidence.image_png(image_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown validation image") from None
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/examples")
    def examples() -> dict[str, Any]:
        return {
            "split": "train", "purpose": "portable illustration inputs, not held-out evaluation",
            "images": [
                {"image_id": image_id, "name": row["id"], "label": row["label"],
                 "label_index": row["label_index"], "split": "train", "url": f"/api/examples/{row['id']}"}
                for image_id, row in evidence.examples.items()
            ],
        }

    @app.get("/api/examples/{name}")
    def example(name: str) -> Response:
        image_id = next((image_id for image_id, row in evidence.examples.items() if row["id"] == name), None)
        if image_id is None:
            raise HTTPException(status_code=404, detail="unknown bundled training example")
        return Response(content=evidence.image_png(image_id), media_type="image/png", headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})

    @app.post("/api/infer")
    def infer(request: InferRequest) -> dict[str, Any]:
        if request.image_id not in evidence.image_indices and request.image_id not in evidence.examples:
            raise HTTPException(status_code=404, detail="unknown approved image")
        if float(request.snr_db) not in evidence.grid or request.ratio not in {"r_1_6", "r_1_24"}:
            raise HTTPException(status_code=422, detail="SNR or bandwidth ratio outside frozen W10 grid")
        learned: dict[str, Any] = {
            "status": "unavailable", "predicted_label": None, "confidence": None, "image_url": None,
            "detail": "Only r_1_6 frozen learned checkpoint is provisioned."
            if request.ratio != "r_1_6" else "Frozen learned checkpoint is absent or fails SHA-256."
        }
        if request.ratio == "r_1_6" and evidence.live.available:
            try:
                live = evidence.live.infer(evidence.product(request.image_id), request.snr_db)
                learned = {
                    "status": "delivered", "predicted_label": CLASS_NAMES[live["label_index"]],
                    "confidence": live["confidence"], "image_url": live["reconstruction_png_data_url"],
                    "detail": "Fresh CPU W8 inference · uncalibrated softmax; reconstruction is visualization only."
                    + (" Cached result." if live["cache_hit"] else f" {live['inference_ms']} ms compute."),
                    "checkpoint_sha256": live["checkpoint_sha256"], "noise_id": live["noise_id"],
                    "cache_hit": live["cache_hit"], "inference_ms": live["inference_ms"],
                }
            except Exception as exc:
                learned["detail"] = f"Learned inference failed closed ({type(exc).__name__}); no prediction was produced."
        classical = {
            "status": "unavailable",
            "predicted_label": None, "confidence": None, "image_url": None,
            "detail": "Frozen artifact scorer is unavailable." if request.ratio == "r_1_6" else "Only r_1_6 adaptive classical live path is enabled.",
        }
        split = "train" if request.image_id in evidence.examples else "val"
        if request.ratio == "r_1_6" and evidence.classical.available:
            try:
                classical = evidence.classical.infer(
                    evidence.product(request.image_id),
                    label=evidence.examples[request.image_id]["label_index"] if split == "train" else evidence.images[evidence.image_indices[request.image_id]]["label_index"],
                    snr_db=request.snr_db,
                )
            except Exception as exc:
                classical["detail"] = f"Classical inference failed closed ({type(exc).__name__}); no prediction was produced."
        return {
            "split": split, "image_id": request.image_id,
            "snr_db": request.snr_db, "ratio": request.ratio,
            "input_image_url": f"/api/examples/{evidence.examples[request.image_id]['id']}" if split == "train" else f"/api/images/{request.image_id}",
            "classical": classical, "learned": learned,
            "inference_mode": "fresh_cpu_demo_not_published_per_image_evidence",
        }

    return app


app = create_app()
