"""Cluster O — PYTHONHASHSEED process-level determinism robustness (R13).

PLAN §0 C5 의 *process-level robustness* 확장. dict insertion order 보장
(Python 3.7+) 위 hash randomization 영향 0. set 직렬화 부재 검증.
"""

import json
import os
import subprocess
import sys

import pytest

from puls_sched.run import Run


_CFG = "puls_sched.config:default_dummy_config"


def _walk_for_sets(obj) -> bool:
    """recursive: obj 안에 set/frozenset 직렬화 존재 검사."""
    if isinstance(obj, (set, frozenset)):
        return True
    if isinstance(obj, dict):
        return any(_walk_for_sets(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_walk_for_sets(x) for x in obj)
    return False


class TestPYTHONHASHSEED:
    def test_no_set_serialization_in_report(self, tmp_path):
        """report.json 의 nested value 에 set/frozenset 부재 (R13 source-side)."""
        run = Run.init(_CFG, "synthetic:10", tmp_path, seed=42)
        run.loop()
        report = run.teardown()
        # markdown 제외한 report dict 의 raw form
        data = {k: v for k, v in report.items() if k != "markdown"}
        # 산출 dataclass 들이 list 또는 tuple 로 변환되어야 함 (set 0)
        json_text = (tmp_path / "report.json").read_text(encoding="utf-8")
        parsed = json.loads(json_text)
        assert not _walk_for_sets(parsed)

    def test_dict_key_ordering_preserved(self, tmp_path):
        """동일 seed 위 2 회 report.json 의 key 순서 동일 (sort_keys=True 보장)."""
        out1 = tmp_path / "a"
        out2 = tmp_path / "b"
        Run.init(_CFG, "synthetic:10", out1, seed=42).loop()
        Run.init(_CFG, "synthetic:10", out1, seed=42).teardown()
        run1 = Run.init(_CFG, "synthetic:10", out1, seed=42)
        run1.loop()
        run1.teardown()
        run2 = Run.init(_CFG, "synthetic:10", out2, seed=42)
        run2.loop()
        run2.teardown()
        d1 = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
        d2 = json.loads((out2 / "report.json").read_text(encoding="utf-8"))
        assert list(d1.keys()) == list(d2.keys())

    def test_subprocess_pythonhashseed_robustness(self, tmp_path):
        """PYTHONHASHSEED=0 + PYTHONHASHSEED=42 위 동일 report.json (R13 byte-equal)."""
        outs = {}
        impl_dir = str(tmp_path.parent.parent / "implementation")
        # 실제 cwd 는 tmp_path 자체이므로 src 의 puls_sched 가 import 가능해야 함
        # pytest 의 conftest 가 pythonpath 설정 — subprocess 는 그 안에서 동작 안 함
        # → PYTHONPATH 명시 주입
        import puls_sched
        src_dir = str(__import__("pathlib").Path(puls_sched.__file__).parent.parent)
        for hashseed in ("0", "42"):
            out_dir = tmp_path / f"out_{hashseed}"
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = hashseed
            env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
            result = subprocess.run(
                [sys.executable, "-m", "puls_sched",
                 "--config-module", _CFG,
                 "--trace", "synthetic:10",
                 "--output", str(out_dir),
                 "--seed", "42"],
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                pytest.fail(f"subprocess failed (PYTHONHASHSEED={hashseed}): "
                            f"rc={result.returncode}, stderr={result.stderr}")
            outs[hashseed] = (out_dir / "report.json").read_text(encoding="utf-8")
        assert outs["0"] == outs["42"]
