"""domyn_bfcl.generation: failed-entry purge and per-category subsampling."""

import json
from types import SimpleNamespace

from bfcl_eval.utils import get_directory_structure_by_category, get_file_name_by_category, load_file

from domyn_bfcl.generation import is_inference_error, purge_failed_results, subsample_test_cases


def test_is_inference_error():
    assert is_inference_error("Error during inference: timeout")
    assert not is_inference_error("[f(a=1)]")
    assert not is_inference_error([{"f": {}}])


def test_subsample_keeps_first_n_per_category():
    cases = [{"id": f"{cat}_{i}"} for cat in ("simple_python", "multiple") for i in range(3)]
    kept = subsample_test_cases(cases, 2)
    assert [c["id"] for c in kept] == ["simple_python_0", "simple_python_1", "multiple_0", "multiple_1"]


def test_subsample_pulls_back_prerequisites():
    cases = [
        {"id": "memory_kv_prereq_0-customer-0"},
        {"id": "memory_kv_prereq_1-customer-1"},
        {"id": "memory_kv_0-customer-0", "depends_on": ["memory_kv_prereq_1-customer-1", "gone"]},
    ]
    kept = subsample_test_cases(cases, 1)
    assert {c["id"] for c in kept} == {
        "memory_kv_prereq_0-customer-0",
        "memory_kv_prereq_1-customer-1",
        "memory_kv_0-customer-0",
    }
    assert kept[-1]["depends_on"] == ["memory_kv_prereq_1-customer-1"]


def _write_results(tmp_path, category, entries):
    path = (
        tmp_path
        / "m"
        / get_directory_structure_by_category(category)
        / get_file_name_by_category(category, is_result_file=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return path


def test_purge_drops_failed_entries(tmp_path):
    path = _write_results(
        tmp_path,
        "simple_python",
        [{"id": "simple_python_0", "result": "ok"}, {"id": "simple_python_1", "result": "Error during inference: x"}],
    )
    purge_failed_results(SimpleNamespace(result_dir=tmp_path, allow_overwrite=False), "m", ["simple_python"])
    assert [e["id"] for e in load_file(path)] == ["simple_python_0"]


def test_purge_leaves_overwrite_runs_to_upstream(tmp_path):
    entries = [{"id": "simple_python_1", "result": "Error during inference: x"}]
    path = _write_results(tmp_path, "simple_python", entries)
    purge_failed_results(SimpleNamespace(result_dir=tmp_path, allow_overwrite=True), "m", ["simple_python"])
    assert load_file(path) == entries
