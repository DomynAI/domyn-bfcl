# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Generation-loop additions, applied around upstream ``collect_test_cases``.

- Resume regenerates failed entries: a result recorded as ``"Error during inference:
  ..."`` is removed from the result file so it is generated again (upstream keeps it,
  and it scores as a wrong answer, or as a correct abstention in irrelevance).
- ``max_samples > 0`` (``domyn_bfcl.config``) keeps the first N cases of each test
  category (upstream has no sampling; the driver then scores with ``--partial-eval``).
"""

from argparse import Namespace
from collections import defaultdict

from bfcl_eval.utils import (
    extract_test_category_from_id,
    get_directory_structure_by_category,
    get_file_name_by_category,
    is_memory,
    load_file,
    write_list_of_dicts_to_file,
)
from tqdm import tqdm


def is_inference_error(result: object) -> bool:
    """Whether generation failed for this entry: the result is the error text written by `multi_threaded_inference`."""
    return isinstance(result, str) and result.startswith("Error during inference")


def purge_failed_results(args: Namespace, model_name: str, all_test_categories: list[str]) -> None:
    """Drop failed entries from the existing result files, so they are generated again.

    Mirrors how upstream ``collect_test_cases`` locates the files it resumes from; only
    resumes (no ``--allow-overwrite``) are affected.
    """
    if args.allow_overwrite:
        return
    model_result_dir = args.result_dir / model_name.replace("/", "_")
    for test_category in all_test_categories:
        categories = [test_category]
        if is_memory(test_category):
            # Memory test cases have the pre-requisite entries in a separate file
            categories.append(f"{test_category}_prereq")
        for category in categories:
            file_path = (
                model_result_dir
                / get_directory_structure_by_category(test_category)
                / get_file_name_by_category(category, is_result_file=True)
            )
            if not file_path.exists():
                continue
            entries = load_file(file_path)
            kept = [entry for entry in entries if not is_inference_error(entry.get("result"))]
            if len(kept) < len(entries):
                write_list_of_dicts_to_file(file_path, kept)
                tqdm.write(f"Regenerating {len(entries) - len(kept)} failed entries in {file_path.name}")


def subsample_test_cases(test_cases_total: list[dict], max_samples: int) -> list[dict]:
    """Keep the first `max_samples` cases of *each* test category.

    A flat `test_cases_total[:max_samples]` slices a list that is ordered by
    category, so a small cap starves every category but the first and leaves the
    rest unscored (reported as N/A / 0%) — useless for a smoke run that is meant
    to touch each category. Entries required via `depends_on` (memory
    prerequisites) are pulled back in so the dependency graph built downstream
    cannot deadlock, and any dependency that still did not survive is dropped
    from the list for the same reason.

    Memory prerequisites are bucketed under their own `*_prereq` category rather
    than merged into the category they support: prerequisites sort first, so a
    merged bucket would spend the whole budget on them and score no question.
    """
    by_category: defaultdict[str, list[dict]] = defaultdict(list)
    for test_case in test_cases_total:
        category = extract_test_category_from_id(test_case["id"])
        by_category[category].append(test_case)

    kept_ids: set[str] = set()
    for cases in by_category.values():
        kept_ids.update(case["id"] for case in cases[:max_samples])

    # Close over depends_on: a kept case whose prerequisite was cut would block
    # forever in the ready-queue, so re-add the prerequisites it needs.
    by_id = {test_case["id"]: test_case for test_case in test_cases_total}
    pending = list(kept_ids)
    while pending:
        test_case = by_id.get(pending.pop())
        if test_case is None:
            continue
        for dependency_id in test_case.get("depends_on", []):
            if dependency_id in by_id and dependency_id not in kept_ids:
                kept_ids.add(dependency_id)
                pending.append(dependency_id)

    subsampled = [tc for tc in test_cases_total if tc["id"] in kept_ids]
    for test_case in subsampled:
        if "depends_on" in test_case:
            test_case["depends_on"] = [
                dependency_id for dependency_id in test_case["depends_on"] if dependency_id in kept_ids
            ]
    return subsampled
