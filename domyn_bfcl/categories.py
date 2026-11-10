# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Test collections added to upstream's ``TEST_COLLECTION_MAPPING`` (no score effect)."""

from bfcl_eval.constants.category_mapping import (
    MEMORY_CATEGORY,
    MULTI_TURN_CATEGORY,
    SINGLE_TURN_CATEGORY,
    WEB_SEARCH_CATEGORY,
)

EXTRA_TEST_COLLECTIONS = {
    "all_scoring_no_mem": SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY + WEB_SEARCH_CATEGORY,
    # all_scoring minus the web_search tasks, which require a BRAVE_API_KEY.
    "all_scoring_no_websearch": SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY + MEMORY_CATEGORY,
    "single_multi": SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY,
    # for a quick general overview
    "core": [
        "simple_python",  # baseline single-turn accuracy
        "multiple",  # choose among several functions
        "parallel",  # call multiple functions at once
        "parallel_multiple",  # both challenges combined — hardest single-turn
        "irrelevance",  # knows when NOT to call a function (critical for agents)
    ],
}
