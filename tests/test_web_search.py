"""Tests for BFCL's WebSearchAPI, which is backed by the Brave Search API.

The tool contract the model sees is frozen (the docstring of `search_engine_query`
is compiled into data/multi_turn_func_doc/web_search.json), so these tests pin the
two things a backend swap can silently break: the DuckDuckGo -> Brave region
translation, and the shape of what the tool hands back to the agent loop.

No network: `requests.get` is monkeypatched throughout.
"""

import itertools
import time

import pytest

from domyn_bfcl import web_search
from domyn_bfcl.web_search import WebSearchAPI, _brave_region_params


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


def brave_payload(*results):
    return {"web": {"results": list(results)}}


@pytest.fixture
def api():
    instance = WebSearchAPI()
    instance._load_scenario({"show_snippet": True})
    return instance


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    """Reset the module-global pacing clock so tests don't wait on each other."""
    monkeypatch.setattr(web_search, "_brave_next_allowed", 0.0)
    monkeypatch.setattr(web_search, "_BRAVE_MIN_INTERVAL", 0.0)


def capture_get(monkeypatch, *responses):
    """Patch requests.get to return `responses` in order; record the calls."""
    calls = []
    queue = list(responses)

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append({"url": url, "headers": headers, "params": dict(params)})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(web_search.requests, "get", fake_get)
    return calls


# --- region translation -------------------------------------------------------


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        ("wt-wt", {}),  # "No region"
        ("us-en", {"country": "US", "search_lang": "en"}),
        ("uk-en", {"country": "GB", "search_lang": "en"}),  # DDG uk -> ISO GB
        ("sl-sl", {"country": "SI", "search_lang": "sl"}),  # Slovenia, not Sierra Leone
        ("ue-es", {"country": "US", "search_lang": "es"}),
        ("hk-tzh", {"country": "HK", "search_lang": "zh"}),
        ("hk-tz", {"country": "HK", "search_lang": "zh"}),  # schema's typo
        ("kr-kr", {"country": "KR", "search_lang": "ko"}),
        ("jp-jp", {"country": "JP", "search_lang": "jp"}),  # Brave uses jp, not ja
        ("xa-ar", {"search_lang": "ar"}),  # pseudo-region
        ("ct-ca", {"search_lang": "ca"}),  # Catalan is a language
        ("garbage", {}),
        ("a-b-c", {}),
        (None, {}),
    ],
)
def test_region_params(region, expected):
    assert _brave_region_params(region) == expected


# --- result shape -------------------------------------------------------------


def test_results_strip_brave_markup(api, monkeypatch):
    capture_get(
        monkeypatch,
        FakeResponse(
            payload=brave_payload(
                {
                    "title": "Best <strong>Greek</strong> Food",
                    "url": "https://example.com/a",
                    "description": "A <strong>Greek</strong> caf&eacute;.",
                }
            )
        ),
    )
    results = api.search_engine_query("greek food")
    assert results == [
        {
            "title": "Best Greek Food",
            "href": "https://example.com/a",
            "body": "A Greek café.",
        }
    ]


def test_no_snippet_split_drops_body(api, monkeypatch):
    capture_get(
        monkeypatch,
        FakeResponse(payload=brave_payload({"title": "T", "url": "https://example.com/a", "description": "D"})),
    )
    api._load_scenario({"show_snippet": False})
    assert api.search_engine_query("q") == [{"title": "T", "href": "https://example.com/a"}]


def test_missing_fields_do_not_raise(api, monkeypatch):
    """The SerpAPI code indexed result["link"]/["snippet"] and would KeyError."""
    capture_get(monkeypatch, FakeResponse(payload=brave_payload({})))
    assert api.search_engine_query("q") == [{"title": "", "href": "", "body": ""}]


def test_absent_web_key_returns_legacy_error(api, monkeypatch):
    """Brave omits `web` when there are no results; SerpAPI omitted organic_results."""
    capture_get(monkeypatch, FakeResponse(payload={"query": {"original": "q"}}))
    assert api.search_engine_query("q") == {
        "error": "Failed to retrieve the search results from server. Please try again later."
    }


# --- request construction -----------------------------------------------------


def test_count_is_clamped_to_brave_maximum(api, monkeypatch):
    calls = capture_get(
        monkeypatch,
        FakeResponse(payload=brave_payload(*({"title": str(i)} for i in range(20)))),
    )
    results = api.search_engine_query("q", max_results=50)
    assert calls[0]["params"]["count"] == web_search.BRAVE_MAX_COUNT
    assert len(results) <= web_search.BRAVE_MAX_COUNT


def test_api_key_is_sent_as_subscription_token(api, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    calls = capture_get(monkeypatch, FakeResponse(payload=brave_payload({})))
    api.search_engine_query("q")
    assert calls[0]["headers"]["X-Subscription-Token"] == "test-key"
    assert calls[0]["url"] == web_search.BRAVE_ENDPOINT


# --- retry behaviour ----------------------------------------------------------


def test_429_retries_and_honours_reset_header(api, monkeypatch):
    slept = []
    monkeypatch.setattr(web_search.time, "sleep", slept.append)
    capture_get(
        monkeypatch,
        FakeResponse(status_code=429, headers={"X-RateLimit-Reset": "3, 1419704"}),
        FakeResponse(payload=brave_payload({"title": "T", "url": "u"})),
    )
    results = api.search_engine_query("q")
    assert results[0]["title"] == "T"
    # Brave's per-second window is the first entry, and it beats the 2-4s backoff.
    # Exactly one sleep: the retry waits on the shared clock, not on top of it.
    assert slept == pytest.approx([3.0], abs=0.05)


def test_rejected_locale_retries_unlocalized(api, monkeypatch):
    """Brave rejects locales the DuckDuckGo enum allows; that must not fail the turn."""
    calls = capture_get(
        monkeypatch,
        FakeResponse(status_code=422, text="invalid country"),
        FakeResponse(payload=brave_payload({"title": "T", "url": "u"})),
    )
    results = api.search_engine_query("q", region="gr-el")
    assert isinstance(results, list)
    assert results[0]["title"] == "T"
    assert calls[0]["params"]["country"] == "GR"
    assert "country" not in calls[1]["params"]
    assert "search_lang" not in calls[1]["params"]


def test_non_retryable_http_error_returns_error_dict(api, monkeypatch):
    capture_get(monkeypatch, FakeResponse(status_code=401, text="unauthorized"))
    result = api.search_engine_query("q")
    assert "401" in result["error"]


def test_quota_exhaustion_gives_up_instead_of_hanging(api, monkeypatch):
    """A dead monthly quota 429s forever; the run must not park on it."""
    monkeypatch.setattr(web_search.time, "sleep", lambda s: None)
    capture_get(
        monkeypatch,
        FakeResponse(status_code=429, headers={"X-RateLimit-Reset": "600"}),
    )
    assert api.search_engine_query("q") == {"error": "Brave Search rate limit / quota exhausted."}


# --- throttle -----------------------------------------------------------------


def test_throttle_spaces_out_concurrent_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(web_search, "_BRAVE_MIN_INTERVAL", 0.05)
    monkeypatch.setattr(web_search, "_brave_next_allowed", 0.0)

    stamps = []

    def fake_get(url, headers=None, params=None, timeout=None):
        stamps.append(time.monotonic())
        return FakeResponse(payload=brave_payload({"title": "T", "url": "u"}))

    monkeypatch.setattr(web_search.requests, "get", fake_get)

    api = WebSearchAPI()
    api._load_scenario({"show_snippet": True})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: api.search_engine_query(f"q{i}"), range(8)))

    stamps.sort()
    gaps = [b - a for a, b in itertools.pairwise(stamps)]
    assert all(g >= 0.04 for g in gaps), gaps
