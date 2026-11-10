# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Brave Search behind BFCL's ``web_search`` tool, in place of upstream's SerpAPI.

Divergence from upstream ``bfcl_eval...func_source_code.web_search``: only
``search_engine_query`` differs. It queries Brave (``$BRAVE_API_KEY``), with a request
rate shared across threads (``$BRAVE_MIN_REQUEST_INTERVAL``, default 0.11 s), backoff
on 429 capped at 900 s in total (upstream retries forever), and the tool's DuckDuckGo
``region`` mapped to Brave's ``country``/``search_lang``. Results keep upstream's
``title``/``href``/``body`` shape. ``fetch_url_content`` is upstream's.

``domyn_bfcl.install`` points ``CLASS_FILE_PATH_MAPPING["WebSearchAPI"]`` here; the
multi-turn executor looks the class up by that name in this module.
"""

import os
import random
import threading
import time

import requests
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import (
    WebSearchAPI as UpstreamWebSearchAPI,
)
from bs4 import BeautifulSoup

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_MAX_COUNT = 20  # Brave caps `count` at 20 results per request
BRAVE_MAX_BACKOFF = 120
BRAVE_MAX_RATE_LIMIT_WAIT = 900

try:
    _BRAVE_MIN_INTERVAL = float(os.getenv("BRAVE_MIN_REQUEST_INTERVAL") or 0.11)
except ValueError:
    _BRAVE_MIN_INTERVAL = 0.11
_brave_lock = threading.Lock()
_brave_next_allowed = 0.0

_DDG_COUNTRY_FIXUPS = {
    "uk": "GB",  # DDG says `uk`; ISO 3166 is GB
    "ue": "US",  # DDG's "United States (es)"
    "sl": "SI",  # `sl-sl` is Slovenia; a bare ISO `SL` would be Sierra Leone
    "wt": None,  # "No region"
    "xa": None,  # pseudo-region: Arabia
    "xl": None,  # pseudo-region: Latin America
    "ct": None,  # pseudo-region: Catalan is a language, not a country
}
_DDG_LANG_FIXUPS = {
    "tzh": "zh",  # DDG traditional Chinese (hk-tzh, tw-tzh)
    "tz": "zh",  # the compiled schema's `hk-tz` typo
    "kr": "ko",  # `kr-kr` is Korean; ISO 639-1 is ko
    "jp": "jp",  # identity on purpose: Brave documents `jp`, not the ISO `ja`
    "wt": None,
}


def _brave_throttle() -> None:
    """Block until this thread is allowed to issue the next Brave request."""
    global _brave_next_allowed
    with _brave_lock:
        wait = _brave_next_allowed - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _brave_next_allowed = time.monotonic() + _BRAVE_MIN_INTERVAL


def _brave_defer_all(seconds: float) -> None:
    """Push the shared clock forward so every thread backs off, not just this one.

    Without this, one thread absorbing a 429 while the others keep firing at the
    nominal rate just produces more 429s.
    """
    global _brave_next_allowed
    with _brave_lock:
        _brave_next_allowed = max(_brave_next_allowed, time.monotonic() + seconds)


def _brave_retry_delay(response: requests.Response, backoff: float) -> float:
    """Seconds to wait after a 429, preferring Brave's own hint.

    Brave sends one value per quota window in `X-RateLimit-Reset` (e.g. "1, 1419704"
    for the per-second and per-month policies); the first entry is the per-second
    window.
    """
    raw = response.headers.get("Retry-After") or response.headers.get("X-RateLimit-Reset")
    try:
        seconds = float((raw or "").split(",")[0].strip())
    except (AttributeError, ValueError):
        seconds = 0.0
    if seconds <= 0:
        seconds = backoff + random.uniform(0, backoff)
    return min(seconds, BRAVE_MAX_BACKOFF)


def _strip_html(text: object) -> str:
    """Brave wraps matched query terms in <strong>; SerpAPI snippets were plain."""
    if not isinstance(text, str):
        return ""
    if "<" not in text and "&" not in text:
        return text
    return BeautifulSoup(text, "html.parser").get_text()


def _brave_region_params(region: object) -> dict[str, str]:
    """Best-effort DDG `<region>-<language>` -> Brave `country` / `search_lang`.

    Anything that does not resolve to a 2-letter code is dropped: Brave rejects
    unknown values rather than ignoring them, and an omitted parameter means
    "unlocalized", which is what the default `wt-wt` ("No region") asks for.
    """
    if not isinstance(region, str):
        return {}
    parts = region.strip().lower().split("-")
    if len(parts) != 2:
        return {}
    country_code, lang_code = parts

    params: dict[str, str] = {}
    country = _DDG_COUNTRY_FIXUPS.get(country_code, country_code.upper())
    if country and len(country) == 2 and country.isalpha():
        params["country"] = country
    lang = _DDG_LANG_FIXUPS.get(lang_code, lang_code)
    if lang and len(lang) == 2 and lang.isalpha():
        params["search_lang"] = lang
    return params


def _print_error_block(message: str) -> None:
    """Print an error in upstream's banner."""
    print("*" * 100 + f"\n❗️❗️ [WebSearchAPI] {message}" + "*" * 100)


class WebSearchAPI(UpstreamWebSearchAPI):
    """Upstream's ``WebSearchAPI``, searching with Brave."""

    # Upstream annotates `int | None` and `-> list` yet returns error dicts too; C901: one retry loop
    def search_engine_query(  # pyright: ignore[reportIncompatibleMethodOverride]  # noqa: C901
        self,
        keywords: str,
        max_results: int = 10,
        region: str | None = "wt-wt",
    ) -> list | dict:
        """Upstream's tool (same arguments, same result shape), answered by Brave Search."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": os.getenv("BRAVE_API_KEY", ""),
        }
        # Brave caps `count` at 20 per request and we deliberately do not paginate:
        # extra pages cost rate-limit budget for results the model rarely reads.
        try:
            count = max(1, min(int(max_results), BRAVE_MAX_COUNT))
        except (TypeError, ValueError):
            count = 10
        params = {"q": keywords, "count": count}
        params.update(_brave_region_params(region))

        backoff = 2  # initial back-off in seconds
        waited = 0.0

        # Infinite retry loop with exponential backoff
        while True:
            _brave_throttle()
            try:
                response = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=30)
            except requests.RequestException as e:
                _print_error_block(
                    f"Error from Brave Search: {e!s}. This is not a rate-limit error, so it will not be retried."
                )
                return {"error": str(e)}

            if response.status_code == 429:
                wait_time = _brave_retry_delay(response, backoff)
                if waited + wait_time > BRAVE_MAX_RATE_LIMIT_WAIT:
                    message = "Brave Search rate limit / quota exhausted."
                    _print_error_block(f"{message} Giving up after {waited:.0f}s of rate-limit waiting.")
                    return {"error": message}
                # Defer every thread, not just this one: the retry itself then waits
                # on the shared clock via _brave_throttle(), so the backoff is
                # counted once rather than slept twice.
                _brave_defer_all(wait_time)
                _print_error_block(
                    "Received 429 from Brave Search. The request rate exceeds the plan limit OR the subscription "
                    f"quota is exhausted. Retrying in {wait_time:.1f} seconds…"
                )
                waited += wait_time
                backoff = min(backoff * 2, BRAVE_MAX_BACKOFF)
                continue

            if response.status_code >= 400 and ("country" in params or "search_lang" in params):
                # Brave supports far fewer locales than the DuckDuckGo enum in this
                # tool's schema, and rejects unknown ones rather than ignoring them.
                # Both params must go: while either remains the condition above stays
                # true and the retry spins forever on a persistent 4xx.
                params.pop("country", None)
                params.pop("search_lang", None)
                continue

            if response.status_code >= 400:
                message = f"Brave Search returned HTTP {response.status_code}: {response.text[:500]}"
                _print_error_block(f"{message}. This is not a rate-limit error, so it will not be retried.")
                return {"error": message}

            try:
                payload = response.json()
            except ValueError as e:
                return {"error": f"Malformed response from Brave Search: {e!s}"}

            break  # Success - no rate-limit error detected

        web_block = payload.get("web") if isinstance(payload, dict) else None
        search_results = web_block.get("results") if isinstance(web_block, dict) else None
        if not isinstance(search_results, list):
            return {"error": "Failed to retrieve the search results from server. Please try again later."}

        # Convert the search results to the desired format
        results = []
        for result in search_results[:count]:
            if not isinstance(result, dict):
                continue
            if self.show_snippet:
                results.append(
                    {
                        "title": _strip_html(result.get("title", "")),
                        "href": result.get("url", ""),
                        "body": _strip_html(result.get("description", "")),
                    }
                )
            else:
                results.append(
                    {
                        "title": _strip_html(result.get("title", "")),
                        "href": result.get("url", ""),
                    }
                )

        return results
