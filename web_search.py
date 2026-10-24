"""Optional external search. PDF-only mode works when no search key is set."""

import json
import os
from urllib import error, request

from dotenv import load_dotenv

TAVILY_URL = "https://api.tavily.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def _key(name):
    load_dotenv()
    value = (os.getenv(name) or "").strip()
    if not value or value == "your-key-here":
        return ""
    return value


def search_status():
    """Which provider is configured, without returning secret values."""
    if _key("TAVILY_API_KEY"):
        return {"available": True, "provider": "tavily"}
    if _key("BRAVE_SEARCH_API_KEY"):
        return {"available": True, "provider": "brave"}
    return {
        "available": False,
        "provider": None,
        "reason": "Set TAVILY_API_KEY or BRAVE_SEARCH_API_KEY in .env to enable web search.",
    }


def search_web(query, max_results=3):
    """Return external snippets. Unavailable credentials are not an exception."""
    status = search_status()
    if not status["available"]:
        return {"available": False, "provider": None, "results": [], "reason": status["reason"]}

    try:
        if status["provider"] == "tavily":
            results = _tavily(query, max_results)
        else:
            results = _brave(query, max_results)
    except (error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
        return {
            "available": True,
            "provider": status["provider"],
            "results": [],
            "reason": f"Web search failed: {exc.__class__.__name__}",
        }

    return {
        "available": True,
        "provider": status["provider"],
        "results": results,
        "reason": "",
    }


def _post_json(url, payload, headers):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url, headers):
    req = request.Request(url, headers=headers, method="GET")
    with request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _tavily(query, max_results):
    data = _post_json(
        TAVILY_URL,
        {"api_key": _key("TAVILY_API_KEY"), "query": query, "max_results": max_results},
        {"Content-Type": "application/json"},
    )
    results = []
    for item in data.get("results", [])[:max_results]:
        url = item.get("url") or ""
        if not url:
            continue
        results.append({
            "title": item.get("title") or url,
            "url": url,
            "snippet": (item.get("content") or "")[:800],
        })
    return results


def _brave(query, max_results):
    from urllib.parse import urlencode

    url = BRAVE_URL + "?" + urlencode({"q": query, "count": max_results})
    data = _get_json(url, {
        "Accept": "application/json",
        "X-Subscription-Token": _key("BRAVE_SEARCH_API_KEY"),
    })
    results = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        link = item.get("url") or ""
        if not link:
            continue
        results.append({
            "title": item.get("title") or link,
            "url": link,
            "snippet": (item.get("description") or "")[:800],
        })
    return results
