"""Live market quotes for the methanol production plan.

Posted methanol prices are read from Methanex when the request runs.
Gas, gold, crude, the euro and Methanex shares come from the Yahoo chart
feed. The free-market dollar and ounce gold in rials come from TGJU.
Zagros and Fanavaran closes come from TSETMC. A failed source stays empty.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_MMBTU_PER_GJ = 1 / 1.055
_CACHE_S = 180
_FAIL_CACHE_S = 30
_LOCK = threading.Lock()
_CACHE: dict = {}
_UA = {"User-Agent": "Mozilla/5.0"}

# Capacities published by CCFGroup on 20 Mar 2026. Not a live operating rate.
_IRAN_CAPACITY = [
    {"name": "زاگرس", "where": "عسلویه", "capacity_mt": 3.3},
    {"name": "کاوه", "where": "دیر", "capacity_mt": 2.3},
    {"name": "فن‌آوران", "where": "بندر امام", "capacity_mt": 1.0},
    {"name": "مرجان", "where": "عسلویه", "capacity_mt": 1.65},
    {"name": "بوشهر", "where": "عسلویه", "capacity_mt": 1.65},
    {"name": "کیمیا", "where": "عسلویه", "capacity_mt": 1.65},
    {"name": "سبلان", "where": "عسلویه", "capacity_mt": 1.65},
    {"name": "آرین", "where": "عسلویه", "capacity_mt": 1.65},
    {"name": "آپادانا", "where": "عسلویه", "capacity_mt": 1.65},
    {"name": "خارک", "where": "خارک", "capacity_mt": 0.66},
]


def _http_json(url: str, timeout: float = 12.0):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _http_text(url: str, timeout: float = 12.0) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _quote(label: str, value, unit: str, change_pct=None, as_of: str | None = None, ok: bool = True) -> dict:
    return {
        "label": label,
        "value": value,
        "unit": unit,
        "change_pct": change_pct,
        "as_of": as_of,
        "ok": ok,
    }


def _yahoo(symbol: str) -> dict:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + urllib.parse.quote(symbol, safe="=")
        + "?interval=1d&range=1d"
    )
    meta = _http_json(url)["chart"]["result"][0]["meta"]
    price = float(meta["regularMarketPrice"])
    change = meta.get("regularMarketChangePercent")
    stamp = meta.get("regularMarketTime")
    return {
        "value": round(price, 4 if price < 20 else 2),
        "change_pct": None if change is None else round(float(change), 2),
        "as_of": None if stamp is None else time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(stamp))) + " UTC",
        "ok": True,
    }


def parse_methanex_html(html: str) -> dict:
    """Europe euro price, then North America, Asia and China dollar prices in page order."""
    euro = re.search(r"Euro\s+([\d,]+)\s*/\s*MT", html, re.I)
    dollars = [int(item.replace(",", "")) for item in re.findall(r"USD\s+([\d,]+)\s*/\s*MT", html, re.I)]
    if euro is None or len(dollars) < 3:
        raise ValueError("Methanex price sheet did not contain the four regional prices")
    return {
        "europe_eur_t": int(euro.group(1).replace(",", "")),
        "north_america_usd_t": dollars[0],
        "asia_usd_t": dollars[1],
        "china_usd_t": dollars[2],
    }


def _methanex() -> dict:
    parsed = parse_methanex_html(_http_text("https://www.methanex.com/our-products/about-methanol/pricing/"))
    parsed["ok"] = True
    parsed["as_of"] = "صفحه قیمت متانکس، همین خواندن"
    return parsed


def _tgju_pair(key: str) -> dict:
    current = _http_json("https://call1.tgju.org/ajax.json")["current"][key]
    value = float(str(current["p"]).replace(",", ""))
    change = current.get("dp")
    return {
        "value": value,
        "change_pct": None if change in (None, "") else round(float(change), 2),
        "as_of": current.get("ts"),
        "ok": True,
    }


def _tsetmc_close(ins_code: str) -> dict:
    info = _http_json(
        "https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceInfo/" + ins_code
    )["closingPriceInfo"]
    close = float(info["pClosing"] or 0)
    if close <= 0:
        raise ValueError("no closing price")
    previous = float(info["priceYesterday"] or 0)
    change = None if previous <= 0 else round((close - previous) / previous * 100, 2)
    return {
        "value": close,
        "change_pct": change,
        "as_of": str(info.get("dEven") or ""),
        "ok": True,
    }


def _marjan_listing() -> dict:
    """Marjan is on Farabourse's third market but has no closing print."""
    try:
        return _tsetmc_close("57714949492552862")
    except Exception:
        info = _http_json(
            "https://cdn.tsetmc.com/api/Instrument/GetInstrumentInfo/57714949492552862"
        )["instrumentInfo"]
        return {
            "ok": False,
            "listed": True,
            "board": info.get("cgrValCotTitle") or info.get("flowTitle") or "",
            "as_of": str(info.get("dEven") or ""),
            "error": "no_print",
        }


def _safe(fn) -> dict:
    last = {"ok": False, "error": "failed"}
    for attempt in range(2):
        try:
            return fn()
        except Exception as exc:
            last = {"ok": False, "error": type(exc).__name__}
            if attempt == 0:
                time.sleep(1.5 if getattr(exc, "code", None) == 429 else 0.4)
    return last


def _cached_quote(key: str, fn) -> dict:
    now = time.monotonic()
    with _LOCK:
        previous = _CACHE.get(key)
        if previous is not None:
            age = now - previous[0]
            fresh_for = _CACHE_S if previous[1].get("ok") else _FAIL_CACHE_S
            if age < fresh_for:
                return previous[1]
    result = _safe(fn)
    with _LOCK:
        previous = _CACHE.get(key)
        if result.get("ok") or previous is None or not previous[1].get("ok"):
            _CACHE[key] = (time.monotonic(), result)
            return result
        return previous[1]


def fetch_market() -> dict:
    """Return quotes. A failed source keeps its last good value, otherwise stays empty."""
    jobs = {
        "gas": lambda: _yahoo("NG=F"),
        "gold": lambda: _yahoo("GC=F"),
        "crude": lambda: _yahoo("CL=F"),
        "euro": lambda: _yahoo("EURUSD=X"),
        "meoh": lambda: _yahoo("MEOH"),
        "methanol": _methanex,
        "dollar": lambda: _tgju_pair("price_dollar_rl"),
        "zagros": lambda: _tsetmc_close("13235547361447092"),
        "fanavaran": lambda: _tsetmc_close("65122215875355555"),
        "marjan": _marjan_listing,
    }
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        fetched = {key: pool.submit(_cached_quote, key, fn) for key, fn in jobs.items()}
        payload = {key: item.result() for key, item in fetched.items()}
    payload["fetched_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    return payload


def build_plan(physical_mt, gap_mt, sec_gj_per_t, market: dict) -> dict:
    """Hold production at the contract-gas cap. Price shocks change revenue, not tonnes."""
    gas = market.get("gas") or {}
    dollar = market.get("dollar") or {}
    gold = market.get("gold") or {}
    methanol = market.get("methanol") or {}
    implied = None
    if gas.get("ok") and sec_gj_per_t:
        implied = round(float(sec_gj_per_t) * _MMBTU_PER_GJ * float(gas["value"]), 1)

    def scenario(sid, title, shock, action, extra=None):
        row = {
            "id": sid,
            "title": title,
            "shock": shock,
            "methanol_mt": physical_mt,
            "action": action,
        }
        if extra:
            row.update(extra)
        return row

    dollar_value = dollar.get("value") if dollar.get("ok") else None
    gold_value = gold.get("value") if gold.get("ok") else None
    gap_txt = "شکاف ظرفیت" if not gap_mt else f"شکاف {gap_mt:.2f} میلیون تن"
    scenarios = [
        scenario(
            "base",
            "برنامه پایه",
            "وضعیت فعلی خوراک و قیمت",
            "تولید را روی متانولی بگذارید که همین گاز قرارداد با انرژی ویژه فعلی می‌سازد، نه روی ۱٫۶۵ میلیون تن.",
        ),
        scenario(
            "dollar_break",
            "شکست دلار",
            "دلار آزاد ۲۰٪ ارزان‌تر",
            "درآمد ریالی هر تن صادراتی کم می‌شود، چون متانول دلاری فروخته می‌شود. تناژ را کم نکنید.",
            {
                "dollar_irr": None if dollar_value is None else round(dollar_value * 0.8),
                "rial_revenue_index": 80,
            },
        ),
        scenario(
            "gold_up",
            "گران شدن طلا",
            "طلا ۲۰٪ گران‌تر",
            "طلا خوراک متانول نیست و برنامه تولید متانکس را هم با طلا تنظیم نمی‌کنند. تناژ عوض نمی‌شود.",
            {"gold_usd_oz": None if gold_value is None else round(gold_value * 1.2, 1)},
        ),
        scenario(
            "demand_down",
            "کاهش تقاضا",
            "خاموشی MTO ساحلی چین",
            f"برای بستن {gap_txt} گاز اضافه نخرید. سقف همین خوراک قرارداد است.",
        ),
        scenario(
            "demand_up",
            "افزایش تقاضا",
            "برگشت تقاضای آسیا",
            "سقف بالا رفتن تولید همین گاز قرارداد است. بار را برای رسیدن به ظرفیت نامی زیاد نکنید.",
        ),
    ]
    return {
        "physical_mt": physical_mt,
        "gap_mt": gap_mt,
        "nameplate_mt": 1.65,
        "sec_gj_per_t": sec_gj_per_t,
        "scenarios": scenarios,
        "feed_compare": {
            "henry_hub_usd_mmbtu": gas.get("value") if gas.get("ok") else None,
            "implied_usd_per_t": implied,
            "is_assaluyeh_tariff": False,
            "note": "این هزینه، هنری‌هاب آمریکا روی انرژی ویژه همین ریفرمر است، نه تعرفه گاز عسلویه.",
        },
        "iran_capacity": _IRAN_CAPACITY,
        "quotes": {
            "gas": gas,
            "gold": gold,
            "crude": market.get("crude") or {"ok": False},
            "euro": market.get("euro") or {"ok": False},
            "dollar": dollar,
            "methanol": methanol,
            "meoh": market.get("meoh") or {"ok": False},
            "zagros": market.get("zagros") or {"ok": False},
            "fanavaran": market.get("fanavaran") or {"ok": False},
            "marjan": market.get("marjan") or {"ok": False},
        },
        "fetched_at": market.get("fetched_at"),
        "dated_notes": [
            "میانگین قیمت صادرات زاگرس در نیمه اول ۲۰۲۶ حدود ۳۳۴ دلار بر تن گزارش شده و قیمت گاز خوراک همان شرکت ۱۵۸ درصد نسبت به مدت مشابه سال قبل بالا رفته است. این عدد قیمت لحظه‌ای نیست.",
            "متانکس در ۳۰ اکتبر ۲۰۲۶ گفت تقاضای صنعت حدود ۹۰ میلیون تن شده و واحدهای MTO ساحلی چین خاموش بوده‌اند. این وضعیت گزارش تماس سود است، نه سنسور زنده.",
        ],
    }
