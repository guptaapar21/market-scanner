"""
Live market scanner for NASDAQ-100 + S&P 500.

Two modes, controlled by the RUN_MODE env var:

- "intraday" (default): runs every 15-30 min during US market hours via
  live_scan.yml. Cheap rule-based scan of ALL tickers using 15-min bars
  (price move %, volume spike). Sends an alert the moment a rule fires.

- "daily_summary": runs once, after market close, via daily_summary.yml.
  Uses each ticker's full daily candle (yfinance intraday data goes stale
  once the market's closed, so this looks at settled end-of-day data
  instead) and sends one consolidated digest of top gainers/losers/volume
  movers, rather than repeated per-ticker pings.

In both modes, Tier 2 (optional): shells out to the daily_stock_analysis
project for a full AI decision dashboard on the most significant tickers,
capped by a shared daily budget (DEEP_ANALYSIS_DAILY_LIMIT).
"""

import io
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ---------- Config ----------
NY_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)

PRICE_MOVE_THRESHOLD_PCT = float(os.environ.get("PRICE_MOVE_THRESHOLD_PCT", "3.0"))
VOLUME_SPIKE_MULTIPLE = float(os.environ.get("VOLUME_SPIKE_MULTIPLE", "3.0"))

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

STATE_FILE = Path("state/alerted_today.json")
TICKER_CACHE_FILE = Path("state/tickers_cache.json")

DEEP_ANALYSIS_ENABLED = os.environ.get("DEEP_ANALYSIS_ENABLED", "false").lower() == "true"
DEEP_ANALYSIS_REPO_DIR = os.environ.get("DEEP_ANALYSIS_REPO_DIR", "daily_stock_analysis")
# Cap on how many deep (LLM) analyses to run per day, across all runs.
# Keeps you well under Gemini's free-tier daily request cap even on a
# volatile day with many triggers. Conservative default -- each ticker's
# analysis can itself involve more than one underlying LLM call (search,
# integrity retries, etc.), so this counts tickers, not raw API calls.
DEEP_ANALYSIS_DAILY_LIMIT = int(os.environ.get("DEEP_ANALYSIS_DAILY_LIMIT", "40"))

# "intraday" (default): the every-15-min scan during market hours above.
# "daily_summary": a once-a-day digest using the prior session's full daily
# candle, meant to run after market close (see run_daily_summary()).
RUN_MODE = os.environ.get("RUN_MODE", "intraday")
DAILY_SUMMARY_TOP_N = int(os.environ.get("DAILY_SUMMARY_TOP_N", "10"))
DAILY_SUMMARY_DEEP_COUNT = int(os.environ.get("DAILY_SUMMARY_DEEP_COUNT", "5"))


def is_market_open_now() -> bool:
    """Rough check: weekday + within regular NYSE/NASDAQ hours (ET).
    Does not account for market holidays -- yfinance simply returns
    stale/empty data on those days, which scan_batch() handles gracefully."""
    now = datetime.now(NY_TZ)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    close_t = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
    return open_t <= now <= close_t


def _fetch_wikipedia_tables(url: str) -> list:
    """pd.read_html() alone sends no User-Agent, and Wikipedia now 403s
    requests that look bot-like. Fetch the page ourselves with a normal
    browser User-Agent first, then hand the HTML to pandas to parse."""
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        timeout=20,
    )
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_universe(force_refresh: bool = False) -> list:
    """S&P 500 + Nasdaq-100 tickers, deduped, fetched from Wikipedia with a
    same-day local cache so we don't hit Wikipedia every 15 minutes."""
    if TICKER_CACHE_FILE.exists() and not force_refresh:
        cached = json.loads(TICKER_CACHE_FILE.read_text())
        if cached.get("date") == str(date.today()):
            return cached["tickers"]

    tickers = set()
    try:
        sp500 = _fetch_wikipedia_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers.update(sp500["Symbol"].astype(str).str.replace(".", "-", regex=False))
    except Exception as e:
        print(f"WARN: failed to fetch S&P 500 list: {e}", file=sys.stderr)

    try:
        tables = _fetch_wikipedia_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
        table = next(t for t in tables if "Ticker" in t.columns or "Symbol" in t.columns)
        col = "Ticker" if "Ticker" in table.columns else "Symbol"
        tickers.update(table[col].astype(str).str.replace(".", "-", regex=False))
    except Exception as e:
        print(f"WARN: failed to fetch Nasdaq-100 list: {e}", file=sys.stderr)

    if not tickers:
        if TICKER_CACHE_FILE.exists():
            print("WARN: using stale cached ticker list", file=sys.stderr)
            return json.loads(TICKER_CACHE_FILE.read_text())["tickers"]
        raise RuntimeError("Could not fetch ticker universe and no cache available")

    result = sorted(tickers)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TICKER_CACHE_FILE.write_text(json.dumps({"date": str(date.today()), "tickers": result}))
    return result


def load_state() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        if state.get("date") == str(date.today()):
            state.setdefault("deep_analysis_count", 0)
            return state
    return {"date": str(date.today()), "alerted": {}, "deep_analysis_count": 0}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        if not resp.ok:
            print(f"WARN: Telegram send failed: {resp.status_code} {resp.text}", file=sys.stderr)
    except Exception as e:
        print(f"WARN: Telegram send raised: {e}", file=sys.stderr)


def scan_batch(tickers: list, batch_size: int = 100) -> list:
    """Download recent intraday price/volume in batches and compute simple triggers."""
    triggers = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            data = yf.download(
                tickers=batch,
                period="5d",
                interval="15m",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"WARN: batch download failed for {batch[:3]}...: {e}", file=sys.stderr)
            continue

        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data
                df = df.dropna()
                if df.empty or len(df) < 20:
                    continue

                latest = df.iloc[-1]
                today_rows = df[df.index.date == df.index[-1].date()]
                if today_rows.empty:
                    continue
                today_open = today_rows.iloc[0]["Open"]
                if not today_open:
                    continue
                pct_change = (latest["Close"] - today_open) / today_open * 100

                avg_volume = df["Volume"].iloc[:-1].mean()
                volume_ratio = latest["Volume"] / avg_volume if avg_volume else 0

                reasons = []
                if abs(pct_change) >= PRICE_MOVE_THRESHOLD_PCT:
                    direction = "up" if pct_change > 0 else "down"
                    reasons.append(f"moved {direction} {abs(pct_change):.1f}% today")
                if volume_ratio >= VOLUME_SPIKE_MULTIPLE:
                    reasons.append(f"volume {volume_ratio:.1f}x average")

                if reasons:
                    triggers.append(
                        {
                            "ticker": ticker,
                            "price": round(float(latest["Close"]), 2),
                            "pct_change": round(float(pct_change), 2),
                            "volume_ratio": round(float(volume_ratio), 2),
                            "reasons": reasons,
                        }
                    )
            except Exception as e:
                print(f"WARN: failed to process {ticker}: {e}", file=sys.stderr)
                continue

    return triggers


def run_deep_analysis(ticker: str) -> None:
    """Shell out to daily_stock_analysis's own documented CLI for one ticker.
    That project handles its own LLM call and pushes to the SAME Telegram
    bot/chat as the fast alert above, since it inherits TELEGRAM_BOT_TOKEN
    and TELEGRAM_CHAT_ID from this process's environment by default -- no
    extra config needed, both messages land in the same chat."""
    if not Path(DEEP_ANALYSIS_REPO_DIR).exists():
        print(f"WARN: deep analysis repo not found at {DEEP_ANALYSIS_REPO_DIR}, skipping", file=sys.stderr)
        return

    try:
        subprocess.run(
            # --no-market-review: this call only wants ticker's dashboard,
            # not a full market-wide recap (main.py runs both by default) --
            # otherwise every deep-dived ticker re-triggers a whole extra
            # market review, which is what was flooding Telegram.
            [sys.executable, "main.py", "--stocks", ticker, "--no-market-review"],
            cwd=DEEP_ANALYSIS_REPO_DIR,
            check=True,
            timeout=600,
        )
    except Exception as e:
        print(f"WARN: deep analysis failed for {ticker}: {e}", file=sys.stderr)


def score_trigger(t: dict) -> float:
    """Rank triggers by how 'significant' they look, so that when we can
    only afford to run deep analysis on some of them (daily LLM budget),
    the strongest signals go first rather than whichever ticker happened
    to sort alphabetically or come back first from yfinance.

    Simple, transparent scoring: how far past each threshold the move is,
    summed. A stock at 2x its price threshold AND 2x its volume threshold
    scores higher than one that barely crossed either line alone.
    """
    price_score = abs(t["pct_change"]) / PRICE_MOVE_THRESHOLD_PCT if PRICE_MOVE_THRESHOLD_PCT else 0
    volume_score = t["volume_ratio"] / VOLUME_SPIKE_MULTIPLE if VOLUME_SPIKE_MULTIPLE else 0
    return price_score + volume_score


def scan_daily(tickers: list, batch_size: int = 100) -> list:
    """Like scan_batch(), but uses the prior session's full daily candle
    instead of intraday 15-min bars. Meant to run once, after market close,
    when yfinance's daily data has settled. Returns ALL tickers with valid
    data (not threshold-gated) -- run_daily_summary() picks the top movers
    from this, since it's a once-a-day digest rather than repeated pings."""
    results = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            data = yf.download(
                tickers=batch,
                period="1mo",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"WARN: daily batch download failed for {batch[:3]}...: {e}", file=sys.stderr)
            continue

        for ticker in batch:
            try:
                df = data[ticker] if len(batch) > 1 else data
                df = df.dropna()
                if len(df) < 6:
                    continue

                latest = df.iloc[-1]
                prev_close = df.iloc[-2]["Close"]
                if not prev_close:
                    continue
                pct_change = (latest["Close"] - prev_close) / prev_close * 100

                avg_volume = df["Volume"].iloc[-6:-1].mean()
                volume_ratio = latest["Volume"] / avg_volume if avg_volume else 0

                results.append(
                    {
                        "ticker": ticker,
                        "price": round(float(latest["Close"]), 2),
                        "pct_change": round(float(pct_change), 2),
                        "volume_ratio": round(float(volume_ratio), 2),
                        "reasons": [],
                    }
                )
            except Exception as e:
                print(f"WARN: failed to process {ticker} (daily): {e}", file=sys.stderr)
                continue

    return results


def build_summary_message(gainers: list, losers: list, volume_movers: list) -> str:
    lines = [f"*Daily Summary — {date.today().isoformat()}*\n"]

    lines.append("*Top Gainers*")
    for t in gainers:
        lines.append(f"  {t['ticker']}: ${t['price']} ({t['pct_change']:+.1f}%)")

    lines.append("\n*Top Losers*")
    for t in losers:
        lines.append(f"  {t['ticker']}: ${t['price']} ({t['pct_change']:+.1f}%)")

    lines.append("\n*Volume Spikes*")
    for t in volume_movers:
        lines.append(f"  {t['ticker']}: {t['volume_ratio']:.1f}x avg volume ({t['pct_change']:+.1f}%)")

    return "\n".join(lines)


def run_daily_summary() -> None:
    """Runs once, after market close (see the separate cron in
    daily_summary.yml), regardless of is_market_open_now(). Uses the prior
    session's full daily candle instead of intraday bars, sends one
    consolidated digest, then runs deep analysis on just the top few
    tickers by significance (see score_trigger)."""
    tickers = get_universe()
    print(f"Daily summary: scanning {len(tickers)} tickers using prior session's daily candle...")

    results = scan_daily(tickers)
    print(f"Got valid daily data for {len(results)} tickers.")
    if not results:
        print("No data -- likely no trading session available yet, skipping.")
        return

    by_change = sorted(results, key=lambda t: t["pct_change"], reverse=True)
    gainers = by_change[:DAILY_SUMMARY_TOP_N]
    losers = list(reversed(by_change[-DAILY_SUMMARY_TOP_N:]))
    volume_movers = sorted(results, key=lambda t: t["volume_ratio"], reverse=True)[:DAILY_SUMMARY_TOP_N]

    send_telegram(build_summary_message(gainers, losers, volume_movers))

    # Dedup the union of movers, ranked by significance, deep-analyze only
    # the top handful so a slow news day doesn't burn the whole daily
    # LLM budget on marginal names.
    union = {t["ticker"]: t for t in gainers + losers + volume_movers}
    ranked = sorted(union.values(), key=score_trigger, reverse=True)

    state = load_state()
    top_for_deep_dive = ranked[:DAILY_SUMMARY_DEEP_COUNT]
    for t in top_for_deep_dive:
        remaining_budget = DEEP_ANALYSIS_DAILY_LIMIT - state["deep_analysis_count"]
        if DEEP_ANALYSIS_ENABLED and remaining_budget > 0:
            run_deep_analysis(t["ticker"])
            state["deep_analysis_count"] += 1
    save_state(state)
    print(f"Daily summary done. Deep-dived {len(top_for_deep_dive)} tickers.")


def run_intraday() -> None:
    if not is_market_open_now():
        print("Market closed (or outside 9:30-16:00 ET), skipping run.")
        return

    tickers = get_universe()
    print(f"Scanning {len(tickers)} tickers (NASDAQ-100 + S&P 500 union)...")

    state = load_state()
    triggers = scan_batch(tickers)
    print(f"Found {len(triggers)} raw triggers.")

    new_triggers = [t for t in triggers if t["ticker"] not in state["alerted"]]
    # Strongest signals first, so if the daily deep-analysis budget runs
    # out partway through, it's the marginal triggers that get skipped,
    # not the standout ones.
    new_triggers.sort(key=score_trigger, reverse=True)
    print(f"{len(new_triggers)} are new today (not yet alerted), sorted by significance.")

    for t in new_triggers:
        remaining_budget = DEEP_ANALYSIS_DAILY_LIMIT - state["deep_analysis_count"]
        will_run_deep = DEEP_ANALYSIS_ENABLED and remaining_budget > 0

        msg = (
            f"*{t['ticker']}* alert\n"
            f"Price: ${t['price']} ({t['pct_change']:+.1f}%)\n"
            f"Volume: {t['volume_ratio']:.1f}x average\n"
            f"Reasons: {', '.join(t['reasons'])}"
        )
        if DEEP_ANALYSIS_ENABLED and not will_run_deep:
            msg += "\n_(AI dashboard skipped: daily analysis limit reached)_"
        send_telegram(msg)
        state["alerted"][t["ticker"]] = datetime.now(NY_TZ).isoformat()

        if will_run_deep:
            run_deep_analysis(t["ticker"])
            state["deep_analysis_count"] += 1

    save_state(state)
    print(
        f"Done. Deep analyses used today: {state['deep_analysis_count']}/{DEEP_ANALYSIS_DAILY_LIMIT}."
    )


def main() -> None:
    if RUN_MODE == "daily_summary":
        run_daily_summary()
    else:
        run_intraday()


if __name__ == "__main__":
    main()
