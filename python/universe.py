"""Default liquid optionable universe. Edit as needed."""
import json
import sys

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META",
    "AMZN", "TSLA", "NFLX", "AVGO", "CRM", "ORCL", "ADBE", "INTC", "MU", "QCOM",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "V", "MA", "AXP", "PYPL",
    "XOM", "CVX", "COP", "OXY", "SLB",
    # MRO removed: ConocoPhillips acquired Marathon Oil Nov 2024, ticker delisted.
    "WMT", "TGT", "COST", "HD", "LOW", "NKE", "MCD", "SBUX", "DIS",
    "UNH", "JNJ", "PFE", "MRK", "LLY", "ABBV",
    "BA", "CAT", "GE", "F", "GM", "DE",
    "COIN", "XYZ", "SHOP", "UBER", "ABNB", "PLTR", "SNOW", "CRWD", "ZM", "DOCU",
    # XYZ replaces SQ: Block Inc. rebranded its ticker from SQ to XYZ in Feb 2025.
]

if __name__ == "__main__":
    json.dump({"tickers": DEFAULT_UNIVERSE}, sys.stdout)
