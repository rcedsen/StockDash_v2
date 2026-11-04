"""Russell 1000 parser + fallback, with the network mocked."""
import pandas as pd

from core import universe

HEADER = ('iShares Russell 1000 ETF\n"Fund Holdings as of","Nov 03, 2026"\n"Inception Date","May 15, 2000"\n'
          '"Shares Outstanding","1"\n \n'
          '"Ticker","Name","Sector","Asset Class","Market Value","Weight (%)","Notional Value","Quantity",'
          '"Price","Location","Exchange","Currency","FX Rate","Market Currency","Accrual Date"\n')


def _csv(n):
    rows = [f'"T{i:03d}","CO {i}","Information Technology","Equity","1","0.1","1","1","1","US","NASDAQ","USD","1","USD","-"'
            for i in range(n)]
    rows += ['"BRKB","BERKSHIRE HATHAWAY INC CLASS B","Financials","Equity","1","1","1","1","1","US","NYSE","USD","1","USD","-"',
             '"GOOGL","ALPHABET INC CLASS A","Communication","Equity","1","1","1","1","1","US","NASDAQ","USD","1","USD","-"',
             '"USD","USD CASH","Cash and/or Derivatives","Cash","1","0","1","1","1","US","-","USD","1","USD","-"',
             '"ESZ6","S&P500 EMINI DEC 26","Cash and/or Derivatives","Futures","1","0","1","1","1","US","-","USD","1","USD","-"']
    return HEADER + "\n".join(rows) + '\n \n"The content contained herein is owned or licensed by BlackRock"\n'


class _R:
    def __init__(self, text):
        self.text = text


def test_russell_parser(monkeypatch):
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: _R(_csv(900)))
    u = universe.russell1000()
    assert len(u) == 902
    assert "BRK.B" in set(u.symbol) and "USD" not in set(u.symbol) and "ESZ6" not in set(u.symbol)
    assert u.set_index("symbol").loc["GOOGL", "sector"] == "Communication Services"


def test_russell_fallback(monkeypatch):
    def boom():
        raise ValueError("layout changed")
    monkeypatch.setitem(universe.SOURCES, "russell1000", boom)
    monkeypatch.setitem(universe.SOURCES, "sp500", lambda: pd.DataFrame(dict(symbol=["AAA"], name=["A"], sector=["Energy"])))
    monkeypatch.setitem(universe.SOURCES, "ndx100", lambda: pd.DataFrame(dict(symbol=["BBB"], name=["B"], sector=[None])))
    monkeypatch.setitem(universe.SOURCES, "sp400", lambda: pd.DataFrame(dict(symbol=["CCC"], name=["C"], sector=["Materials"])))
    u, note = universe.load_universe(["sp500", "ndx100", "russell1000"])
    assert set(u.symbol) == {"AAA", "BBB", "CCC"}
    assert "fallback for russell1000" in note and "russell1000 (ValueError)" in note
