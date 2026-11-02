"""Smoke test: run the Streamlit app headless on synthetic data and press Run for every strategy."""
import pandas as pd
from streamlit.testing.v1 import AppTest

import core.data
import core.earnings
import core.universe
from tests.synthetic import SyntheticProvider, universe

END = (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)).date()


def test_app_runs_all_strategies(monkeypatch):
    prov = SyntheticProvider(END)
    monkeypatch.setattr(core.data, "get_provider", lambda secret: prov)
    monkeypatch.setattr(core.universe, "load_universe", lambda names: (universe(), "synthetic"))
    monkeypatch.setattr(core.earnings, "next_earnings",
                        lambda syms, today, w, key: ({s: None for s in syms}, {s: "clear" for s in syms}, "test"))
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    expect = {"Breakout": "BRKO", "Pullback": "PULL", "Oversold bounce": "OVSD", "Momentum": "MOMO"}
    for name, sym in expect.items():
        at.segmented_control[0].set_value(name).run()
        at.button[0].click().run()
        assert not at.exception, at.exception
        html = " ".join(m.value for m in at.markdown)
        assert sym in html, (name, html[:500])
