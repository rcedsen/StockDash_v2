"""Smoke test: run the Streamlit app headless on synthetic data, press Run for every strategy,
and render the Screening Definitions tab."""
import pandas as pd
from streamlit.testing.v1 import AppTest

import core.data
import core.earnings
import core.fundamentals
import core.universe
from tests.synthetic import SyntheticProvider, fake_fundamentals, universe

END = (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)).date()


def test_app_runs_all_strategies(monkeypatch):
    prov = SyntheticProvider(END)
    monkeypatch.setattr(core.data, "get_provider", lambda secret: prov)
    monkeypatch.setattr(core.universe, "load_universe", lambda names: (universe(), "synthetic"))
    monkeypatch.setattr(core.earnings, "next_earnings",
                        lambda syms, today, h, key: ({s: None for s in syms}, set(syms), "test"))
    monkeypatch.setattr(core.fundamentals, "fetch_many", fake_fundamentals)
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Screen", "Screening Definitions", "Tracker", "Status"]

    expect = {"Breakout": "BRKO", "Pullback": "PULL", "Oversold bounce": "OVSD", "Momentum": "MOMO",
              "Quality dip": "QDIP"}
    for name, sym in expect.items():
        at.segmented_control[0].set_value(name).run()
        at.button[0].click().run()
        assert not at.exception, at.exception
        html = " ".join(m.value for m in at.markdown)
        assert sym in html, (name, html[:500])
        assert "time stop" in html
        if name == "Quality dip":
            assert "russell1000" in html and "Stop distance" in html
            assert {b.label for b in at.get("download_button")} >= {"CSV · all constituents", "Ranked report"}

    defs = " ".join(m.value for m in at.tabs[1].markdown)
    for frag in ["10%–30% below the 252-session high", "ROIC ≥ 10%", "200 SMA ≥ its value 20 sessions ago",
                 "A+D (15)", "within 5%"]:
        assert frag.lower() in defs.lower(), frag
    assert "{" not in defs.replace("{'", "").replace("{\"", ""), "unfilled placeholder"
