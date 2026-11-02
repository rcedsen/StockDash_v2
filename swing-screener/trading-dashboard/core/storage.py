"""Google Sheets log. Optional: if the service account / SHEET_ID secrets are missing, logging is
disabled and the screener still works."""
from __future__ import annotations

import pandas as pd

SIGNAL_COLS = ["run_id", "run_time_et", "strategy", "rank", "symbol", "score", "price", "entry",
               "stop_structure", "stop_atr", "target_2r", "bar_date", "data_mode", "regime_ok",
               "sector_state", "earnings", "status", "outcome", "outcome_r", "exit_date", "mfe_r",
               "mae_r", "notes"]
RUN_COLS = ["run_id", "run_time_et", "strategy", "provider", "data_mode", "data_asof", "universe_n",
            "qualified", "excluded_earnings", "shown", "warnings"]


class SheetLog:
    def __init__(self, secret):
        self.error = None
        self.sh = None
        creds, sheet_id = secret("gcp_service_account"), secret("SHEET_ID")
        if not creds or not sheet_id:
            self.error = "Logging off — add SHEET_ID and [gcp_service_account] to secrets."
            return
        try:
            import gspread
            self.sh = gspread.service_account_from_dict(dict(creds)).open_by_key(sheet_id)
        except Exception as e:
            self.error = f"Logging off — Sheets connection failed: {type(e).__name__}: {e}"

    @property
    def on(self) -> bool:
        return self.sh is not None

    def _ws(self, name, cols):
        import gspread
        try:
            ws = self.sh.worksheet(name)
        except gspread.WorksheetNotFound:
            ws = self.sh.add_worksheet(name, rows=1000, cols=len(cols))
            ws.append_row(cols)
        return ws

    def log_run(self, run: dict, signals: list[dict]):
        self._ws("runs", RUN_COLS).append_row([_cell(run.get(c)) for c in RUN_COLS],
                                              value_input_option="USER_ENTERED")
        if signals:
            self._ws("signals", SIGNAL_COLS).append_rows(
                [[_cell(s.get(c)) for c in SIGNAL_COLS] for s in signals], value_input_option="USER_ENTERED")

    def read_signals(self) -> pd.DataFrame:
        vals = self._ws("signals", SIGNAL_COLS).get_all_values()
        if len(vals) <= 1:
            return pd.DataFrame(columns=SIGNAL_COLS)
        return pd.DataFrame(vals[1:], columns=vals[0])

    def write_signals(self, df: pd.DataFrame):
        ws = self._ws("signals", SIGNAL_COLS)
        df = df.reindex(columns=SIGNAL_COLS)
        ws.clear()
        ws.update([SIGNAL_COLS] + [[_cell(v) for v in row] for row in df.itertuples(index=False)],
                  value_input_option="USER_ENTERED")


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else round(v, 4)
    if isinstance(v, (list, tuple)):
        return " | ".join(map(str, v))
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v
