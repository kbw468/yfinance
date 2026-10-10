"""Plain-language names for features and rule legs."""
import re

EXACT = {
    "off_low_252": "gain off 52w low", "new_high63_share_63": "share of days at a 63d high", "new_high252_count_63": "52w highs in the last 63d",
    "higher_lows_6": "rising 10-day lows (of 6)", "close_loc_21": "close location in the day's range", "gap_share_21": "overnight share of movement",
    "vol_roc_21_252": "typical move 21d vs 1y", "vol_roc_5_63": "typical move 5d vs 63d", "vol_roc_21_vs_21ago": "typical move vs a month ago",
    "vol_pctile_own_252": "typical move vs own year", "range_comp_10_252": "daily range 10d vs 1y", "range_roc_21_252": "daily range 21d vs 1y",
    "down_up_move_63": "down-day size vs up-day size", "worst_day_63": "worst day vs typical move", "volume_pctile_own_252": "volume vs own year",
    "accum_minus_dist_50": "accumulation minus distribution days", "log_dollar_vol_63": "dollar volume", "capture_spread_252": "up-capture minus down-capture",
    "defend_share_63": "up on SPY down days", "rs_off_high_252": "RS line vs its 52w high", "shock_volx_63": "largest volume shock (63d)",
    "shock_age_63": "sessions since the volume shock", "shock_xs_63": "excess return on the volume-shock day", "shock_gap_63": "gap on the volume-shock day",
    "shock_hold_63": "price vs the shock-day close", "shock_lowhold_63": "price vs the shock-day low", "shock_xs_since_63": "excess return since the shock day",
    "shock_signed_63": "volume-weighted shock-day excess", "earn_due_21": "report due (volume cadence)", "since_spike": "sessions since a 2.5x volume day",
    "vshock_1": "today's volume vs 50d median", "vshock_5": "5-day volume vs 50d median", "vrank_50": "today's volume vs own 50 days",
    "vshock_up_10": "biggest up-day volume shock (10d)", "vshock_dn_10": "biggest down-day volume shock (10d)", "max1_21": "biggest daily gain (21d)",
    "max5_21": "five biggest daily gains (21d)", "max_rel_21": "biggest daily gain vs typical move", "min1_21": "worst daily loss (21d)",
    "tail_ratio_126": "upside tail vs downside tail (126d)", "mom_12_1": "12-month return ex last month", "id_12_1": "information discreteness 12-1",
    "lead_persist_63": "days in the trailing-MAR top 20% (63d)", "lead_persist_252": "days in the trailing-MAR top 20% (1y)", "rank_mom_21": "63d return rank change over 21d",
    "on_pos_63": "share of up opening gaps", "bounce_21": "bounce off the 21d low", "streak": "current up/down close streak", "sign_persist_63": "day-to-day sign repeats",
    "log_price": "share price", "ind_breadth_63h": "industry peers near 63d highs", "peer_gap_21": "peers' 21d return minus own",
    "seas_xs_med": "same window, prior years: median excess", "seas_pos": "same window, prior years: share positive",
    "seas_marpct_med": "same window, prior years: risk-adjusted rank", "seas_sup_share": "same window, prior years: share superior",
    "own_sup_rate_756": "own 3-year record of superior trades", "own_sup_rate_252": "own 1-year record of superior trades",
    "own_stop_rate_252": "own stop-out rate over the last year", "own_dd_med_252": "own typical in-trade drawdown", "own_marpct_med_252": "own median trade rank",
    "ind_sup_rate_21": "industry peers' recent superior rate", "si_dtc": "days to cover", "si_chg_1": "short interest change, last settlement",
    "si_chg_3": "short interest change, last three settlements", "si_dtc_pctile_own": "days to cover vs own year", "si_dtc_now": "days to cover on current volume",
}
PATTERNS = [
    (r"roc_(\d+)$", "price change {}d"), (r"xs_spy_(\d+)$", "excess vs SPY {}d"), (r"xs_sec_(\d+)$", "excess vs sector {}d"),
    (r"off_high_(\d+)$", "closeness to {}d high"), (r"days_since_high_(\d+)$", "sessions since {}d high"), (r"mdd_(\d+)$", "{}d drawdown shallowness"),
    (r"mean_dd_(\d+)$", "{}d under-water shallowness"), (r"eff_(\d+)$", "{}d path efficiency"), (r"up_share_(\d+)$", "{}d share of up days"),
    (r"up_pair_(\d+)_63$", "rising {}-day pairs (63d)"), (r"gain_pain_(\d+)$", "{}d gain-to-pain"), (r"mad_(\d+)$", "typical daily move {}d"),
    (r"volume_roc_(\d+)_(\d+)$", "volume {}d vs {}d"), (r"up_volume_share_(\d+)$", "{}d up-volume share"), (r"pullback_volume_(\d+)$", "{}d down-day vs up-day volume"),
    (r"up_capture_(\d+)$", "up-capture {}d"), (r"down_capture_(\d+)$", "down-capture {}d"), (r"beta_l1_(\d+)$", "L1 beta {}d"),
    (r"effx_spy_(\d+)$", "excess-path efficiency vs SPY {}d"), (r"effx_sec_(\d+)$", "excess-path efficiency vs sector {}d"),
    (r"mar_trail_(\d+)$", "trailing excess per drawdown {}d"), (r"on_(\d+)$", "overnight return {}d"), (r"in_(\d+)$", "intraday return {}d"),
    (r"rpos_(\d+)$", "position in {}d range"), (r"ind_roc_(\d+)$", "industry peers' {}d return"),
]
MKT = {"mkt:SPY within 5% of high": "SPY within 5% of its high", "mkt:SPY >5% below high": "SPY more than 5% below its high",
       "mkt:SPY 21d up": "SPY up over 21d", "mkt:SPY 21d down": "SPY down over 21d", "mkt:VIX top third of its year": "VIX in the top third of its year",
       "mkt:VIX bottom third of its year": "VIX in the bottom third of its year", "mkt:breadth low": "breadth low", "mkt:breadth high": "breadth high"}


def name(f: str) -> str:
    own = f.endswith("_pctile_own"); b = f[:-len("_pctile_own")] if own else f
    if b in EXACT: s = EXACT[b]
    else:
        s = b
        for pat, fmt in PATTERNS:
            m = re.fullmatch(pat, b)
            if m: s = fmt.format(*m.groups()); break
    return s + (" vs own year" if own else "")


def leg(c: str) -> str:
    if c in MKT: return MKT[c]
    f, side = c.rsplit(":", 1)
    return f"{name(f)} {'top' if side == 'TOP' else 'bottom'} 20%"


def rule(r: str) -> str:
    return " AND ".join(leg(c) for c in r.split(" & "))
