"""Post-simulation diagnostics; outputs never feed historical decisions."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence


def _get(value: Any, name: str, default: Any=None) -> Any:
    return value.get(name,default) if isinstance(value,Mapping) else getattr(value,name,default)


def calculate_diagnostics(run_id: str, trades: Sequence[Any], signals: Sequence[Any], snapshots: Sequence[Any], rejected: Sequence[Any], histories: Mapping[str,list[dict[str,Any]]], maximum_positions: int) -> dict[str,Any]:
    grouped: dict[str,list[Any]]=defaultdict(list)
    for trade in trades: grouped[str(_get(trade,"symbol"))].append(trade)
    attribution=[]
    total_profit=sum(max(float(_get(t,"realized_pnl",0) or 0),0) for t in trades)
    for symbol,items in sorted(grouped.items()):
        pnls=[float(_get(t,"realized_pnl",0) or 0) for t in items]; commissions=[float(_get(t,"total_commission",0) or 0) for t in items]; slippage=[float(_get(t,"total_slippage",0) or 0) for t in items]
        attribution.append({"run_id":run_id,"symbol":symbol,"trades":len(items),"gross_pnl":sum(pnls)+sum(commissions)+sum(slippage),"net_pnl":sum(pnls),
          "profit_contribution_pct":sum(max(v,0) for v in pnls)/total_profit if total_profit else None,"win_rate":sum(v>0 for v in pnls)/len(pnls),
          "average_trade":mean(pnls),"average_holding_period":mean(float(_get(t,"holding_period_days",0) or 0) for t in items),"commission":sum(commissions),"slippage":sum(slippage)})
    positive=sorted((float(_get(t,"realized_pnl",0) or 0) for t in trades),reverse=True); by_symbol={r["symbol"]:r["net_pnl"] for r in attribution}
    concentration={"best_trade_profit_pct":positive[0]/total_profit if positive and total_profit else None,"best_three_profit_pct":sum(positive[:3])/total_profit if total_profit else None,
      "best_symbol_profit_pct":max(by_symbol.values())/total_profit if by_symbol and total_profit else None,"net_profit_excluding_best_trade":sum(float(_get(t,"realized_pnl",0) or 0) for t in trades)-(positive[0] if positive else 0),
      "net_profit_excluding_best_three":sum(float(_get(t,"realized_pnl",0) or 0) for t in trades)-sum(positive[:3]),
      "net_profit_excluding_best_symbol":sum(by_symbol.values())-(max(by_symbol.values()) if by_symbol else 0),"net_profit_excluding_worst_symbol":sum(by_symbol.values())-(min(by_symbol.values()) if by_symbol else 0)}
    outcomes=[]
    price_maps={s:{str(r.get("trade_date"))[:10]:float(r.get("adjusted_close") or r.get("close")) for r in rows if r.get("adjusted_close") or r.get("close")} for s,rows in histories.items()}
    for signal in signals:
        if _get(signal,"action")!="entry": continue
        symbol=str(_get(signal,"symbol")); day=str(_get(signal,"signal_date")); dates=sorted(price_maps.get(symbol,{}));
        if day not in dates: continue
        idx=dates.index(day); base=price_maps[symbol][day]; later=[price_maps[symbol][d] for d in dates[idx+1:idx+64]]
        returns={n:((price_maps[symbol][dates[idx+n]]/base-1) if idx+n<len(dates) else None) for n in (5,21,63)}
        outcomes.append({"run_id":run_id,"signal_id":str(_get(signal,"signal_id")),"symbol":symbol,"signal_date":day,"return_5":returns[5],"return_21":returns[21],"return_63":returns[63],
          "maximum_favorable_excursion":max((p/base-1 for p in later),default=None),"maximum_adverse_excursion":min((p/base-1 for p in later),default=None)})
    exits=[]
    for trade in trades:
        symbol=str(_get(trade,"symbol")); dates=sorted(price_maps.get(symbol,{})); entry=str(_get(trade,"execution_date")); exit_day=str(_get(trade,"exit_execution_date")); entry_price=float(_get(trade,"fill_price",0) or 0); exit_price=float(_get(trade,"exit_fill_price",0) or 0)
        if not entry_price or not exit_price or exit_day not in dates: continue
        exit_idx=dates.index(exit_day); entry_idx=dates.index(entry) if entry in dates else exit_idx; before=[price_maps[symbol][d] for d in dates[entry_idx:exit_idx+1]]; after=[price_maps[symbol][d] for d in dates[exit_idx+1:exit_idx+22]]; end_price=price_maps[symbol][dates[-1]]
        later_end=after[-1] if after else exit_price; later_max=max(after,default=exit_price)
        exits.append({"run_id":run_id,"trade_id":str(_get(trade,"trade_id")),"symbol":symbol,"exit_reason":_get(trade,"exit_reason"),"realized_pnl":_get(trade,"realized_pnl"),"maximum_before_exit":max(before,default=None),"maximum_after_exit":later_max,"research_window_return":later_end/exit_price-1,"hold_to_end_return":end_price/exit_price-1,"avoided_later_loss":int(min(after,default=exit_price)<exit_price),"exited_before_later_gain":int(later_max>exit_price)})
    daily=[]; rejected_dates={str(_get(r,"signal_date")) for r in rejected}
    for point in snapshots:
        equity=float(_get(point,"equity",0) or 0); cash=float(_get(point,"cash",0) or 0); positions=int(_get(point,"position_count",0) or 0); day=str(_get(point,"snapshot_date"))
        daily.append({"run_id":run_id,"trade_date":day,"cash_percentage":cash/equity if equity else None,"fully_invested":int(positions>=maximum_positions),"below_maximum_positions":int(positions<maximum_positions),"no_eligible_candidate":0,"rejected_eligible_candidates":int(day in rejected_dates),"benchmark_return":None})
    return {"run_id":run_id,"attribution":attribution,"concentration":concentration,"signal_outcomes":outcomes,"exit_diagnostics":exits,"daily":daily}


def persist_diagnostics(conn: Any, diagnostics: Mapping[str,Any]) -> None:
    for name,value in diagnostics.get("concentration",{}).items(): conn.execute("INSERT OR REPLACE INTO backtest_metrics(run_id,metric_name,metric_value,metric_json) VALUES(?,?,?,NULL)",(diagnostics["run_id"],f"diagnostic_{name}",value))
    for row in diagnostics.get("attribution",[]): conn.execute("""INSERT OR REPLACE INTO backtest_symbol_attribution(run_id,symbol,trades,gross_pnl,net_pnl,profit_contribution_pct,win_rate,average_trade,average_holding_period,commission,slippage) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",tuple(row[k] for k in ("run_id","symbol","trades","gross_pnl","net_pnl","profit_contribution_pct","win_rate","average_trade","average_holding_period","commission","slippage")))
    for row in diagnostics.get("signal_outcomes",[]): conn.execute("""INSERT OR REPLACE INTO backtest_signal_outcomes(run_id,signal_id,symbol,signal_date,return_5,return_21,return_63,maximum_favorable_excursion,maximum_adverse_excursion) VALUES(?,?,?,?,?,?,?,?,?)""",tuple(row[k] for k in ("run_id","signal_id","symbol","signal_date","return_5","return_21","return_63","maximum_favorable_excursion","maximum_adverse_excursion")))
    for row in diagnostics.get("daily",[]): conn.execute("""INSERT OR REPLACE INTO backtest_daily_diagnostics(run_id,trade_date,cash_percentage,fully_invested,below_maximum_positions,no_eligible_candidate,rejected_eligible_candidates,benchmark_return) VALUES(?,?,?,?,?,?,?,?)""",tuple(row[k] for k in ("run_id","trade_date","cash_percentage","fully_invested","below_maximum_positions","no_eligible_candidate","rejected_eligible_candidates","benchmark_return")))
    for row in diagnostics.get("exit_diagnostics",[]): conn.execute("""INSERT OR REPLACE INTO backtest_exit_diagnostics(run_id,trade_id,symbol,exit_reason,realized_pnl,maximum_before_exit,maximum_after_exit,research_window_return,hold_to_end_return,avoided_later_loss,exited_before_later_gain) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",tuple(row[k] for k in ("run_id","trade_id","symbol","exit_reason","realized_pnl","maximum_before_exit","maximum_after_exit","research_window_return","hold_to_end_return","avoided_later_loss","exited_before_later_gain")))
