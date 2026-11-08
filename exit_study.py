"""手仕舞いルールの検証 (docs/signal_accuracy_2026-07.md 参照)。

4年バックフィルの終値だけを使い、BUY後の手仕舞いルールを
(1) 固定保有との比較 (2) 同じ保有日数のランダム手仕舞いとの比較 で評価する。
使い方: python3 exit_study.py  (results/signals_history_backfill.csv が必要)
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from signal_tracker import adjust_for_splits
b = pd.read_csv('results/signals_history_backfill.csv', parse_dates=['signal_date'])
cls = b.drop_duplicates('ticker').set_index('ticker').ticker_class
panel = b.pivot_table(index='signal_date', columns='ticker', values='close', aggfunc='last').sort_index().apply(adjust_for_splits)
H = 60

def prep(t):
    c = panel[t].dropna()
    sig = {s: b[(b.ticker==t)&(b.strategy==s)].set_index('signal_date') for s in b.strategy.unique()}
    r = c.pct_change()
    return c, r.rolling(20).std(), c.rolling(20).mean(), sig

def simulate(c, sd, sma, rule, i, rawsell=None):
    """entry at close i. return (ret, days, mdd). exit at next-day close after trigger"""
    n = len(c); cv = c.values; peak = cv[i]
    end = min(i+H, n-1)
    for j in range(i+1, end):
        peak = max(peak, cv[j])
        trig = False
        kind, par = rule
        if kind == 'trail':   trig = cv[j] <= peak*(1 - par*sd.values[j]) if not np.isnan(sd.values[j]) else False
        elif kind == 'sma':   trig = cv[j] < sma.values[j]
        elif kind == 'raw':   trig = rawsell is not None and rawsell.values[j]
        elif kind == 'trail_or_sma': trig = (cv[j] <= peak*(1-par*sd.values[j])) or (cv[j] < sma.values[j])
        elif kind == 'hold': trig = (j-i) >= par
        if trig:
            x = min(j+1, end); break
    else:
        x = end
    seg = cv[i:x+1]/cv[i]-1
    return seg[-1], x-i, seg.min()

rules = {'保有60日(基準)':('hold',60), '固定20日':('hold',20),
         'trail 3σ':('trail',3), 'trail 4σ':('trail',4), 'trail 6σ':('trail',6),
         'SMA20割れ':('sma',0), 'trail4σ or SMA20':('trail_or_sma',4), 'rawSELL(トレンド)':('raw',0)}

def run(entries_fn, label, classes):
    rows = []
    for t in cls.index[cls.isin(classes)]:
        c, sd, sma, sig = prep(t)
        ent = entries_fn(t, c, sig)
        rawsell = (sig['トレンドフォロー'].raw_signal=='SELL').reindex(c.index).fillna(False)
        for i in ent:
            if i+H >= len(c) or i < 25: continue
            for name, rule in rules.items():
                ret, d, mdd = simulate(c, sd, sma, rule, i, rawsell)
                rows.append((t, c.index[i].year, name, ret, d, mdd))
    df = pd.DataFrame(rows, columns=['t','yr','rule','ret','days','mdd'])
    g = df.groupby('rule', sort=False)
    out = pd.DataFrame({'n': g.size(), '平均ret%': g.ret.mean()*100, '中央値%': g.ret.median()*100,
        '勝率%': g.ret.apply(lambda x:(x>0).mean()*100), '平均保有日': g.days.mean(),
        '平均最大DD%': g.mdd.mean()*100, 'ワースト10%点': g.ret.quantile(.1)*100})
    out['1日あたりret%x20'] = out['平均ret%']/out['平均保有日']*20
    print(f'\n===== {label} =====');print(out.round(2).to_string())
    return df

def onset(strategy):
    def f(t, c, sig):
        s = sig[strategy].signal.reindex(c.index)
        on = (s=='BUY') & (s.shift(1)!='BUY')
        return [k for k,v in enumerate(on.values) if v]
    return f
def every5(t, c, sig): return list(range(25, len(c), 5))

d1 = run(onset('トレンドフォロー'), 'plain+long_lev: トレンドBUY onset', ['plain','long_lev'])
d2 = run(onset('逆張り'), 'plain+long_lev: 逆張りBUY onset', ['plain','long_lev'])
d3 = run(every5, '対照: plain 全日(5日おき)エントリー', ['plain'])

# ---- ランダム手仕舞い対照: 同じ保有日数分布を銘柄・日付と無関係に割り当てる ----
print('\n\n########## 対照検定: ルールの手仕舞いタイミング vs 同じ保有日数のランダム手仕舞い ##########')
rng = np.random.default_rng(0)
def timing_test(entries_fn, label, classes, rule_names, nrep=300):
    ents = []
    for t in cls.index[cls.isin(classes)]:
        c, sd, sma, sig = prep(t)
        rawsell = (sig['トレンドフォロー'].raw_signal=='SELL').reindex(c.index).fillna(False)
        for i in entries_fn(t, c, sig):
            if i+H >= len(c) or i < 25: continue
            ents.append((c, sd, sma, rawsell, i))
    print(f'\n--- {label} (n={len(ents)}) ---')
    for name in rule_names:
        rule = rules[name]
        res = [simulate(c, sd, sma, rule, i, rs) for c, sd, sma, rs, i in ents]
        ret = np.array([r[0] for r in res]); dur = np.array([r[1] for r in res]); mdd = np.array([r[2] for r in res])
        cvs = [c.values for c, *_ in ents]; idx = [i for *_, i in ents]
        diffs, ddd = [], []
        for _ in range(nrep):
            d_rand = rng.permutation(dur)
            r2 = np.array([cvs[k][idx[k]+d_rand[k]]/cvs[k][idx[k]]-1 for k in range(len(ents))])
            m2 = np.array([(cvs[k][idx[k]:idx[k]+d_rand[k]+1]/cvs[k][idx[k]]-1).min() for k in range(len(ents))])
            diffs.append(ret.mean()-r2.mean()); ddd.append(mdd.mean()-m2.mean())
        diffs=np.array(diffs)*100; ddd=np.array(ddd)*100
        print(f'{name:<18} 保有{dur.mean():5.1f}日  リターン差(ルール-ランダム) {diffs.mean():+6.2f}pt [90%範囲 {np.percentile(diffs,5):+.2f}~{np.percentile(diffs,95):+.2f}]   最大DD差 {ddd.mean():+6.2f}pt (正=ルールの方が浅い)')
names = ['trail 3σ','trail 4σ','trail 6σ','SMA20割れ','rawSELL(トレンド)']
timing_test(every5, 'plain 全日エントリー', ['plain'], names)
timing_test(onset('トレンドフォロー'), 'トレンドBUY onset (plain+long_lev)', ['plain','long_lev'], names)
