"""押し目候補の「反転確認エントリー」検証 (事前登録: docs/signal_accuracy_2026-07.md 2026-11-21)。

使い方: python3 confirm_study.py  (results/ の履歴 CSV 2 本を使う。ネットワーク不要)
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from signal_tracker import adjust_for_splits
a = pd.read_csv('results/signals_history_backfill.csv', parse_dates=['signal_date'])
l = pd.read_csv('results/signals_history.csv', parse_dates=['signal_date'])
allr = pd.concat([a, l[l.signal_date > a.signal_date.max()]])
cl = a.drop_duplicates('ticker').set_index('ticker').ticker_class
tk = [t for t in cl.index if cl[t] != 'inverse_lev']
P = allr.pivot_table(index='signal_date', columns='ticker', values='close', aggfunc='last').sort_index().apply(adjust_for_splits)[tk]
rsi = allr[allr.strategy == 'トレンドフォロー'].pivot_table(index='signal_date', columns='ticker', values='rsi', aggfunc='last').reindex(P.index)[tk]
setup = (P > P.rolling(200).mean()) & (rsi < 35)
onset = setup & ~setup.shift(1, fill_value=False)

def confirm(onset, window=5, k=2):
    out = pd.DataFrame(False, index=P.index, columns=tk); skipped = total = 0
    cv = P.values
    for j, t in enumerate(tk):
        for i in np.where(onset[t].values)[0]:
            total += 1; hit = False
            for d in range(i + 1, min(i + 1 + window, len(P))):
                if d - k >= 0 and all(cv[d, j] > cv[d - m, j] for m in range(1, k + 1)):
                    out.iat[d, j] = True; hit = True; break
            skipped += (not hit)
    return out, total, skipped

F = {h: P.shift(-h) / P - 1 for h in (10, 20)}
D = {h: F[h].sub(F[h].mean()) for h in F}
design = P.index <= pd.Timestamp('2026-09-02'); unseen = P.index > pd.Timestamp('2026-09-02')
def stats(m, mk, h):
    r = F[h][m].loc[mk].stack().dropna(); d = D[h][m].loc[mk].stack().dropna()
    base = F[h].loc[mk].stack().dropna().gt(0).mean() * 100
    return len(r), ((r > 0).mean() * 100 - base) if len(r) else np.nan, d.mean() * 100 if len(d) else np.nan
def episodes(m, mk):
    ds = sorted({d for d in m.loc[mk].index[m.loc[mk].any(axis=1)] if not F[10][m].loc[[d]].stack().dropna().empty})
    eps = []; 
    for d in ds:
        if eps and P.index.get_loc(d) - P.index.get_loc(eps[-1][-1]) <= 5: eps[-1].append(d)
        else: eps.append([d])
    return eps

conf, total, skipped = confirm(onset)
print(f'データ: {P.index.min().date()}〜{P.index.max().date()} / 27銘柄 / 押し目onset {total}件 → 確認あり {total-skipped}件 (見送り {skipped/total*100:.0f}%)')
for name, m in (('押し目候補 onset (基準)', onset), ('反転確認エントリー (登録ルール)', conf)):
    print(f'\n== {name} ==  週あたり {m.values.sum()/len(P)*5:.2f}')
    for lbl, mk in (('設計時の範囲 〜09-02', design), ('未知の期間 09-03〜', unseen)):
        for h in (10, 20):
            n, wd, ex = stats(m, mk, h)
            print(f'  {lbl:<16} {h}d n={n:3d} 勝率差{wd:+6.1f}pt 超過{ex:+6.2f}pt')
    eps = episodes(m, unseen)
    for e in eps:
        rr = F[10][m].loc[e].stack().dropna(); print(f'    未知期間エピソード {e[0].date()}〜{e[-1].date()} {len(rr)}件 10d平均{rr.mean()*100:+.1f}% 上昇{(rr>0).mean()*100:.0f}%')
    yrs = []
    for y in (2023, 2024, 2025, 2026):
        n, wd, ex = stats(m, P.index.year == y, 10); yrs.append(f'{y}:{ex:+.2f}(n{n})')
    print('  年別10d超過:', ' '.join(yrs))

print('\n-- 感度確認 (参考値。判定には使わない) --')
for w, k in ((3, 2), (5, 1), (5, 3), (10, 2)):
    m, tt, sk = confirm(onset, w, k)
    n1, _, e1 = stats(m, design, 10); n2, w2, e2 = stats(m, unseen, 10)
    print(f'  {w}日以内・直前{k}本超え: 見送り{sk/tt*100:3.0f}%  設計範囲 超過{e1:+.2f}(n{n1})  未知期間 超過{e2:+.2f} 勝率差{w2:+.1f}(n{n2})')
