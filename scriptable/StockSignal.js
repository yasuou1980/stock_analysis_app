// Variables used by Scriptable.
// icon-color: green; icon-glyph: chart-line;
//
// stock_analysis_app のシグナルロジック（backtester.py の「トレンドフォロー」「逆張り」）を
// JavaScript に移植した Scriptable 用スクリプト。
// yfinance の代わりに Yahoo Finance の chart API を直接叩いてデータを取得する。
//
// 使い方:
//   - Scriptable アプリでこのスクリプトを実行すると、ティッカー入力ダイアログが出る
//     （前回入力した銘柄がデフォルトで入っているので、差し替えて連続チェックできる）
//   - ホーム画面ウィジェットとして追加した場合は、ウィジェットパラメータに
//     ティッカーシンボル（例: "NVDA"）を指定する。省略時は前回値 or "AAPL"

const LAST_TICKER_KEY = "stock_signal_last_ticker";

// batch_runner.py のデフォルトパラメータ（スイングトレード設定）と同一
const PARAMS = {
  short_window: 10, long_window: 40, rsi_period: 10,
  macd_fast: 10, macd_slow: 20, macd_signal: 7,
  bb_length: 20, bb_std: 2.0, stoch_k: 14, stoch_d: 3,
  dev_upper: 10, dev_lower: -10,
  rsi_upper: 70, rsi_lower: 30,
  stoch_upper: 80, stoch_lower: 20,
  score_smooth_period: 3, ema_slope_period: 5,
  adx_threshold: 20,
};

// 商品クラス定義 (config.toml の [ticker_classes] と同期すること)
// inverse_lev: 構造的減価があるため BUY シグナル禁止
// long_lev:    急落後の反発が大きいため SELL シグナルを制限
const TICKER_CLASSES = {
  inverse_lev: ["SOXS", "SQQQ", "SPXS"],
  long_lev: ["SOXL", "TQQQ", "UPRO", "TECL", "TSLL", "NUGT", "FNGG"],
};

function resolveTickerClass(ticker) {
  const t = ticker.toUpperCase();
  for (const cls of Object.keys(TICKER_CLASSES)) {
    if (TICKER_CLASSES[cls].includes(t)) return cls;
  }
  return "plain";
}

// backtester.py compute_signal_gates() の移植。
// 実測 (results/signals_history.csv の onset 分析) に基づく負けパターン遮断:
// - インバース型レバETFの BUY 禁止 (逆張りBUY 勝率19% 平均-13.9%)
// - ロング型レバETFのトレンドSELL 禁止 (勝率7-18% 平均-12〜-18%)
// - 5日で-12%超の急落直後の SELL 禁止 (投げ売りの底で売らない)
// - 逆張りSELLは RSI>=50 かつ 乖離率-3〜+15% かつ 急落直後でない場合のみ
function computeSignalGates(tickerClass, ind, i) {
  const { close, rsi, deviation } = ind;
  const ret5 = i >= 5 && close[i - 5] ? close[i] / close[i - 5] - 1 : 0;
  const isInverse = tickerClass === "inverse_lev";
  const isLongLev = tickerClass === "long_lev";
  const crashCooldown = ret5 < -0.12;

  const buyOk = !isInverse;
  const trendSellOk = isInverse ? true : isLongLev ? false : !crashCooldown;
  const counterSellOk = !isLongLev
    && rsi[i] >= 50 && deviation[i] >= -3 && deviation[i] <= 15 && ret5 > -0.10;
  return { buyOk, trendSellOk, counterSellOk };
}

// ---------------------------------------------------------------------------
// 配列ユーティリティ（pandas の shift / rolling / ewm 相当）
// ---------------------------------------------------------------------------
function shift(arr, n) {
  const out = new Array(arr.length).fill(NaN);
  for (let i = 0; i < arr.length; i++) {
    const j = i - n;
    if (j >= 0 && j < arr.length) out[i] = arr[j];
  }
  return out;
}

function diff(arr) {
  const out = new Array(arr.length).fill(NaN);
  for (let i = 1; i < arr.length; i++) out[i] = arr[i] - arr[i - 1];
  return out;
}

function rollingMean(arr, w) {
  const out = new Array(arr.length).fill(NaN);
  for (let i = w - 1; i < arr.length; i++) {
    let s = 0, ok = true;
    for (let k = i - w + 1; k <= i; k++) { if (isNaN(arr[k])) { ok = false; break; } s += arr[k]; }
    out[i] = ok ? s / w : NaN;
  }
  return out;
}

function rollingStd(arr, w) {
  const mean = rollingMean(arr, w);
  const out = new Array(arr.length).fill(NaN);
  for (let i = w - 1; i < arr.length; i++) {
    if (isNaN(mean[i])) continue;
    let s = 0;
    for (let k = i - w + 1; k <= i; k++) s += (arr[k] - mean[i]) ** 2;
    out[i] = Math.sqrt(s / (w - 1));
  }
  return out;
}

function rollingMax(arr, w) {
  const out = new Array(arr.length).fill(NaN);
  for (let i = w - 1; i < arr.length; i++) {
    let m = -Infinity;
    for (let k = i - w + 1; k <= i; k++) m = Math.max(m, arr[k]);
    out[i] = m;
  }
  return out;
}

function rollingMin(arr, w) {
  const out = new Array(arr.length).fill(NaN);
  for (let i = w - 1; i < arr.length; i++) {
    let m = Infinity;
    for (let k = i - w + 1; k <= i; k++) m = Math.min(m, arr[k]);
    out[i] = m;
  }
  return out;
}

// ewm(adjust=False): y0 = x0, y[t] = alpha*x[t] + (1-alpha)*y[t-1]
function ewm(arr, alpha) {
  const out = new Array(arr.length).fill(NaN);
  let prev = null;
  for (let i = 0; i < arr.length; i++) {
    const x = arr[i];
    if (prev === null) { prev = x; } else { prev = alpha * x + (1 - alpha) * prev; }
    out[i] = prev;
  }
  return out;
}

function emaSpan(arr, span) { return ewm(arr, 2 / (span + 1)); }

// ---------------------------------------------------------------------------
// テクニカル指標（backtester.py の _ema/_rsi/_macd/_bbands/_adx/_stoch/_atr/_psar 相当）
// ---------------------------------------------------------------------------
function rsiCalc(close, length) {
  const delta = diff(close);
  const gain = delta.map((d) => (isNaN(d) ? 0 : d > 0 ? d : 0));
  const loss = delta.map((d) => (isNaN(d) ? 0 : d < 0 ? -d : 0));
  const avgGain = ewm(gain, 1 / length);
  const avgLoss = ewm(loss, 1 / length);
  return avgGain.map((g, i) => {
    const l = avgLoss[i];
    if (l === 0) return 100;
    const rs = g / l;
    return 100 - 100 / (1 + rs);
  });
}

function macdCalc(close, fast, slow, signal) {
  const emaFast = emaSpan(close, fast);
  const emaSlow = emaSpan(close, slow);
  const macd = close.map((_, i) => emaFast[i] - emaSlow[i]);
  const macds = emaSpan(macd, signal);
  const macdh = macd.map((v, i) => v - macds[i]);
  return { macd, macdh, macds };
}

function bbandsCalc(close, length, std) {
  const bbm = rollingMean(close, length);
  const sd = rollingStd(close, length);
  const bbu = bbm.map((m, i) => m + std * sd[i]);
  const bbl = bbm.map((m, i) => m - std * sd[i]);
  return { bbl, bbm, bbu };
}

function trueRange(high, low, close) {
  const prevClose = shift(close, 1);
  const tr = new Array(close.length).fill(0);
  for (let i = 1; i < close.length; i++) {
    tr[i] = Math.max(high[i] - low[i], Math.abs(high[i] - prevClose[i]), Math.abs(low[i] - prevClose[i]));
  }
  return tr;
}

function adxCalc(high, low, close, length) {
  const n = close.length;
  const prevHigh = shift(high, 1), prevLow = shift(low, 1);
  const plusDM = new Array(n).fill(0), minusDM = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const up = high[i] - prevHigh[i];
    const down = prevLow[i] - low[i];
    plusDM[i] = up > down ? Math.max(up, 0) : 0;
    minusDM[i] = down > up ? Math.max(down, 0) : 0;
  }
  const tr = trueRange(high, low, close);
  const atr = ewm(tr, 1 / length);
  const ewmPlus = ewm(plusDM, 1 / length);
  const ewmMinus = ewm(minusDM, 1 / length);
  const plusDI = new Array(n), minusDI = new Array(n), dx = new Array(n);
  for (let i = 0; i < n; i++) {
    plusDI[i] = atr[i] ? (100 * ewmPlus[i]) / atr[i] : 0;
    minusDI[i] = atr[i] ? (100 * ewmMinus[i]) / atr[i] : 0;
    const sum = plusDI[i] + minusDI[i];
    dx[i] = sum ? (Math.abs(plusDI[i] - minusDI[i]) / sum) * 100 : 0;
  }
  const adx = ewm(dx, 1 / length);
  return { adx, dmp: plusDI, dmn: minusDI };
}

function stochCalc(high, low, close, k, d) {
  const lowestLow = rollingMin(low, k);
  const highestHigh = rollingMax(high, k);
  const stochK = close.map((c, i) => {
    const range = highestHigh[i] - lowestLow[i];
    return range ? (100 * (c - lowestLow[i])) / range : NaN;
  });
  const stochD = rollingMean(stochK, d);
  return { stochK, stochD };
}

function atrCalc(high, low, close, length) {
  return ewm(trueRange(high, low, close), 1 / length);
}

function psarCalc(high, low, close, afStart = 0.02, afStep = 0.02, afMax = 0.2) {
  const n = close.length;
  const psar = new Array(n).fill(0);
  const psarDir = new Array(n).fill(1);
  if (n < 2) return { psar, psarDir };

  let ep;
  if (close[1] >= close[0]) { psarDir[0] = 1; psar[0] = low[0]; ep = high[0]; }
  else { psarDir[0] = -1; psar[0] = high[0]; ep = low[0]; }
  let af = afStart;

  for (let i = 1; i < n; i++) {
    const prevPsar = psar[i - 1];
    const prevDir = psarDir[i - 1];
    if (prevDir === 1) {
      let p = prevPsar + af * (ep - prevPsar);
      p = Math.min(p, low[i - 1]);
      if (i >= 2) p = Math.min(p, low[i - 2]);
      if (low[i] < p) { psarDir[i] = -1; psar[i] = ep; ep = low[i]; af = afStart; }
      else {
        psarDir[i] = 1; psar[i] = p;
        if (high[i] > ep) { ep = high[i]; af = Math.min(af + afStep, afMax); }
      }
    } else {
      let p = prevPsar + af * (ep - prevPsar);
      p = Math.max(p, high[i - 1]);
      if (i >= 2) p = Math.max(p, high[i - 2]);
      if (high[i] > p) { psarDir[i] = 1; psar[i] = ep; ep = high[i]; af = afStart; }
      else {
        psarDir[i] = -1; psar[i] = p;
        if (low[i] < ep) { ep = low[i]; af = Math.min(af + afStep, afMax); }
      }
    }
  }
  return { psar, psarDir };
}

// 週足トレンド判定 (pandas resample('W') 相当: 日曜締めの週足終値で EMA クロス判定)
function weeklyTrendUp(dates, close, wkShort, wkLong) {
  const map = new Map();
  for (let i = 0; i < dates.length; i++) {
    const d = dates[i];
    const dow = d.getUTCDay(); // 0 = Sun
    const addDays = (7 - dow) % 7;
    const weekEnd = new Date(d.getTime() + addDays * 86400000);
    map.set(weekEnd.toISOString().slice(0, 10), close[i]);
  }
  const weeklyClose = Array.from(map.values());
  if (weeklyClose.length < Math.max(wkLong, 5)) return true;
  const emaS = emaSpan(weeklyClose, wkShort);
  const emaL = emaSpan(weeklyClose, wkLong);
  return emaS[emaS.length - 1] > emaL[emaL.length - 1];
}

// ---------------------------------------------------------------------------
// トレンドフォロー戦略 (backtester.py L236-402 相当)
// ---------------------------------------------------------------------------
function computeTrendStrategy(ind, params, gates) {
  const { dates, close, high, low, volume, smaShort, smaLong, rsi, macd, macdh, bbl, bbu,
          adx, dmp, dmn, volSma, ema50, psar, psarDir } = ind;
  const n = close.length;
  const adxThreshold = params.adx_threshold;

  const adxScore = new Array(n), adxTrendFilter = new Array(n);
  for (let i = 0; i < n; i++) {
    const buy = adx[i] > adxThreshold && dmp[i] > dmn[i];
    const sell = adx[i] > adxThreshold && dmn[i] > dmp[i];
    adxScore[i] = buy ? 1.5 : sell ? -1.5 : 0;
    adxTrendFilter[i] = adx[i] >= adxThreshold;
  }

  const rsiZoneScore = rsi.map((v) => (v > 50 ? 0.5 : v < 50 ? -0.5 : 0));

  const smaShortPrev = shift(smaShort, 1), smaLongPrev = shift(smaLong, 1);
  const crossBonus = new Array(n);
  for (let i = 0; i < n; i++) {
    const up = smaShort[i] > smaLong[i] && smaShortPrev[i] <= smaLongPrev[i];
    const down = smaShort[i] < smaLong[i] && smaShortPrev[i] >= smaLongPrev[i];
    crossBonus[i] = up ? 1.0 : down ? -1.0 : 0;
  }

  const recentHigh20 = shift(rollingMax(close, 20), 1);
  const recentLow20 = shift(rollingMin(close, 20), 1);
  const recentHigh50 = shift(rollingMax(close, 50), 1);
  const recentHigh100 = shift(rollingMax(close, 100), 1);
  const breakoutScore = new Array(n);
  for (let i = 0; i < n; i++) {
    const bo20 = close[i] > recentHigh20[i] ? 1.5 : 0;
    const bo50 = close[i] > recentHigh50[i] ? 1.0 : 0;
    const bo100 = close[i] > recentHigh100[i] ? 1.0 : 0;
    const boDown20 = close[i] < recentLow20[i] ? -1.5 : 0;
    breakoutScore[i] = bo20 + bo50 + bo100 + boDown20;
  }

  const bbTrendScore = close.map((c, i) => (c > bbu[i] ? 0.5 : c < bbl[i] ? -0.5 : 0));

  const macdPrev = shift(macd, 1);
  const macdZeroCrossScore = new Array(n);
  for (let i = 0; i < n; i++) {
    const up = macd[i] > 0 && macdPrev[i] <= 0;
    const down = macd[i] < 0 && macdPrev[i] >= 0;
    macdZeroCrossScore[i] = up ? 2.0 : down ? -2.0 : 0;
  }

  const rsiPrev = shift(rsi, 1);
  const rsiReversalScore = new Array(n);
  for (let i = 0; i < n; i++) {
    const up = rsi[i] > 35 && rsiPrev[i] <= 35;
    const down = rsi[i] < 65 && rsiPrev[i] >= 65;
    rsiReversalScore[i] = up ? 2.0 : down ? -2.0 : 0;
  }

  const rollMaxClose15 = shift(rollingMax(close, 15), 1);
  const rollMaxMacdh15 = shift(rollingMax(macdh, 15), 1);
  const rollMaxRsi15 = shift(rollingMax(rsi, 15), 1);
  const bearishDivScore = new Array(n);
  for (let i = 0; i < n; i++) {
    const rmc = isNaN(rollMaxClose15[i]) ? 0 : rollMaxClose15[i];
    const rmm = isNaN(rollMaxMacdh15[i]) ? 0 : rollMaxMacdh15[i];
    const rmr = isNaN(rollMaxRsi15[i]) ? 100 : rollMaxRsi15[i];
    const atHigh = close[i] >= rmc * 0.97;
    const macdDiverging = macdh[i] > 0 && rmm > 0 && macdh[i] < rmm * 0.7;
    const rsiDiverging = atHigh && rsi[i] > 50 && rsi[i] < rmr * 0.9;
    const doubleDiv = atHigh && macdDiverging && rsiDiverging;
    bearishDivScore[i] = doubleDiv ? -3.5 : atHigh && macdDiverging ? -2.0 : rsiDiverging ? -1.5 : 0;
  }

  const slopePeriod = params.ema_slope_period;
  const smaShortShiftP = shift(smaShort, slopePeriod);
  const emaSlope = new Array(n);
  for (let i = 0; i < n; i++) {
    const base = smaShortShiftP[i];
    emaSlope[i] = base && !isNaN(base) ? ((smaShort[i] - base) / base) * 100 : 0;
  }
  const emaSlopeScore = emaSlope.map((v) => (v > 1.0 ? 1.0 : v > 0.3 ? 0.5 : v < -1.0 ? -1.0 : v < -0.3 ? -0.5 : 0));

  const closePrev = shift(close, 1), psarPrev = shift(psar, 1);
  const psarEventScore = new Array(n), psarStateScore = new Array(n);
  for (let i = 0; i < n; i++) {
    const bull = close[i] > psar[i] && closePrev[i] <= psarPrev[i];
    const bear = close[i] < psar[i] && closePrev[i] >= psarPrev[i];
    psarEventScore[i] = bull ? 1.5 : bear ? -3.5 : 0;
    psarStateScore[i] = psarDir[i] === 1 ? 0.5 : -0.5;
  }

  const smaLongShift5 = shift(smaLong, 5);
  const rsiRollMin5Shift1 = shift(rollingMin(rsi, 5), 1);
  const pullbackScore = new Array(n);
  for (let i = 0; i < n; i++) {
    const longMaRising = smaLong[i] > smaLongShift5[i];
    const uptrendConfirmed = close[i] > smaLong[i] && longMaRising;
    const rsiRecentLow = rsiRollMin5Shift1[i] < 40;
    const rsiCrossUp50 = rsi[i] >= 50 && rsiPrev[i] < 50;
    pullbackScore[i] = uptrendConfirmed && rsiRecentLow && rsiCrossUp50 ? 2.8 : 0;
  }

  const eventScores = new Array(n), stateScores = new Array(n);
  for (let i = 0; i < n; i++) {
    const emaStateScore = smaShort[i] > smaLong[i] ? 0.5 : -0.5;
    const macdHistScore = macdh[i] > 0 ? 1.5 : -1.5;
    stateScores[i] = emaStateScore + rsiZoneScore[i] + macdHistScore + adxScore[i] + bbTrendScore[i] + emaSlopeScore[i] + psarStateScore[i];
    eventScores[i] = crossBonus[i] + breakoutScore[i] + macdZeroCrossScore[i] + rsiReversalScore[i] + bearishDivScore[i] + psarEventScore[i] + pullbackScore[i];
  }
  const smoothPeriod = params.score_smooth_period;
  const smoothedEvents = emaSpan(eventScores, smoothPeriod).map((v) => v * (smoothPeriod * 0.6));
  const scores = new Array(n);
  for (let i = 0; i < n; i++) scores[i] = adxTrendFilter[i] ? stateScores[i] + smoothedEvents[i] : (stateScores[i] + smoothedEvents[i]) * 0.5;

  const volCondition = volume.map((v, i) => (isNaN(volSma[i]) ? true : v > volSma[i] * 1.2));

  const wkShort = Math.max(Math.floor(params.short_window / 5), 2);
  const wkLong = Math.max(Math.floor(params.long_window / 5), 3);
  const weeklyUp = weeklyTrendUp(dates, close, wkShort, wkLong);

  const i = n - 1; // 最新日のみ判定すればよい
  const buyThreshold = adx[i] > 30 ? 3.5 : adx[i] > adxThreshold ? 4.5 : 5.5;
  const sellThreshold = adx[i] > 30 ? -3.0 : adx[i] > adxThreshold ? -4.0 : -5.0;
  const buyThresholdAdj = buyThreshold + (weeklyUp ? 0 : 1.5);
  const sellThresholdAdj = sellThreshold + (weeklyUp ? -1.5 : 0);

  const structureBroken = close[i] < ema50[i];
  const buy = scores[i] >= buyThresholdAdj && volCondition[i] && gates.buyOk;
  const sell = scores[i] <= sellThresholdAdj && structureBroken && gates.trendSellOk;

  return { signal: buy ? "BUY" : sell ? "SELL" : "HOLD", score: scores[i] };
}

// ---------------------------------------------------------------------------
// 逆張り戦略 (backtester.py L432-549 相当)
// ---------------------------------------------------------------------------
function computeCounterStrategy(ind, params, gates) {
  const { close, smaShort, smaLong, rsi, macdh, bbl, bbu, deviation, stochK, stochD, ema200 } = ind;
  const n = close.length;
  const { stoch_upper: stochUpper, stoch_lower: stochLower, rsi_upper: rsiUpper, rsi_lower: rsiLower } = params;

  const stochKPrev = shift(stochK, 1), stochDPrev = shift(stochD, 1);
  const rsiPrev = shift(rsi, 1);
  const closePrev = shift(close, 1);
  const smaShortPrev = shift(smaShort, 1);
  const macdhPrev = shift(macdh, 1);
  const smaLongShift10 = shift(smaLong, 10);
  const recentLow5Shift1 = shift(rollingMin(close, 5), 1);
  const slopePeriod = params.ema_slope_period;
  const smaShortShiftP = shift(smaShort, slopePeriod);

  const counterScores = new Array(n);
  const emaSlope = new Array(n);
  for (let i = 0; i < n; i++) {
    const base = smaShortShiftP[i];
    emaSlope[i] = base && !isNaN(base) ? ((smaShort[i] - base) / base) * 100 : 0;

    const stochBuyCross = stochK[i] > stochD[i] && stochKPrev[i] <= stochDPrev[i] && stochK[i] < stochLower;
    const stochExitOverbought = stochK[i] < stochUpper && stochKPrev[i] >= stochUpper;
    const stochCrossScore = stochBuyCross ? 2.0 : stochExitOverbought ? -2.0 : 0;
    const stochStateScore = stochK[i] < stochLower ? 1.0 : stochExitOverbought ? -1.0 : 0;

    const rsiReboundBuy = rsi[i] < rsiLower && rsi[i] > rsiPrev[i];
    const rsiExitOverbought = rsi[i] < rsiUpper && rsiPrev[i] >= rsiUpper;
    const rsiReboundScore = rsiReboundBuy ? 2.0 : rsiExitOverbought ? -2.0 : 0;
    const rsiStateScore = rsi[i] < rsiLower ? 1.0 : rsiExitOverbought ? -1.0 : 0;

    const bbScore = close[i] < bbl[i] ? 1.5 : close[i] > bbu[i] ? -1.5 : 0;
    const devScore = deviation[i] < params.dev_lower ? 1.5 : deviation[i] > params.dev_upper ? -1.5 : 0;

    const uptrendMomentum = smaShort[i] > smaLong[i] && emaSlope[i] > 0;
    const allScores = [stochCrossScore, stochStateScore, rsiReboundScore, rsiStateScore, bbScore, devScore];
    const sellComponent = allScores.reduce((s, v) => s + Math.min(v, 0), 0);
    const buyComponent = allScores.reduce((s, v) => s + Math.max(v, 0), 0);
    const dampening = uptrendMomentum ? 0.5 : 1.0;
    counterScores[i] = buyComponent + sellComponent * dampening;
  }

  const i = n - 1;
  const uptrendMomentum = smaShort[i] > smaLong[i] && emaSlope[i] > 0;

  const priceBreakDown = close[i] < recentLow5Shift1[i];
  const macdhPositive = macdh[i] > 0;
  const emaSlopeStrong = emaSlope[i] > 0.3;
  const sellThresholdC = macdhPositive && emaSlopeStrong ? -7.0 : !macdhPositive || emaSlope[i] < 0 ? -3.0 : -5.0;

  const longMaRisingC = smaLong[i] > smaLongShift10[i];
  const longTrendOk = close[i] > ema200[i] || longMaRisingC;
  const fallingKnife = deviation[i] < -15 || rsi[i] < 20;
  const reboundConfirm = close[i] > closePrev[i] || rsi[i] > rsiPrev[i];

  let oversoldRecent = -Infinity;
  for (let k = Math.max(0, i - 2); k <= i; k++) oversoldRecent = Math.max(oversoldRecent, counterScores[k]);
  const buySignalC = oversoldRecent >= 5.0 && longTrendOk && !fallingKnife && reboundConfirm && gates.buyOk;

  const sellSignalC = counterScores[i] <= sellThresholdC && priceBreakDown;

  let recentRsiMax = -Infinity;
  for (let k = Math.max(0, i - 4); k <= i; k++) recentRsiMax = Math.max(recentRsiMax, rsi[k]);
  const recentlyOverbought = recentRsiMax >= rsiUpper;
  const smaBreakDown = close[i] < smaShort[i] && closePrev[i] >= smaShortPrev[i];
  const profitTakeSignal = recentlyOverbought && smaBreakDown;
  const macdCrossDownC = macdh[i] < 0 && macdhPrev[i] >= 0;
  const earlySellSignalC = (profitTakeSignal || macdCrossDownC) && close[i] < smaShort[i];

  const strongUptrendBlock = uptrendMomentum && deviation[i] > 5;
  const sellFinalC = (sellSignalC || earlySellSignalC) && !strongUptrendBlock && gates.counterSellOk;

  return { signal: buySignalC ? "BUY" : sellFinalC ? "SELL" : "HOLD", score: counterScores[i] };
}

// ---------------------------------------------------------------------------
// データ取得 (Yahoo Finance chart API / yfinance auto_adjust=True 相当)
// ---------------------------------------------------------------------------
async function fetchDaily(ticker) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?range=1y&interval=1d`;
  const req = new Request(url);
  req.headers = { "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)" };
  const json = await req.loadJSON();
  if (!json.chart || !json.chart.result || json.chart.result.length === 0) {
    const msg = json.chart && json.chart.error ? json.chart.error.description : "データが見つかりません";
    throw new Error(`${ticker}: ${msg}`);
  }
  const result = json.chart.result[0];
  const ts = result.timestamp;
  const q = result.indicators.quote[0];
  const adjclose = result.indicators.adjclose ? result.indicators.adjclose[0].adjclose : q.close;

  const dates = [], open = [], high = [], low = [], close = [], volume = [];
  for (let i = 0; i < ts.length; i++) {
    if (q.close[i] == null || adjclose[i] == null) continue;
    const ratio = q.close[i] !== 0 ? adjclose[i] / q.close[i] : 1;
    dates.push(new Date(ts[i] * 1000));
    open.push(q.open[i] * ratio);
    high.push(q.high[i] * ratio);
    low.push(q.low[i] * ratio);
    close.push(adjclose[i]);
    volume.push(q.volume[i] || 0);
  }
  if (close.length < 210) {
    throw new Error(`${ticker}: 十分な過去データがありません（${close.length}日分）`);
  }
  return { dates, open, high, low, close, volume };
}

function buildIndicators(bars, params) {
  const { dates, high, low, close, volume } = bars;
  const smaShort = emaSpan(close, params.short_window);
  const smaLong = emaSpan(close, params.long_window);
  const rsi = rsiCalc(close, params.rsi_period);
  const { macd, macdh } = macdCalc(close, params.macd_fast, params.macd_slow, params.macd_signal);
  const { bbl, bbu } = bbandsCalc(close, params.bb_length, params.bb_std);
  const { adx, dmp, dmn } = adxCalc(high, low, close, 14);
  const volSma = rollingMean(volume, 20);
  const { stochK, stochD } = stochCalc(high, low, close, params.stoch_k, params.stoch_d);
  const ema50 = emaSpan(close, 50);
  const ema200 = emaSpan(close, 200);
  const { psar, psarDir } = psarCalc(high, low, close);
  const deviation = close.map((c, i) => (smaLong[i] ? ((c - smaLong[i]) / smaLong[i]) * 100 : 0));

  return { dates, close, high, low, volume, smaShort, smaLong, rsi, macd, macdh, bbl, bbu,
           adx, dmp, dmn, volSma, ema50, ema200, psar, psarDir, deviation, stochK, stochD };
}

async function analyzeTicker(ticker) {
  const bars = await fetchDaily(ticker);
  const ind = buildIndicators(bars, PARAMS);
  const i = bars.close.length - 1;
  const gates = computeSignalGates(resolveTickerClass(ticker), ind, i);
  const trend = computeTrendStrategy(ind, PARAMS, gates);
  const counter = computeCounterStrategy(ind, PARAMS, gates);
  return {
    ticker,
    date: bars.dates[i],
    close: bars.close[i],
    rsi: ind.rsi[i],
    deviation: ind.deviation[i],
    trend,
    counter,
  };
}

// ---------------------------------------------------------------------------
// 表示
// ---------------------------------------------------------------------------
const SIGNAL_EMOJI = { BUY: "🟢 BUY", SELL: "🔴 SELL", HOLD: "⚪ HOLD" };

function buildResultTable(r) {
  const table = new UITable();
  table.showSeparators = true;

  const header = new UITableRow();
  header.isHeader = true;
  header.addText(`${r.ticker}  ${r.close.toFixed(2)}`, `終値日: ${r.date.toISOString().slice(0, 10)}`);
  table.addRow(header);

  const info = new UITableRow();
  info.addText("RSI / 乖離率", `${r.rsi.toFixed(1)} / ${r.deviation >= 0 ? "+" : ""}${r.deviation.toFixed(2)}%`);
  table.addRow(info);

  for (const [label, res] of [["トレンドフォロー", r.trend], ["逆張り", r.counter]]) {
    const row = new UITableRow();
    row.height = 60;
    const cell = row.addText(label, `${SIGNAL_EMOJI[res.signal]}  (score: ${res.score.toFixed(1)})`);
    row.onSelect = () => {};
    table.addRow(row);
  }
  return table;
}

async function promptTicker(defaultTicker) {
  const alert = new Alert();
  alert.title = "銘柄シグナル確認";
  alert.message = "ティッカーシンボルを入力してください（例: AAPL, NVDA, SOXL）";
  alert.addTextField("ティッカー", defaultTicker);
  alert.addAction("実行");
  alert.addCancelAction("閉じる");
  const idx = await alert.presentAlert();
  if (idx === -1) return null;
  const value = alert.textFieldValue(0).trim().toUpperCase();
  return value || defaultTicker;
}

async function runInteractive() {
  let ticker = Keychain.contains(LAST_TICKER_KEY) ? Keychain.get(LAST_TICKER_KEY) : "AAPL";

  while (true) {
    const input = await promptTicker(ticker);
    if (!input) break;
    ticker = input;
    Keychain.set(LAST_TICKER_KEY, ticker);

    try {
      const result = await analyzeTicker(ticker);
      const table = buildResultTable(result);
      await table.present(false);
    } catch (e) {
      const err = new Alert();
      err.title = "エラー";
      err.message = String(e.message || e);
      err.addAction("OK");
      await err.presentAlert();
    }

    const again = new Alert();
    again.title = "続けますか？";
    again.addAction("別の銘柄を見る");
    again.addCancelAction("終了");
    const choice = await again.presentAlert();
    if (choice === -1) break;
  }
}

async function runWidget() {
  const ticker = (args.widgetParameter || (Keychain.contains(LAST_TICKER_KEY) ? Keychain.get(LAST_TICKER_KEY) : "AAPL")).toUpperCase();
  const widget = new ListWidget();
  widget.backgroundColor = new Color("#111111");
  try {
    const r = await analyzeTicker(ticker);
    const title = widget.addText(`${r.ticker}  ${r.close.toFixed(2)}`);
    title.textColor = Color.white();
    title.font = Font.boldSystemFont(16);
    widget.addSpacer(6);

    for (const [label, res] of [["トレンド", r.trend], ["逆張り", r.counter]]) {
      const row = widget.addStack();
      const l = row.addText(`${label}: `);
      l.textColor = Color.lightGray();
      l.font = Font.systemFont(13);
      const v = row.addText(SIGNAL_EMOJI[res.signal]);
      v.font = Font.systemFont(13);
      widget.addSpacer(2);
    }
    widget.addSpacer(6);
    const updated = widget.addText(`RSI ${r.rsi.toFixed(0)} / 乖離 ${r.deviation.toFixed(1)}%`);
    updated.textColor = Color.gray();
    updated.font = Font.systemFont(11);
  } catch (e) {
    const errText = widget.addText(`${ticker}: 取得エラー`);
    errText.textColor = Color.red();
  }
  Script.setWidget(widget);
}

// ---------------------------------------------------------------------------
// エントリーポイント
// ---------------------------------------------------------------------------
if (config.runsInWidget) {
  await runWidget();
} else {
  await runInteractive();
}
Script.complete();
