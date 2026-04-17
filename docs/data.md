# Data Layer / 数据层

## Raw Data / 原始数据

| File | Size | Description |
|------|------|-------------|
| `datasets/QQQ_1min_adjusted.parquet` | 75 MB | Split/dividend adjusted 1-min OHLCV |
| `datasets/QQQ_1min_unadjusted.parquet` | 70 MB | Raw unadjusted prices |

- **Coverage**: 2000-01-03 to 2026-02-20
- **Total bars**: 4,124,413 candlesticks across 6,572 trading days
- **Sessions**:
  - Regular hours: 9:30 – 15:59 (standard US market session)
  - Pre-market: 4:00 – 9:29 (early morning session)
  - After-hours: 16:01 – 19:59 (extended evening session)

> **覆盖范围**：2000年1月3日至2026年2月20日
> **总K线数**：4,124,413根，横跨6,572个交易日
> **交易时段**：
> - 常规交易时段：9:30–15:59（美股标准交易时间）
> - 盘前交易：4:00–9:29（早盘前交易时段）
> - 盘后交易：16:01–19:59（收盘后延长交易时段）

### Adjusted vs Unadjusted / 调整版 vs 未调整版

| Aspect | Adjusted | Unadjusted |
|--------|----------|-----------|
| Stock splits | Retroactively adjusted | Raw prices at time of trade |
| Dividends | Adjusted for cash distributions | Not adjusted |
| Use case | ML training, backtesting | Reference, sanity checks |
| Continuity | Smooth — no jumps on split dates | Has discontinuities at split events |

> **调整版**：回溯调整了拆股和分红，价格连续无跳空，适用于机器学习训练和回测
> **未调整版**：保留原始交易价格，在拆股日期有不连续跳变，用于参考和数据验证

---

## Daily Metrics / 日线指标

`data.daily_metrics` aggregates 1-minute data into 35 daily fields.

> `data.daily_metrics` 将1分钟数据聚合为35个日线字段。

```python
from data import load_1min_data, build_daily_metrics

raw = load_1min_data(Path("datasets/QQQ_1min_adjusted.parquet"))
daily = build_daily_metrics(raw)  # -> DataFrame (6572 x 35)
```

### Fields Produced / 生成的字段

| Category | Fields | Description |
|----------|--------|-------------|
| Regular session OHLCV | `reg_open`, `reg_high`, `reg_low`, `reg_close`, `volume_regular` | Standard 9:30-16:00 aggregation |
| Pre-market | `premarket_open/high/low/close`, `volume_premarket`, `premarket_return`, `premarket_range` | 4:00-9:29 session metrics |
| Full day | `full_day_high`, `full_day_low` | All-session high/low |
| VWAP | `vwap` | Volume-weighted average price |
| Intraday extremes | `max_drawdown`, `max_runup` | Worst pullback / best rally within a single regular session |
| Derived returns | `close_to_close`, `open_to_close`, `intraday_range`, `gap_return` | Key return metrics |
| Absolute returns | `abs_close_to_close`, `abs_open_to_close`, etc. | Unsigned magnitude of moves |
| Large move flags | `is_large_move_{1,2,3,5}pct` | Binary indicators: did the move exceed threshold? |

> | 类别 | 字段 | 说明 |
> |------|------|------|
> | 常规OHLCV | `reg_open/high/low/close`, `volume_regular` | 9:30-16:00 标准聚合 |
> | 盘前 | `premarket_*`, `volume_premarket`, `premarket_return/range` | 4:00-9:29 盘前指标 |
> | 全天 | `full_day_high/low` | 所有时段的最高/最低价 |
> | VWAP | `vwap` | 成交量加权平均价 |
> | 日内极值 | `max_drawdown`, `max_runup` | 单日内最大回撤/最大涨幅 |
> | 衍生收益率 | `close_to_close`, `open_to_close`, `intraday_range`, `gap_return` | 关键收益指标 |
> | 绝对收益率 | `abs_close_to_close` 等 | 波动幅度（无方向） |
> | 大波动标记 | `is_large_move_{1,2,3,5}pct` | 二元标记：波动是否超过阈值？ |

### Return Statistics / 收益率统计

| Metric | Mean | Std | Kurtosis | Note |
|--------|------|-----|----------|------|
| Close-to-close | 0.04% | 1.69% | 7.87 | Fat tails >> 3.0 (Gaussian) |
| Open-to-close | -0.001% | 1.44% | 12.14 | Even fatter tails |
| Intraday range | 1.87% | 1.50% | 32.83 | Extremely leptokurtic |
| Gap return | 0.04% | 0.79% | 14.63 | — |

> **峰度（Kurtosis）** 远大于正态分布的3.0，说明QQQ的收益率具有极端的"肥尾"特性——极端波动发生的频率远超正态假设。日内范围（range）的峰度高达32.83，是最"尖峰厚尾"的指标。

### Large Move Frequencies / 大幅波动频率

| Threshold | \|C2C\| | \|O2C\| | Range |
|-----------|---------|---------|-------|
| > 1% | 38.9% | 27.5% | 57.7% |
| > 2% | 16.1% | 11.5% | 30.9% |
| > 3% | 7.4% | 5.0% | 15.7% |
| > 5% | 2.3% | 1.5% | 5.8% |

> - **C2C (Close-to-Close)**：收盘到收盘的绝对涨跌幅
> - **O2C (Open-to-Close)**：开盘到收盘的绝对涨跌幅
> - **Range**：日内最高最低价之差占比
> - 例如：16.1%的交易日中，QQQ的收盘收益绝对值超过2%。Range >2% 的频率是 C2C 的近两倍（30.9% vs 16.1%），因为日内波动可以先涨后跌，Range 捕获的是总波幅。

---

## External Data / 外部市场数据

`data.external_data` downloads and caches VIX/VVIX/Treasury yields from Yahoo Finance.

> `data.external_data` 从 Yahoo Finance 下载并缓存 VIX/VVIX/国债收益率数据。

```python
from data import download_external_data

ext = download_external_data()  # cached in output/external_data.parquet
# Force refresh: download_external_data(force=True)
```

| Ticker | Field Prefix | Available From | Description |
|--------|-------------|----------------|-------------|
| ^VIX | `vix_` | 2000 | CBOE Volatility Index — implied vol of S&P 500 options (30-day) |
| ^VVIX | `vvix_` | 2007 | Volatility of VIX — how volatile is VIX itself |
| ^TNX | `tnx_10y_` | 2000 | 10-Year Treasury Note yield |
| ^IRX | `irx_3m_` | 2000 | 3-Month Treasury Bill yield |
| ^FVX | `fvx_5y_` | 2000 | 5-Year Treasury Note yield |

> | 代码 | 字段前缀 | 起始年份 | 说明 |
> |------|---------|---------|------|
> | ^VIX | `vix_` | 2000 | CBOE波动率指数——标普500期权的30天隐含波动率 |
> | ^VVIX | `vvix_` | 2007 | VIX的波动率——衡量VIX本身的波动程度 |
> | ^TNX | `tnx_10y_` | 2000 | 10年期美国国债收益率 |
> | ^IRX | `irx_3m_` | 2000 | 3个月期美国国债收益率 |
> | ^FVX | `fvx_5y_` | 2000 | 5年期美国国债收益率 |

Each ticker provides `open`, `high`, `low`, `close` columns (4 per ticker, 20 total, but only open/high/close are used → 15 effective columns).

> 每个代码提供 `open/high/low/close` 四列。实际使用中主要取 open/high/close（每个3列，共15列有效列）。

---

## Event Calendar / 事件日历

`data.event_calendar` provides FOMC, NFP, and earnings season dates.

> `data.event_calendar` 提供 FOMC、非农就业报告（NFP）和财报季日期。

```python
from data import load_fomc_dates, compute_nfp_dates

fomc = load_fomc_dates()              # 216 dates from datasets/fomc_dates.csv
nfp = compute_nfp_dates(2000, 2026)   # first Friday of each month
```

### Event Types / 事件类型

| Event | Source | Frequency | Impact |
|-------|--------|-----------|--------|
| **FOMC** | `datasets/fomc_dates.csv` (manual) | 8x/year (+ unscheduled) | Highest — rate decisions move all risk assets |
| **NFP** | Algorithmically computed (1st Friday of month) | 12x/year | High — labor market data, often surprises |
| **Earnings Season** | Rule-based (late Jan/Apr/Jul/Oct + first 2 weeks next month) | 4x/year | Medium — elevated vol from tech mega-caps |

> | 事件 | 数据来源 | 频率 | 影响程度 |
> |------|---------|------|---------|
> | **FOMC** | `datasets/fomc_dates.csv`（手动维护） | 每年8次（+不定期紧急会议） | 最高——利率决议影响所有风险资产 |
> | **NFP** | 算法计算（每月第一个周五） | 每年12次 | 高——就业数据经常超预期 |
> | **财报季** | 规则判定（1/4/7/10月下旬 + 次月前两周） | 每年4次 | 中——科技巨头财报带来波动 |

### Derived Fields / 衍生字段

| Field | Description |
|-------|-------------|
| `is_fomc_day` / `is_fomc_eve` | FOMC announcement day / the business day before |
| `days_to_fomc` | Integer days until next FOMC (capped at 60) |
| `fomc_week` | Binary: is FOMC within this calendar week? |
| `is_nfp_day` / `is_nfp_eve` | NFP release day / the business day before |
| `days_to_nfp` | Integer days until next NFP (capped at 40) |
| `is_macro_event_day` / `is_macro_event_eve` | Either FOMC or NFP |
| `is_earnings_season` | Binary: within earnings reporting window |

> | 字段 | 说明 |
> |------|------|
> | `is_fomc_day` / `is_fomc_eve` | 是否为FOMC公布日 / FOMC前一个交易日 |
> | `days_to_fomc` | 距下次FOMC的天数（上限60天） |
> | `fomc_week` | 本周是否有FOMC |
> | `is_nfp_day` / `is_nfp_eve` | 是否为非农公布日 / 非农前一个交易日 |
> | `days_to_nfp` | 距下次非农的天数（上限40天） |
> | `is_macro_event_day/eve` | 是否为宏观事件日（FOMC或NFP） |
> | `is_earnings_season` | 是否处于财报季窗口 |

### Maintaining FOMC Dates / 维护 FOMC 日期

To add new FOMC dates, simply append rows to `datasets/fomc_dates.csv`. The file contains one date per line (YYYY-MM-DD format). Both scheduled and emergency meetings should be included.

> 新增 FOMC 日期只需在 `datasets/fomc_dates.csv` 中追加行即可。文件格式为每行一个日期（YYYY-MM-DD）。定期会议和紧急会议都应包含在内。
