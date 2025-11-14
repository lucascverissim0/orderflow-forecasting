import { useEffect, useMemo, useState } from "react";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type SymbolId = string;

interface TimePoint {
  timestamp: string; // ISO
  symbol: string;
  close?: number;
  volume?: number;
  cvd?: number;
  pcr?: number;
  at_ask_bias?: number;
}

interface PredictionPoint {
  timestamp: string; // ISO
  symbol: string;
  pred_1d: number;
}

interface LatestRow {
  [key: string]: any;
  timestamp?: string;
  symbol?: string;
  pred_1d?: number;
}

export default function HomePage() {
  const [symbols, setSymbols] = useState<SymbolId[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<SymbolId | "">("");
  const [timeseries, setTimeseries] = useState<TimePoint[]>([]);
  const [predictions, setPredictions] = useState<PredictionPoint[]>([]);
  const [latest, setLatest] = useState<LatestRow | null>(null);

  const [loadingSymbols, setLoadingSymbols] = useState(false);
  const [loadingData, setLoadingData] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ---- Fetch helpers -------------------------------------------------------

  const fetchSymbols = async () => {
    setLoadingSymbols(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/symbols`);
      if (!res.ok) {
        throw new Error(`Failed to fetch symbols: ${res.status}`);
      }
      const data: SymbolId[] = await res.json();
      setSymbols(data || []);
      if (data && data.length > 0 && !selectedSymbol) {
        setSelectedSymbol(data[0]);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to fetch symbols");
    } finally {
      setLoadingSymbols(false);
    }
  };

  const fetchDataForSymbol = async (symbol: string) => {
    if (!symbol) return;
    setLoadingData(true);
    setError(null);
    try {
      const [tsRes, predRes, latestRes] = await Promise.all([
        fetch(`${API_BASE_URL}/timeseries?symbol=${encodeURIComponent(symbol)}`),
        fetch(
          `${API_BASE_URL}/predictions?symbol=${encodeURIComponent(symbol)}`
        ),
        fetch(`${API_BASE_URL}/latest?symbol=${encodeURIComponent(symbol)}`),
      ]);

      if (!tsRes.ok) {
        throw new Error(`Failed to fetch timeseries: ${tsRes.status}`);
      }
      if (!predRes.ok) {
        throw new Error(`Failed to fetch predictions: ${predRes.status}`);
      }
      if (!latestRes.ok) {
        throw new Error(`Failed to fetch latest: ${latestRes.status}`);
      }

      const tsData: TimePoint[] = await tsRes.json();
      const predData: PredictionPoint[] = await predRes.json();
      const latestData: LatestRow = await latestRes.json();

      setTimeseries(tsData || []);
      setPredictions(predData || []);
      setLatest(latestData || null);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to fetch data");
      setTimeseries([]);
      setPredictions([]);
      setLatest(null);
    } finally {
      setLoadingData(false);
    }
  };

  // ---- Effects -------------------------------------------------------------

  useEffect(() => {
    fetchSymbols();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (selectedSymbol) {
      fetchDataForSymbol(selectedSymbol);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSymbol]);

  // ---- Derived table combining price + preds -------------------------------

  const mergedRows = useMemo(() => {
    if (!timeseries.length) return [];

    const predByTimestamp = new Map<string, number>();
    for (const p of predictions) {
      predByTimestamp.set(p.timestamp, p.pred_1d);
    }

    return timeseries.map((row) => ({
      ...row,
      pred_1d: predByTimestamp.get(row.timestamp),
    }));
  }, [timeseries, predictions]);

  // ---- UI ------------------------------------------------------------------

  return (
    <main className="min-h-screen bg-slate-950 text-slate-50 px-6 py-8">
      <div className="max-w-5xl mx-auto space-y-6">
        <header className="flex flex-col gap-3 sm:flex-row sm:items-baseline sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              Orderflow Forecasting Dashboard
            </h1>
            <p className="text-sm text-slate-400">
              Batch features + 1-day signals for indices and metals.
            </p>
          </div>
          <span className="text-xs text-slate-500">
            Backend: <code>{API_BASE_URL}</code>
          </span>
        </header>

        {/* Controls */}
        <section className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2">
            <label htmlFor="symbol" className="text-sm text-slate-300">
              Symbol
            </label>
            <select
              id="symbol"
              className="bg-slate-900 border border-slate-700 rounded-md px-3 py-1 text-sm"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              disabled={loadingSymbols || !symbols.length}
            >
              {symbols.length === 0 && (
                <option value="">No symbols</option>
              )}
              {symbols.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            {loadingSymbols && (
              <span className="text-xs text-slate-500">Loading…</span>
            )}
          </div>

          <button
            className="self-start inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-900 px-3 py-1 text-xs font-medium text-slate-100 hover:bg-slate-800"
            onClick={() => {
              fetchSymbols();
              if (selectedSymbol) {
                fetchDataForSymbol(selectedSymbol);
              }
            }}
          >
            Refresh
          </button>
        </section>

        {/* Error / status */}
        {error && (
          <div className="rounded-md border border-red-500/40 bg-red-950/40 px-3 py-2 text-xs text-red-200">
            {error}
          </div>
        )}

        {/* Latest card */}
        <section className="grid gap-4 sm:grid-cols-3">
          <div className="col-span-2 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-sm font-medium text-slate-200 mb-2">
              Latest snapshot
            </h2>
            {latest && latest.timestamp ? (
              <div className="text-xs text-slate-300 space-y-1">
                <div>
                  <span className="text-slate-500 mr-1">Time:</span>
                  {latest.timestamp}
                </div>
                {latest.close !== undefined && (
                  <div>
                    <span className="text-slate-500 mr-1">Close:</span>
                    {latest.close}
                  </div>
                )}
                {latest.volume !== undefined && (
                  <div>
                    <span className="text-slate-500 mr-1">Volume:</span>
                    {latest.volume}
                  </div>
                )}
                {latest.cvd !== undefined && (
                  <div>
                    <span className="text-slate-500 mr-1">CVD:</span>
                    {latest.cvd}
                  </div>
                )}
                {latest.pcr !== undefined && (
                  <div>
                    <span className="text-slate-500 mr-1">Put/Call:</span>
                    {latest.pcr}
                  </div>
                )}
                {latest.at_ask_bias !== undefined && (
                  <div>
                    <span className="text-slate-500 mr-1">At-ask bias:</span>
                    {latest.at_ask_bias}
                  </div>
                )}
                {latest.pred_1d !== undefined && (
                  <div>
                    <span className="text-slate-500 mr-1">Pred 1d:</span>
                    {latest.pred_1d}
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500">
                No latest data for this symbol yet.
              </p>
            )}
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-sm font-medium text-slate-200 mb-2">
              Status
            </h2>
            <ul className="text-xs text-slate-400 space-y-1">
              <li>
                Symbols:{" "}
                {loadingSymbols
                  ? "loading…"
                  : symbols.length
                  ? `${symbols.length} loaded`
                  : "none"}
              </li>
              <li>
                Timeseries:{" "}
                {loadingData ? "loading…" : `${timeseries.length} points`}
              </li>
              <li>
                Predictions:{" "}
                {loadingData ? "loading…" : `${predictions.length} points`}
              </li>
            </ul>
          </div>
        </section>

        {/* Table */}
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="text-sm font-medium text-slate-200 mb-3">
            Timeseries + predictions
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs text-left">
              <thead className="border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="py-2 pr-4">Timestamp</th>
                  <th className="py-2 pr-4">Close</th>
                  <th className="py-2 pr-4">Volume</th>
                  <th className="py-2 pr-4">Pred 1d</th>
                </tr>
              </thead>
              <tbody>
                {mergedRows.length === 0 && (
                  <tr>
                    <td
                      colSpan={4}
                      className="py-3 text-slate-500 text-center"
                    >
                      {loadingData
                        ? "Loading data…"
                        : "No data available for this symbol."}
                    </td>
                  </tr>
                )}
                {mergedRows.map((row) => (
                  <tr
                    key={row.timestamp}
                    className="border-b border-slate-900/60"
                  >
                    <td className="py-1 pr-4 text-slate-300">
                      {row.timestamp}
                    </td>
                    <td className="py-1 pr-4">
                      {row.close !== undefined ? row.close : "-"}
                    </td>
                    <td className="py-1 pr-4">
                      {row.volume !== undefined ? row.volume : "-"}
                    </td>
                    <td className="py-1 pr-4">
                      {row.pred_1d !== undefined ? row.pred_1d : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}
