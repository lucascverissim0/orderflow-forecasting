import { useEffect, useMemo, useState } from "react";

/**
 * Minimal dashboard page (no extra libs).
 * - Fetches symbols from your FastAPI backend
 * - Lets you pick a symbol and time window
 * - Fetches /timeseries and shows key fields in a table
 * - Highlights the latest model score & signal
 *
 * Backend default: http://localhost:8000 (override with NEXT_PUBLIC_API_URL)
 */

type Row = {
  timestamp: string;
  symbol?: string;
  close?: number;
  volume?: number;
  cvd_proxy?: number;
  pcr?: number;
  at_ask_bias?: number;
  proba_up_1d?: number;
  signal_1d?: number;
};

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState<string>("");
  const [start, setStart] = useState<string>("");
  const [end, setEnd] = useState<string>("");
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  // Load symbols on mount
  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch(`${API}/symbols`);
        const j = await r.json();
        const syms: string[] = j.symbols || [];
        setSymbols(syms);
        if (syms.length && !symbol) setSymbol(syms[0]);
      } catch (e: any) {
        setError(e?.message ?? "Failed to load symbols");
      }
    };
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchData = async () => {
    if (!symbol) return;
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      params.set("symbol", symbol);
      if (start) params.set("start", start);
      if (end) params.set("end", end);
      params.set("limit", "2000");
      const r = await fetch(`${API}/timeseries?` + params.toString());
      const j = await r.json();
      setRows(j.rows || []);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load timeseries");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Auto-load once defaults are ready
    if (symbol) fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  const latest = useMemo(() => (rows.length ? rows[rows.length - 1] : undefined), [rows]);

  return (
    <main className="min-h-screen p-6" style={{ fontFamily: "ui-sans-serif, system-ui" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        <h1 className="text-2xl font-bold mb-2">Orderflow Forecasting (Batch)</h1>
        <p className="text-sm text-gray-600 mb-6">
          Select a symbol and time window, then press <b>Refresh</b>. Data comes from your batch pipeline.
        </p>

        {/* Controls */}
        <div
          className="grid gap-3 mb-6"
          style={{ gridTemplateColumns: "220px 180px 180px 140px" }}
        >
          <select
            className="border rounded px-3 py-2"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          >
            {symbols.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>

          <input
            className="border rounded px-3 py-2"
            type="date"
            value={start}
            onChange={(e) => setStart(e.target.value)}
            placeholder="Start (YYYY-MM-DD)"
          />
          <input
            className="border rounded px-3 py-2"
            type="date"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
            placeholder="End (YYYY-MM-DD)"
          />

          <button
            className="border rounded px-4 py-2 bg-black text-white disabled:opacity-50"
            onClick={fetchData}
            disabled={loading || !symbol}
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>

        {/* Latest snapshot */}
        <div className="mb-6 border rounded p-4">
          <h2 className="font-semibold mb-2">Latest snapshot</h2>
          {latest ? (
            <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(4, minmax(0,1fr))" }}>
              <Stat label="Timestamp" value={latest.timestamp} />
              <Stat label="Close" value={fmt(latest.close)} />
              <Stat label="Proba Up (1d)" value={fmt(latest.proba_up_1d)} />
              <Stat
                label="Signal (1d)"
                value={signalLabel(latest.signal_1d)}
                emphasis={latest.signal_1d !== 0}
              />
            </div>
          ) : (
            <p className="text-gray-600 text-sm">No data yet. Try Refresh after running the pipeline.</p>
          )}
        </div>

        {/* Table */}
        <div className="border rounded overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-gray-50">
                <Th>Timestamp (UTC)</Th>
                <Th>Close</Th>
                <Th>Volume</Th>
                <Th>CVD</Th>
                <Th>PCR</Th>
                <Th>At-Ask Bias</Th>
                <Th>Proba Up (1d)</Th>
                <Th>Signal</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => (
                <tr key={idx} className={idx % 2 ? "bg-white" : "bg-gray-50/30"}>
                  <Td>{r.timestamp}</Td>
                  <Td>{fmt(r.close)}</Td>
                  <Td>{fmt(r.volume)}</Td>
                  <Td>{fmt(r.cvd_proxy)}</Td>
                  <Td>{fmt(r.pcr)}</Td>
                  <Td>{fmt(r.at_ask_bias)}</Td>
                  <Td>{fmt(r.proba_up_1d)}</Td>
                  <Td>{signalLabel(r.signal_1d)}</Td>
                </tr>
              ))}
              {!rows.length && (
                <tr>
                  <td className="p-3 text-gray-600" colSpan={8}>
                    No rows to display.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {error && (
          <p className="mt-4 text-red-600 text-sm">
            {error}
          </p>
        )}

        <p className="mt-6 text-xs text-gray-500">
          Tip: Set <code>NEXT_PUBLIC_API_URL</code> in <code>app/.env.local</code> to point to your FastAPI server.
        </p>
      </div>
    </main>
  );
}

function Th({ children }: { children: any }) {
  return <th className="text-left p-3 font-medium">{children}</th>;
}
function Td({ children }: { children: any }) {
  return <td className="p-3 whitespace-nowrap">{children ?? "—"}</td>;
}
function Stat({ label, value, emphasis = false }: { label: string; value?: string; emphasis?: boolean }) {
  return (
    <div className="border rounded p-3">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-base ${emphasis ? "font-semibold" : ""}`}>{value ?? "—"}</div>
    </div>
  );
}

function fmt(x?: number) {
  if (x === null || x === undefined) return undefined;
  if (Math.abs(x) >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return x.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function signalLabel(s?: number) {
  if (s === 1) return "Long";
  if (s === -1) return "Short";
  if (s === 0) return "Flat";
  return "—";
}
