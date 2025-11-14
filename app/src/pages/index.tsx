import { useEffect, useState } from "react";

const API_URL = "https://redesigned-funicular-r4v4jrrxxq67fpr47-8000.app.github.dev";

type SymbolId = string;

interface Row {
  timestamp: string;
  symbol: string;
  close?: number;
  volume?: number;
  cvd?: number;
  pcr?: number;
  at_ask_bias?: number;
  pred_1d?: number;
}

export default function HomePage() {
  const [symbols, setSymbols] = useState<SymbolId[]>([]);
  const [symbol, setSymbol] = useState<SymbolId>("");
  const [rows, setRows] = useState<Row[]>([]);
  const [latest, setLatest] = useState<Row | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // ---- fetch helpers --------------------------------------------------------

  async function loadSymbols() {
    try {
      setError(null);
      const res = await fetch(`${API_URL}/symbols`);
      if (!res.ok) throw new Error(`GET /symbols -> ${res.status}`);
      const data: string[] = await res.json();
      setSymbols(data);
      if (data.length && !symbol) setSymbol(data[0]);
    } catch (e: any) {
      console.error(e);
      setError(e.message || "Failed to fetch symbols");
    }
  }

  async function loadData() {
    if (!symbol) return;
    try {
      setLoading(true);
      setError(null);

      const [tsRes, predRes, latestRes] = await Promise.all([
        fetch(`${API_URL}/timeseries?symbol=${encodeURIComponent(symbol)}`),
        fetch(`${API_URL}/predictions?symbol=${encodeURIComponent(symbol)}`),
        fetch(`${API_URL}/latest?symbol=${encodeURIComponent(symbol)}`),
      ]);

      if (!tsRes.ok) throw new Error(`GET /timeseries -> ${tsRes.status}`);
      if (!predRes.ok) throw new Error(`GET /predictions -> ${predRes.status}`);
      if (!latestRes.ok) throw new Error(`GET /latest -> ${latestRes.status}`);

      const tsData: Row[] = await tsRes.json();
      const predData: { timestamp: string; pred_1d: number }[] =
        await predRes.json();
      const latestData: any = await latestRes.json();

      // merge preds into ts rows by timestamp
      const predByTs = new Map(
        predData.map((p) => [p.timestamp, p.pred_1d] as [string, number])
      );
      const merged = tsData.map((r) => ({
        ...r,
        pred_1d: predByTs.get(r.timestamp),
      }));

      setRows(merged);
      setLatest(latestData && latestData.timestamp ? latestData : null);
    } catch (e: any) {
      console.error(e);
      setError(e.message || "Failed to fetch data");
      setRows([]);
      setLatest(null);
    } finally {
      setLoading(false);
    }
  }

  // ---- effects --------------------------------------------------------------

  useEffect(() => {
    loadSymbols();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (symbol) {
      loadData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  // ---- UI -------------------------------------------------------------------

  return (
    <main style={{ padding: "24px", fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: "28px", fontWeight: 700, marginBottom: "8px" }}>
        Orderflow Forecasting (Batch)
      </h1>
      <p style={{ marginBottom: "16px" }}>
        Select a symbol and time window, then press <b>Refresh</b>. Data comes
        from your batch pipeline.
      </p>

      {/* Controls row */}
      <div style={{ display: "flex", gap: "8px", marginBottom: "24px" }}>
        {/* Symbol dropdown */}
        <select
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          style={{ padding: "4px 8px" }}
        >
          {symbols.length === 0 && <option value="">No symbols</option>}
          {symbols.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        {/* Dummy date pickers (not wired for now) */}
        <input
          type="date"
          style={{ padding: "4px 8px" }}
          aria-label="start date"
        />
        <input
          type="date"
          style={{ padding: "4px 8px" }}
          aria-label="end date"
        />

        <button
          onClick={loadData}
          disabled={!symbol || loading}
          style={{ padding: "4px 12px" }}
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div
          style={{
            marginBottom: "16px",
            padding: "8px 12px",
            border: "1px solid #f00",
            background: "#fee",
            fontSize: "14px",
          }}
        >
          {error}
        </div>
      )}

      {/* Latest snapshot */}
      <section style={{ marginBottom: "24px" }}>
        <h2 style={{ fontSize: "20px", fontWeight: 600, marginBottom: "8px" }}>
          Latest snapshot
        </h2>
        {latest ? (
          <div style={{ fontSize: "14px" }}>
            <div>
              <b>Time:</b> {latest.timestamp}
            </div>
            {latest.close !== undefined && (
              <div>
                <b>Close:</b> {latest.close}
              </div>
            )}
            {latest.volume !== undefined && (
              <div>
                <b>Volume:</b> {latest.volume}
              </div>
            )}
            {latest.pred_1d !== undefined && (
              <div>
                <b>Pred 1d:</b> {latest.pred_1d}
              </div>
            )}
          </div>
        ) : (
          <p style={{ fontSize: "14px" }}>
            No data yet. Try Refresh after running the pipeline.
          </p>
        )}
      </section>

      {/* Table */}
      <section>
        <h3
          style={{
            fontSize: "16px",
            fontWeight: 600,
            marginBottom: "8px",
          }}
        >
          Timestamp (UTC) | Close | Volume | CVD | PCR | At-Ask Bias | Proba Up
          (1d)
        </h3>
        <table
          style={{ borderCollapse: "collapse", width: "100%", fontSize: "13px" }}
        >
          <thead>
            <tr>
              <th style={{ borderBottom: "1px solid #ccc", padding: "4px" }}>
                Timestamp
              </th>
              <th style={{ borderBottom: "1px solid #ccc", padding: "4px" }}>
                Close
              </th>
              <th style={{ borderBottom: "1px solid #ccc", padding: "4px" }}>
                Volume
              </th>
              <th style={{ borderBottom: "1px solid #ccc", padding: "4px" }}>
                CVD
              </th>
              <th style={{ borderBottom: "1px solid #ccc", padding: "4px" }}>
                PCR
              </th>
              <th style={{ borderBottom: "1px solid #ccc", padding: "4px" }}>
                At-Ask Bias
              </th>
              <th style={{ borderBottom: "1px solid #ccc", padding: "4px" }}>
                Pred 1d
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ padding: "8px" }}>
                  No rows to display.
                </td>
              </tr>
            ) : (
              rows.map((r) => (
                <tr key={r.timestamp}>
                  <td style={{ borderBottom: "1px solid #eee", padding: "4px" }}>
                    {r.timestamp}
                  </td>
                  <td style={{ borderBottom: "1px solid #eee", padding: "4px" }}>
                    {r.close ?? "-"}
                  </td>
                  <td style={{ borderBottom: "1px solid #eee", padding: "4px" }}>
                    {r.volume ?? "-"}
                  </td>
                  <td style={{ borderBottom: "1px solid #eee", padding: "4px" }}>
                    {r.cvd ?? "-"}
                  </td>
                  <td style={{ borderBottom: "1px solid #eee", padding: "4px" }}>
                    {r.pcr ?? "-"}
                  </td>
                  <td style={{ borderBottom: "1px solid #eee", padding: "4px" }}>
                    {r.at_ask_bias ?? "-"}
                  </td>
                  <td style={{ borderBottom: "1px solid #eee", padding: "4px" }}>
                    {r.pred_1d ?? "-"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </section>
    </main>
  );
}
