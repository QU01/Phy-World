import { useState, type ReactNode } from "react";
import type { ApiError } from "../lib/api";

export function Card({ title, sub, children, actions, className }: {
  title?: ReactNode; sub?: ReactNode; children: ReactNode; actions?: ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className ?? ""}`}>
      {(title || actions) && (
        <div className="row" style={{ justifyContent: "space-between", marginBottom: sub ? 6 : 8 }}>
          {title && <h3>{title}</h3>}
          {actions}
        </div>
      )}
      {sub && <p className="sub">{sub}</p>}
      {children}
    </section>
  );
}

export function Loading({ what, slow }: { what: string; slow?: string }) {
  return (
    <div className="loading" role="status">
      <div className="spinner" aria-hidden />
      <div>
        <div>{what}…</div>
        {slow && <div className="small muted">{slow}</div>}
      </div>
    </div>
  );
}

function CopyCmd({ cmd }: { cmd: string }) {
  const [done, setDone] = useState(false);
  return (
    <span className="row" style={{ gap: 6 }}>
      <code>{cmd}</code>
      <button className="btn small" style={{ padding: "1px 8px" }}
              onClick={() => navigator.clipboard?.writeText(cmd).then(() => setDone(true))}>
        {done ? "Copiado" : "Copiar"}
      </button>
    </span>
  );
}

export function ErrorNotice({ error, what }: { error: ApiError; what?: string }) {
  const missing = error.status === 503 || error.status === 0;
  return (
    <div className={`notice ${missing ? "" : "error"}`} role="alert">
      <div className="title">{missing ? "⚠" : "✕"} {missing ? `Falta algo para ${what ?? "esta vista"}` : "Error"}</div>
      <div className="secondary">{error.message}</div>
      {error.remedy && (
        <div className="small">
          Remedio: <CopyCmd cmd={error.remedy} />
        </div>
      )}
    </div>
  );
}

export function Seg<T extends string | number>({ value, options, onChange, label }: {
  value: T; options: { value: T; label: string; title?: string }[]; onChange: (v: T) => void; label?: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} aria-pressed={o.value === value} title={o.title}
                onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

export function Stat({ value, label }: { value: ReactNode; label: ReactNode }) {
  return (
    <div className="bigstat">
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}

export type LegendItem = { label: string; color: string; kind?: "line" | "dash" | "band" | "box" };

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span className="key" key={it.label}>
          {it.kind === "dash" ? (
            <svg width="14" height="4" aria-hidden><line x1="0" x2="14" y1="2" y2="2" stroke={it.color} strokeWidth="2" strokeDasharray="3 2" /></svg>
          ) : (
            <i className={`sw ${it.kind === "line" ? "line" : it.kind === "band" ? "band" : ""}`} style={{ background: it.color }} />
          )}
          {it.label}
        </span>
      ))}
    </div>
  );
}
