import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

export type Async<T> = { data?: T; error?: ApiError; loading: boolean };

/** Ejecuta ``fn`` cuando cambian ``deps``; descarta respuestas de peticiones viejas. ``null`` = no pedir. */
export function useApi<T>(fn: ((signal: AbortSignal) => Promise<T>) | null, deps: unknown[]): Async<T> {
  const [state, setState] = useState<Async<T>>({ loading: fn !== null });
  const seq = useRef(0);
  useEffect(() => {
    if (!fn) {
      setState({ loading: false });
      return;
    }
    const id = ++seq.current;
    const ctl = new AbortController();
    setState((s) => ({ data: s.data, loading: true }));
    fn(ctl.signal)
      .then((data) => id === seq.current && setState({ data, loading: false }))
      .catch((e: unknown) => {
        if ((e as Error).name === "AbortError" || id !== seq.current) return;
        const err = e instanceof ApiError ? e : new ApiError(0, String((e as Error).message ?? e));
        setState({ error: err, loading: false });
      });
    return () => ctl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
