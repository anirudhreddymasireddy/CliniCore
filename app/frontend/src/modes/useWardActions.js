import { useCallback, useRef, useState } from 'react'
import { postFeedback } from '../lib/api'

// Shared nurse-action handling: log locally via the backend, show a confirmation, optionally return to the ward.
export function useWardActions({ onReturnToWard, onAcknowledged }) {
  const [toast, setToast] = useState(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const timer = useRef(null)
  const show = useCallback((text, ms = 2200) => {
    clearTimeout(timer.current)
    setToast(text)
    timer.current = setTimeout(() => setToast(null), ms)
  }, [])
  const act = useCallback(async (action, ctx) => {
    try {
      await postFeedback({ ...ctx, action, reason: ctx.reason || '' })
    } catch {
      show('Could not log the action – please retry', 3000)
      return
    }
    setRefreshKey((k) => k + 1)
    if (action === 'call doctor') {
      show('Doctor notified – logged', 1800)
      onAcknowledged?.(ctx)
      setTimeout(() => onReturnToWard?.(), 1700)
    } else if (action === 'acknowledge') {
      show('Acknowledged – logged')
      onAcknowledged?.(ctx)
    } else if (action === 'false alarm') {
      show(`False alarm logged – ${ctx.reason}`)
      onAcknowledged?.(ctx)
    }
  }, [onAcknowledged, onReturnToWard, show])
  const reset = useCallback(() => { clearTimeout(timer.current); setToast(null) }, [])
  return { toast, show, act, refreshKey, reset }
}

export function latestVitals(points, untilMin, hourly) {
  const out = { hr: null, map: null, rr: null, spo2: null }
  for (let i = points.length - 1; i >= 0 && Object.values(out).some((v) => v == null); i--) {
    const p = points[i]
    if (p.t > untilMin) continue
    if (p.t < untilMin - 120) break
    for (const k of Object.keys(out)) if (out[k] == null && p[k] != null) out[k] = p[k]
  }
  if (hourly) {
    out.hr ??= hourly.heart_rate; out.map ??= hourly.map_mmhg; out.rr ??= hourly.resp_rate; out.spo2 ??= hourly.spo2
  }
  return out
}

export function riskLine(h) {
  if (!h) return null
  return `Shock ${(100 * h.p_shock).toFixed(0)}% · Resp ${(100 * h.p_resp).toFixed(0)}% · Renal ${(100 * h.p_renal).toFixed(0)}%`
}
