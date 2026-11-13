// Bedside-monitor style trends from raw monitor values: black background, standard colours, thin lines,
// large current numerics on the right, time axis in hours before the alert (or before now).
// span = minimum visible range, so real trends show without magnifying noise
const STRIPS = [
  { key: 'hr', label: 'HR', unit: 'bpm', color: 'var(--color-mon-hr)', span: 40, floor: 0, ceil: 300 },
  { key: 'map', label: 'MAP', unit: 'mmHg', color: 'var(--color-mon-bp)', span: 35, floor: 0, ceil: 250 },
  { key: 'spo2', label: 'SpO₂', unit: '%', color: 'var(--color-mon-spo2)', span: 12, floor: 50, ceil: 100 },
  { key: 'rr', label: 'RR', unit: '/min', color: 'var(--color-mon-rr)', span: 20, floor: 0, ceil: 60 },
]

function change6h(pts, key, anchorMin) {
  const avg = (a, b) => {
    const v = pts.filter((p) => p.t > a && p.t <= b && p[key] != null).map((p) => p[key])
    return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null
  }
  const start = avg(anchorMin - 360, anchorMin - 300), end = avg(anchorMin - 60, anchorMin)
  return start == null || end == null ? null : { start, end, diff: end - start }
}
const W = 1000, H = 100, GAP_MIN = 75

function segments(pts, key, x, y) {
  const segs = []
  let cur = [], lastT = null
  for (const p of pts) {
    if (p[key] == null) continue
    if (lastT != null && p.t - lastT > GAP_MIN && cur.length) { segs.push(cur); cur = [] }
    cur.push(`${x(p.t).toFixed(1)},${y(p[key]).toFixed(1)}`)
    lastT = p.t
  }
  if (cur.length) segs.push(cur)
  return segs
}

export default function MonitorVitals({ points, anchorMin, alerts, windowH = 12, anchorLabel = 'alert', resolution, artifactTimes = [] }) {
  const lo = anchorMin - windowH * 60
  const pts = points.filter((p) => p.t >= lo && p.t <= anchorMin)
  const x = (t) => ((t - lo) / (anchorMin - lo)) * W
  const ticks = Array.from({ length: windowH / 2 + 1 }, (_, i) => -windowH + i * 2)
  const marks = alerts.filter((a) => a.t >= lo && a.t <= anchorMin)
  const bpSource = [...pts].reverse().find((p) => p.map != null)?.bp_source
  return (
    <div className="rounded-xl bg-black p-3 ring-1 ring-white/10" data-testid="monitor">
      {STRIPS.map((s) => {
        const vals = pts.map((p) => p[s.key]).filter((v) => v != null)
        const dmin = vals.length ? Math.min(...vals) : s.floor, dmax = vals.length ? Math.max(...vals) : s.floor + s.span
        const mid = (dmin + dmax) / 2, half = Math.max(s.span, dmax - dmin) / 2 * 1.1
        const vmin = Math.max(s.floor, mid - half), vmax = Math.min(s.ceil, mid + half)
        const ch = change6h(pts, s.key, anchorMin)
        const y = (v) => H - 8 - ((v - vmin) / (vmax - vmin || 1)) * (H - 16)
        const current = vals.length ? vals[vals.length - 1] : null
        return (
          <div key={s.key} className="flex items-stretch gap-3 border-b border-white/10 py-1 last:border-0">
            <div className="relative flex-1">
              <span className="absolute left-1 top-0 text-xs font-bold" style={{ color: s.color }}>
                {s.label}{s.key === 'map' && bpSource ? ` (${bpSource})` : ''}
              </span>
              <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-[84px] w-full">
                {marks.map((m) => (
                  <line key={m.t} x1={x(m.t)} x2={x(m.t)} y1={0} y2={H} stroke={m.color} strokeWidth="1.5"
                        strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
                ))}
                {segments(pts, s.key, x, y).map((seg, i) => (
                  <polyline key={i} points={seg.join(' ')} fill="none" stroke={s.color} strokeWidth="1.6"
                            vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
                ))}
              </svg>
            </div>
            <div className="flex w-36 flex-col items-end justify-center" data-testid={`numeric-${s.key}`}>
              <span className="text-xs font-bold" style={{ color: s.color }}>{s.label}</span>
              <span className="font-black tabular-nums leading-none" style={{ color: s.color, fontSize: '2.6rem' }}>
                {current == null ? '–' : Math.round(current)}
              </span>
              <span className="text-xs" style={{ color: s.color }}>{s.unit}</span>
              {ch && Math.abs(ch.diff) >= 1 && (
                <span className="mt-0.5 text-xs font-semibold text-white/75" data-testid={`change-${s.key}`}>
                  {Math.round(ch.start)} → {Math.round(ch.end)} in 6 h
                </span>
              )}
            </div>
          </div>
        )
      })}
      <div className="relative mr-[9.75rem] mt-1 h-6 text-xs text-white/60">
        {ticks.map((t) => (
          <span key={t} className="absolute -translate-x-1/2" style={{ left: `${((t + windowH) / windowH) * 100}%` }}>
            {t === 0 ? (anchorLabel === 'alert' ? 'alert' : 'now') : `${t} h`}
          </span>
        ))}
      </div>
      <div className="mt-1 flex flex-wrap justify-between gap-2 text-xs text-white/55">
        <span>Hours before {anchorLabel === 'alert' ? 'the alert' : 'now'} · dashed line = alert · {resolution}</span>
        <span>{(() => { const n = artifactTimes.filter((t) => t >= lo && t <= anchorMin).length
          return n ? `${n} implausible values hidden in this window (likely artifacts)` : 'no implausible values in this window' })()}</span>
      </div>
    </div>
  )
}
