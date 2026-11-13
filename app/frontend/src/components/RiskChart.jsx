import { RISK } from '../lib/clinical'

const W = 640, H = 300, PAD = { l: 58, r: 14, t: 14, b: 40 }

export default function RiskChart({ hours, anchorHour, thresholds, windowH = 12, anchorLabel = 'alert', alerting = [] }) {
  const rows = hours.filter((h) => h.hour_index > anchorHour - windowH && h.hour_index <= anchorHour)
  const vals = rows.flatMap((r) => Object.values(RISK).map((k) => 100 * (r[k.key] ?? 0)))
  // a threshold is drawn for the alerting risk(s) and for any risk within half of its threshold
  const shown = Object.keys(RISK).filter((c) => alerting.includes(c) || rows.some((r) => (r[RISK[c].key] ?? 0) >= 0.5 * thresholds[c]))
  const ymax = Math.max(5, Math.ceil((Math.max(...vals, ...shown.map((c) => 100 * thresholds[c])) * 1.3) / 5) * 5)
  const x = (h) => PAD.l + ((h - (anchorHour - windowH + 1)) / (windowH - 1)) * (W - PAD.l - PAD.r)
  const y = (v) => H - PAD.b - (v / ymax) * (H - PAD.t - PAD.b)
  const yTicks = [0, ymax / 2, ymax]
  return (
    <div data-testid="risk-chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Risk curves of the three models over the last 12 hours">
        {yTicks.map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="#27324d" />
            <text x={PAD.l - 8} y={y(v) + 5} textAnchor="end" fill="#a6b2c8" fontSize="18">{v}%</text>
          </g>
        ))}
        {Object.entries(RISK).filter(([comp]) => shown.includes(comp)).map(([comp, k]) => (
          <line key={`t-${comp}`} x1={PAD.l} x2={W - PAD.r} y1={y(100 * thresholds[comp])} y2={y(100 * thresholds[comp])}
                stroke={k.color} strokeDasharray="3 7" strokeWidth="1.5" opacity="0.7" />
        ))}
        {Object.entries(RISK).map(([comp, k]) => (
          <g key={comp}>
            <polyline fill="none" stroke={k.color} strokeWidth="4" strokeLinejoin="round"
                      points={rows.map((r) => `${x(r.hour_index)},${y(100 * (r[k.key] ?? 0))}`).join(' ')} />
            {rows.length > 0 && <circle cx={x(rows[rows.length - 1].hour_index)} cy={y(100 * (rows[rows.length - 1][k.key] ?? 0))} r="6" fill={k.color} />}
          </g>
        ))}
        {rows.map((r) => (
          <text key={r.hour_index} x={x(r.hour_index)} y={H - 8} textAnchor="middle" fill="#a6b2c8" fontSize="17">
            {r.hour_index === anchorHour ? (anchorLabel === 'alert' ? 'alert' : 'now') : `${r.hour_index - anchorHour} h`}
          </text>
        ))}
      </svg>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-base">
        {Object.entries(RISK).map(([comp, k]) => (
          <span key={comp} className="flex items-center gap-2">
            <span className="inline-block h-1 w-6 rounded" style={{ background: k.color }} />
            {k.label} {rows.length ? `${(100 * (rows[rows.length - 1][k.key] ?? 0)).toFixed(0)}%` : ''}
          </span>
        ))}
        <span className="text-sm text-muted">dotted = alert threshold</span>
      </div>
    </div>
  )
}
