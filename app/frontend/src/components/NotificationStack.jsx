import { TIER, HORIZON, riskTypes, topReasons } from '../lib/clinical'

// Fixed order, readable in 3 s: tier + horizon, bed + time, risk type, top 3 reasons. No close / minimise.
function Card({ n, onOpen, onAck, onSnooze }) {
  const t = TIER[n.level]
  const reasons = topReasons(n.text)
  return (
    <article className="anim-slide-in overflow-hidden rounded-2xl border-2 bg-panel-2 shadow-2xl shadow-black/40"
             style={{ borderColor: t.color }} data-testid="notification" data-level={n.level} data-bed={n.bed}>
      <header className="px-4 py-2.5 text-base font-black" style={{ background: t.color, color: '#0b0f19' }}>
        <span aria-hidden>{t.icon}</span> {n.level} · {HORIZON}
      </header>
      <div className="space-y-2 px-4 py-3">
        <div className="text-xl font-extrabold">Bed {n.bed} · {n.anonId} · <span className="font-semibold text-muted">{n.since}</span></div>
        <div className="text-lg font-bold" style={{ color: t.color }}>Risk: {riskTypes(n.components).join(', ')}</div>
        <ul className="list-disc space-y-0.5 pl-6 text-base leading-snug">
          {(reasons.length ? reasons : ['pattern-based risk (no single value out of range)']).map((r) => <li key={r}>{r}</li>)}
        </ul>
        <div className="flex gap-2 pt-1">
          <button onClick={onOpen} className="min-h-12 flex-1 rounded-xl bg-blue-500 text-base font-bold text-white hover:bg-blue-400">Open</button>
          <button onClick={onAck} className="min-h-12 flex-[1.4] rounded-xl border-2 border-line bg-panel text-base font-bold hover:bg-line">Acknowledge</button>
          <button onClick={onSnooze} className="min-h-12 flex-[1.4] rounded-xl border-2 border-line bg-panel text-base font-bold hover:bg-line">Snooze 30 min</button>
        </div>
      </div>
    </article>
  )
}

export default function NotificationStack({ items, onOpen, onAck, onSnooze }) {
  return (
    <aside className="sticky top-4 flex max-h-[calc(100vh-2rem)] w-[min(420px,32vw)] min-w-[340px] shrink-0 flex-col gap-3 overflow-y-auto"
           aria-label="Notifications" data-testid="notification-stack">
      <div className="flex items-baseline justify-between">
        <h2 className="text-xl font-extrabold">Notifications</h2>
        <span className="text-sm text-muted">{items.length ? `${items.length} active` : 'none active'}</span>
      </div>
      {items.map((n) => (
        <Card key={n.id} n={n} onOpen={() => onOpen(n)} onAck={() => onAck(n)} onSnooze={() => onSnooze(n)} />
      ))}
      {!items.length && <p className="rounded-2xl border border-line bg-panel p-4 text-base text-muted">No active MEDIUM or CRITICAL predictions.</p>}
      <p className="text-sm text-muted">Only MEDIUM and CRITICAL predictions notify. NORMAL never notifies.</p>
    </aside>
  )
}
