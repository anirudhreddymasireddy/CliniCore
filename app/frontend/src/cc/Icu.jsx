import { memo, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { api, num } from './api'
import { Counter, LEVEL, Loading, Shell, Toast, inputCls, useToast } from './ui'
import { STATUS, Sbar, URG } from './notify'

const LEVELS = ['Critical', 'High', 'Watch', 'Low']
const TREND = { rising: ['↑', 'text-red-300', 'rising'], falling: ['↓', 'text-emerald-300', 'falling'], steady: ['→', 'text-muted', 'steady'], new: ['•', 'text-muted', 'new'] }
const DOMAIN_NAME = { sepsis: 'Sepsis', arf: 'Respiratory failure', aki: 'Kidney injury', shock: 'Circulatory shock' }
const DOMAIN_ICON = { sepsis: '🦠', arf: '🫁', aki: '💧', shock: '🫀' }
const REASON_ICON = { sensor_or_measurement_error: '📟', expected_for_this_patient: '🧍', already_being_treated: '💊', treated_early_event_prevented: '✅', wrong_risk_type: '🔀', other: '✏️' }

function Gauge({ value, level, size = 96 }) {
  const c = LEVEL[level]?.color || '#a6b2c8', r = 40, L = 2 * Math.PI * r, v = Math.max(0, Math.min(100, value || 0))
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" aria-label={`criticality ${num(v)} of 100`}>
      <circle cx="50" cy="50" r={r} fill="none" stroke="#27324d" strokeWidth="9" />
      <circle cx="50" cy="50" r={r} fill="none" stroke={c} strokeWidth="9" strokeLinecap="round" strokeDasharray={`${(L * v) / 100} ${L}`} transform="rotate(-90 50 50)" style={{ transition: 'stroke-dasharray 600ms ease' }} />
      <text x="50" y="52" textAnchor="middle" fontSize="28" fontWeight="900" fill="#eef2fa">{num(v)}</text>
      <text x="50" y="68" textAnchor="middle" fontSize="10" fill="#a6b2c8">of 100</text>
    </svg>
  )
}

function Timeline({ points, height = 170 }) {
  const ref = useRef(null)
  const box = useRef(null)
  const [hover, setHover] = useState(null)
  const [W, setW] = useState(700)
  useEffect(() => {
    if (!box.current) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(300, Math.round(e.contentRect.width))))
    ro.observe(box.current); return () => ro.disconnect()
  }, [points])
  if (!points?.length) return <p className="text-sm text-muted">No changes recorded during this stay.</p>
  if (points[0].hour_idx > 0) points = [{ hour_idx: 0, criticality: 0, top_domain: '', top_tier: 0 }, ...points]
  const H = height, P = { l: 30, r: 10, t: 10, b: 22 }
  const maxX = Math.max(...points.map((p) => p.hour_idx), 1)
  const x = (h) => P.l + (W - P.l - P.r) * (h / maxX), y = (v) => P.t + (H - P.t - P.b) * (1 - v / 100)
  let d = ''
  points.forEach((p, i) => { d += i === 0 ? `M${x(p.hour_idx)},${y(p.criticality)}` : `H${x(p.hour_idx)}V${y(p.criticality)}` })
  d += `H${x(maxX)}`
  const bands = [[70, 100, LEVEL.Critical.color], [40, 70, LEVEL.High.color], [15, 40, LEVEL.Watch.color], [0, 15, LEVEL.Low.color]]
  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect(), hx = ((e.clientX - r.left) / r.width) * W
    const h = ((hx - P.l) / (W - P.l - P.r)) * maxX
    const p = [...points].reverse().find((q) => q.hour_idx <= h) || points[0]
    setHover({ x: Math.max(P.l, Math.min(W - P.r, hx)), p, h: Math.max(0, Math.round(h)) })
  }
  const tier = ['Low', 'Watch', 'High', 'Critical']
  return (
    <div className="relative" ref={box}>
      {points.length === 1 && <p className="mb-1 text-xs text-muted">Level unchanged since admission to the ICU.</p>}
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block cursor-crosshair" onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label="Criticality during the stay">
        {bands.map(([lo, hi, c]) => <rect key={lo} x={P.l} width={W - P.l - P.r} y={y(hi)} height={y(lo) - y(hi)} fill={c} opacity="0.07" />)}
        {[15, 40, 70].map((t) => <g key={t}><line x1={P.l} x2={W - P.r} y1={y(t)} y2={y(t)} stroke="#27324d" strokeDasharray="3 4" /><text x={P.l - 6} y={y(t) + 3} textAnchor="end" fill="#a6b2c8" fontSize="10">{t}</text></g>)}
        <path d={`${d}V${y(0)}H${x(points[0].hour_idx)}Z`} fill="url(#g)" opacity="0.35" />
        <defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#38bdf8" /><stop offset="1" stopColor="#38bdf8" stopOpacity="0" /></linearGradient></defs>
        <path d={d} fill="none" stroke="#38bdf8" strokeWidth="2.5" />
        <text x={W - P.r} y={H - 5} textAnchor="end" fill="#a6b2c8" fontSize="10">hours in ICU →</text>
        <text x={P.l} y={H - 5} fill="#a6b2c8" fontSize="10">0</text>
        {hover && <line x1={hover.x} x2={hover.x} y1={P.t} y2={H - P.b} stroke="#eef2fa" strokeOpacity="0.4" />}
      </svg>
      {hover && (
        <div className="pointer-events-none absolute top-1 rounded-md border border-line bg-bg/95 px-2 py-1 text-xs shadow-lg" style={{ left: `${(hover.x / W) * 100}%`, transform: hover.x > W / 2 ? 'translateX(-105%)' : 'translateX(5%)' }}>
          <b>hour {hover.h}</b> · score {num(hover.p.criticality)}<br />
          <span className="text-muted">{hover.p.top_domain ? `${DOMAIN_NAME[hover.p.top_domain] || hover.p.top_domain} · ${tier[hover.p.top_tier] || ''}` : 'no active risk'}</span>
        </div>
      )}
    </div>
  )
}

function FalseAlarmForm({ reasons, onSubmit, onCancel }) {
  const [reason, setReason] = useState('')
  const [note, setNote] = useState('')
  const needNote = reason === 'other'
  const ok = reason && (!needNote || note.trim())
  return (
    <div className="mt-3 rounded-lg border border-red-400/40 bg-red-500/5 p-3 anim-fade-up">
      <div className="mb-2 text-sm font-bold">Why is this a false alarm? <span className="font-normal text-muted">(required)</span></div>
      <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
        {reasons.map((r) => (
          <button key={r.code} onClick={() => setReason(r.code)}
                  className={`flex items-start gap-2 rounded-md border px-2.5 py-2 text-left text-sm transition ${reason === r.code ? 'border-red-400 bg-red-500/15' : 'border-line hover:bg-panel-2'}`}>
            <span aria-hidden>{REASON_ICON[r.code] || '•'}</span><span>{r.label}</span>
          </button>
        ))}
      </div>
      <textarea rows="2" maxLength="300" className={`${inputCls} mt-2 text-sm`} value={note} onChange={(e) => setNote(e.target.value)}
                placeholder={needNote ? 'Describe what it actually was (required)' : 'What did you see? (optional, saved in the log)'} />
      {reason === 'treated_early_event_prevented' && <p className="mt-1 text-xs text-amber-200">Logged and counted, but not used to teach the model "no event" – the warning was right.</p>}
      <div className="mt-2 flex gap-2">
        <button disabled={!ok} onClick={() => onSubmit(reason, note)} className="rounded-md bg-red-500 px-3 py-1.5 text-sm font-bold text-white transition hover:bg-red-600 disabled:opacity-40">Log false alarm</button>
        <button onClick={onCancel} className="rounded-md px-3 py-1.5 text-sm text-muted hover:bg-panel-2">Cancel</button>
      </div>
    </div>
  )
}

function RiskCard({ r, reasons, done, onFalse }) {
  const [open, setOpen] = useState(false)
  const l = LEVEL[r.tier] || {}
  return (
    <div className="rounded-lg border border-line bg-panel-2/60 p-3" style={{ borderLeft: `4px solid ${l.color}` }}>
      <div className="flex items-start gap-3">
        <span className="text-xl" aria-hidden>{DOMAIN_ICON[r.domain]}</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <b>{r.name}</b>
            <span className="rounded px-1.5 py-0.5 text-xs font-extrabold" style={{ background: l.color, color: '#0b0f19' }}>{r.tier}</span>
            {r.early_warning && <span className="rounded bg-sky-500/20 px-1.5 py-0.5 text-xs text-sky-200">early warning</span>}
          </div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {r.evidence.replace(/\s*\(model expects.*\)$/, '').split(', ').map((e) => <span key={e} className="rounded bg-bg px-2 py-0.5 text-xs">{e}</span>)}
          </div>
          {r.early_warning && <p className="mt-1 text-xs text-sky-200">{r.evidence.match(/\(model expects.*\)$/)?.[0]?.slice(1, -1)}</p>}
        </div>
        {done ? <span className="shrink-0 rounded-md bg-red-500/15 px-2 py-1 text-xs text-red-200">✓ false alarm logged</span>
          : !open && <button onClick={() => setOpen(true)} className="shrink-0 rounded-md border border-red-400/60 px-2.5 py-1 text-xs font-bold text-red-200 transition hover:bg-red-500/20">False alarm</button>}
      </div>
      {open && !done && <FalseAlarmForm reasons={reasons} onCancel={() => setOpen(false)} onSubmit={async (reason, note) => { if (await onFalse(r, reason, note)) setOpen(false) }} />}
    </div>
  )
}



function NotifyPanel({ bed, meta, role, onClose, onSent }) {
  const SPEC = { sepsis: 'infectious', arf: 'pulmonology', aki: 'nephrology', shock: 'cardiology' }
  const [docs, setDocs] = useState(() => ['icu_attending', ...(bed.risks[0] && SPEC[bed.risks[0].domain] ? [SPEC[bed.risks[0].domain]] : [])])
  const [pErr, setPErr] = useState('')
  const [urg, setUrg] = useState(bed.level === 'Critical' ? 'emergency' : bed.level === 'High' ? 'urgent' : 'routine')
  const [risks, setRisks] = useState(bed.risks.map((r) => r.domain))
  const [note, setNote] = useState('')
  const [preview, setPreview] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  useEffect(() => {
    const t = setTimeout(() => api(`/api/notify/preview?stay_id=${bed.stay_id}&urgency=${urg}&risks=${risks.join(',')}&note=${encodeURIComponent(note)}`)
      .then((p) => { setPreview(p.message); setPErr('') }).catch((e) => setPErr(e.message)), 200)
    return () => clearTimeout(t)
  }, [bed.stay_id, urg, risks, note])
  useEffect(() => { const k = (e) => e.key === 'Escape' && onClose(); window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k) }, [onClose])
  const toggle = (arr, set, v) => set(arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v])
  const send = async () => {
    setBusy(true); setErr('')
    try { const r = await api('/api/notify', { stay_id: bed.stay_id, doctors: docs, urgency: urg, risks, note, role }); onSent(r) } catch (e) { setErr(e.message) }
    setBusy(false)
  }
  return (
    <div className="fixed inset-0 z-40 grid place-items-center bg-black/60 p-4 backdrop-blur-sm" onClick={onClose}>
      <div className="anim-fade-up max-h-[92vh] w-full max-w-3xl overflow-auto rounded-2xl border border-line bg-panel p-5 shadow-2xl" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Notify doctor">
        <div className="mb-4 flex items-center gap-3"><span className="text-2xl">🔔</span><div><h2 className="text-lg font-black">Notify doctor</h2><p className="text-xs text-muted">Bed {bed.bed} · #{bed.stay_id} · {bed.level} {num(bed.criticality)}/100</p></div>
          <button onClick={onClose} className="ml-auto rounded-md px-2 py-1 text-muted hover:bg-panel-2" aria-label="close">✕</button></div>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="grid content-start gap-4">
            <div><div className="mb-1.5 text-sm font-bold">Send to <span className="font-normal text-muted">(suggested from the top risk)</span></div>
              <div className="grid gap-1.5">{meta.doctors.map((d) => (
                <label key={d.id} className={`flex cursor-pointer items-center gap-2 rounded-md border px-2.5 py-1.5 text-sm ${docs.includes(d.id) ? 'border-sky-400 bg-sky-500/10' : 'border-line'}`}>
                  <input type="checkbox" className="h-4 w-4" checked={docs.includes(d.id)} onChange={() => toggle(docs, setDocs, d.id)} />{d.label}
                </label>))}</div></div>
            <div><div className="mb-1.5 text-sm font-bold">Urgency</div>
              <div className="grid grid-cols-3 gap-1.5">{meta.urgency.map((u) => (
                <button key={u.id} onClick={() => setUrg(u.id)} className={`rounded-md border px-2 py-1.5 text-xs font-bold ${urg === u.id ? `${URG[u.id].cls} border-transparent` : 'border-line hover:bg-panel-2'}`}>{URG[u.id].label}<div className="font-normal opacity-80">{u.label.replace(/\(.*\)/, '')}</div></button>))}</div></div>
            <div><div className="mb-1.5 text-sm font-bold">Risks to include</div>
              <div className="flex flex-wrap gap-1.5">{bed.risks.map((r) => (
                <button key={r.domain} onClick={() => toggle(risks, setRisks, r.domain)} className={`rounded-full border px-2.5 py-1 text-xs ${risks.includes(r.domain) ? 'border-sky-400 bg-sky-500/15' : 'border-line text-muted line-through'}`}>{DOMAIN_ICON[r.domain]} {r.name} · {r.tier}</button>))}</div></div>
            <textarea rows="2" maxLength="300" className={`${inputCls} text-sm`} placeholder="Add a note for the doctor (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
          </div>
          <div><div className="mb-1.5 text-sm font-bold">Message preview <span className="font-normal text-muted">(SBAR, measured values only)</span></div>{preview ? <Sbar text={preview} /> : <div className="rounded-lg bg-bg p-3 text-xs text-muted">{pErr ? `Preview not available: ${pErr}` : 'Loading preview…'}</div>}</div>
        </div>
        {err && <p className="mt-3 rounded-md bg-red-500/15 p-2 text-sm text-red-200">{err}</p>}
        <div className="mt-4 flex items-center gap-2">
          <button disabled={!docs.length || busy} onClick={send} className="rounded-lg bg-sky-500 px-4 py-2 text-sm font-bold text-white hover:bg-sky-600 disabled:opacity-40">{busy ? 'Sending…' : `Send to ${docs.length} doctor${docs.length === 1 ? '' : 's'}`}</button>
          <button onClick={onClose} className="rounded-lg px-3 py-2 text-sm text-muted hover:bg-panel-2">Cancel</button>
          <span className="ml-auto text-xs text-muted">Shows as a pop-up on the doctor's screen and in the Doctor inbox</span>
        </div>
      </div>
    </div>
  )
}

function AutoSwitch({ value, onChange }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-muted" title="Pops up an emergency alert to the ICU attending for every Critical patient whose score is rising or new, once per patient until a doctor responds.">
      <button role="switch" aria-checked={value} aria-label="Auto-notify doctor for rising Critical" onClick={() => onChange(!value)}
              className={`relative h-5 w-9 rounded-full transition ${value ? 'bg-sky-500' : 'bg-line'}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${value ? 'left-4.5' : 'left-0.5'}`} />
      </button>
      Auto-alert
    </label>
  )
}

// ---------- live vitals (demo: small beat-to-beat movement around the patient's real recorded values; a hospital feed replaces this)
const VIT = {
  hr: { label: 'Heart rate', unit: '/min', jit: 2, d: 0, bad: (v) => v > 110 || v < 50, color: '#4ade80' },
  spo2: { label: 'SpO₂', unit: '%', jit: 0.7, d: 0, max: 100, bad: (v) => v < 92, color: '#38bdf8' },
  sbp: { label: 'Blood pressure', unit: 'mmHg', jit: 2.5, d: 0, bad: (v) => v < 90 || v > 180, color: '#f87171' },
  rr: { label: 'Breathing rate', unit: '/min', jit: 1, d: 0, bad: (v) => v > 24 || v < 8, color: '#facc15' },
  temp_c: { label: 'Temperature', unit: '°C', jit: 0.04, d: 1, bad: (v) => v >= 38.3 || v < 36, color: '#fb923c' },
}
// one shared clock; only the small components that show live numbers subscribe, so the page itself never re-renders for it
const clock = { t: 0, subs: new Set() }
setInterval(() => { clock.t += 0.5; clock.subs.forEach((f) => f()) }, 500)
function useClock() {
  return useSyncExternalStore((f) => { clock.subs.add(f); return () => clock.subs.delete(f) }, () => clock.t)
}
function wobble(seed, t) {                                   // deterministic smooth noise in [-1, 1]
  return Math.sin(t * 0.9 + seed * 1.7) * 0.6 + Math.sin(t * 2.3 + seed * 0.37) * 0.4
}
function liveVal(base, key, seed, t) {
  if (base == null || Number.isNaN(base)) return null
  const v = VIT[key], x = base + wobble(seed + key.length * 13, t) * v.jit
  return v.max ? Math.min(v.max, x) : x
}
const fmt = (key, v) => (v == null ? '–' : Number(v).toFixed(VIT[key]?.d ?? 0))

function TileVitals({ v, seed }) {
  const t = Math.floor(useClock())                       // tiles: once per second
  const hr = liveVal(v.hr, 'hr', seed, t), sp = liveVal(v.spo2, 'spo2', seed, t), sbp = liveVal(v.sbp, 'sbp', seed, t), dbp = liveVal(v.dbp, 'sbp', seed + 7, t)
  const cell = (k, val, txt) => <span className={VIT[k].bad(val ?? NaN) ? 'font-bold text-red-300' : ''}>{txt}</span>
  return (
    <div className="mt-1 grid grid-cols-3 gap-1 border-t border-white/10 pt-1 font-mono text-[11px] tabular-nums">
      <span><span className="text-muted">HR </span>{cell('hr', hr, fmt('hr', hr))}</span>
      <span><span className="text-muted">SpO₂ </span>{cell('spo2', sp, fmt('spo2', sp))}</span>
      <span className="text-right">{cell('sbp', sbp, sbp == null ? '–' : `${fmt('sbp', sbp)}/${fmt('sbp', dbp)}`)}</span>
    </div>
  )
}

const BedTile = memo(function BedTile({ b, active, onOpen }) {
  const l = LEVEL[b.level] || LEVEL.Low
  return (
    <button onClick={() => onOpen(b.stay_id)} title={`patient #${b.stay_id}`}
            className={`rounded-lg border-2 p-2 text-left transition-colors hover:bg-white/5 ${active ? 'ring-2 ring-sky-400' : ''} ${b.level === 'Critical' ? 'tile-pulse' : ''}`}
            style={{ borderColor: `${l.color}${b.level === 'Low' ? '55' : 'cc'}`, background: `${l.color}${b.level === 'Low' ? '0d' : '1c'}` }}>
      <div className="flex items-center justify-between text-xs"><b>Bed {b.bed}</b><span className="flex items-center gap-1">{b.notified && <span>🔔</span>}<span style={{ color: l.color }}>{l.icon}</span></span></div>
      <div className="flex items-end justify-between">
        <span className="text-3xl font-black leading-tight" style={{ color: b.level === 'Low' ? '#a6b2c8' : l.color }}>{num(b.criticality)}</span>
        <span className="truncate pb-1 text-[11px] text-muted">{b.risks[0] ? `${DOMAIN_ICON[b.risks[0].domain] || ''} ${DOMAIN_NAME[b.risks[0].domain] || ''}` : 'stable'}</span>
      </div>
      <TileVitals v={b.vitals || {}} seed={b.stay_id} />
    </button>
  )
}, (p, n) => p.active === n.active && p.onOpen === n.onOpen && p.b.stay_id === n.b.stay_id && p.b.criticality === n.b.criticality && p.b.level === n.b.level && !!p.b.notified === !!n.b.notified)

function LiveTrace({ base, k, seed, t, color, n = 60 }) {
  if (base == null) return <div className="h-10" />
  const pts = Array.from({ length: n }, (_, i) => liveVal(base, k, seed, t - (n - 1 - i) / 2))   // last 30 s, 2 samples per second
  const lo = Math.min(...pts), hi = Math.max(...pts), span = Math.max(hi - lo, VIT[k].jit * 2)
  const mid = (lo + hi) / 2
  const d = pts.map((v, i) => `${i ? 'L' : 'M'}${(i / (n - 1)) * 200},${20 - ((v - mid) / span) * 30}`).join(' ')
  return <svg viewBox="0 0 200 40" className="h-10 w-full" preserveAspectRatio="none" aria-hidden><path d={d} fill="none" stroke={color} strokeWidth="1.8" vectorEffect="non-scaling-stroke" /><circle cx="200" cy={20 - ((pts[n - 1] - mid) / span) * 30} r="2.5" fill={color} /></svg>
}

function Trend({ series, k, color }) {
  const pts = (series || []).filter((r) => r[k] != null)
  if (pts.length < 2) return <div className="text-[11px] text-muted">no earlier values</div>
  const xs = pts.map((r) => r.hour_idx), ys = pts.map((r) => r[k]), x0 = Math.min(...xs), x1 = Math.max(...xs), lo = Math.min(...ys), hi = Math.max(...ys), sp = Math.max(hi - lo, 1e-6)
  const d = pts.map((r, i) => `${i ? 'L' : 'M'}${((r.hour_idx - x0) / Math.max(x1 - x0, 1)) * 200},${26 - ((r[k] - lo) / sp) * 22}`).join(' ')
  return (
    <div title={`last ${x1 - x0} h: ${fmt(k, lo)} – ${fmt(k, hi)}`}>
      <svg viewBox="0 0 200 28" className="h-7 w-full" preserveAspectRatio="none" aria-hidden><path d={d} fill="none" stroke={color} strokeOpacity="0.55" strokeWidth="1.5" vectorEffect="non-scaling-stroke" /></svg>
      <div className="flex justify-between text-[10px] text-muted"><span>{x1 - x0} h ago</span><span>{fmt(k, lo)}–{fmt(k, hi)}</span><span>now</span></div>
    </div>
  )
}

function LiveVitals({ bed, series }) {
  const t = useClock()
  const v = bed.vitals || {}
  return (
    <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-5">
      {Object.entries(VIT).map(([k, m]) => {
        const val = liveVal(v[k], k, bed.stay_id, t), dia = k === 'sbp' ? liveVal(v.dbp, 'sbp', bed.stay_id + 7, t) : null
        const bad = val != null && m.bad(val)
        return (
          <div key={k} className={`rounded-lg border bg-bg/60 p-2.5 ${bad ? 'border-red-400/60' : 'border-line'}`}>
            <div className="flex items-center justify-between text-xs text-muted"><span>{m.label}</span>{val != null && <span className="h-1.5 w-1.5 animate-pulse rounded-full" style={{ background: m.color }} />}</div>
            <div className="font-mono text-2xl font-black tabular-nums" style={{ color: bad ? '#fca5a5' : m.color }}>
              {k === 'sbp' && val != null ? `${fmt(k, val)}/${fmt(k, dia)}` : fmt(k, val)}<span className="ml-1 text-xs font-normal text-muted">{val != null ? m.unit : 'not measured'}</span>
            </div>
            <LiveTrace base={v[k]} k={k} seed={bed.stay_id} t={t} color={m.color} />
            <Trend series={series} k={k} color={m.color} />
          </div>
        )
      })}
    </div>
  )
}

export default function Icu() {
  const [b, setB] = useState(null)
  const [err, setErr] = useState(null)
  const [sel, setSel] = useState(null)
  const [stay, setStay] = useState(null)
  const [log, setLog] = useState([])
  const [role, setRole] = useState('nurse')
  const [filter, setFilter] = useState('all')
  const [q, setQ] = useState('')
  const [toast, show] = useToast()
  const [nmeta, setNmeta] = useState(null)
  const [composing, setComposing] = useState(false)
  const [auto, setAuto] = useState(false)
  const detailRef = useRef(null)
  const load = useCallback(() => {
    api('/api/icu/board').then((d) => { setB(d); setSel((s) => s ?? d.beds[0]?.stay_id) }).catch((e) => setErr(e.message))
    api('/api/icu/false-alarms').then(setLog).catch(() => {})
  }, [])
  useEffect(() => { load(); const id = setInterval(load, 10000); return () => clearInterval(id) }, [load])
  useEffect(() => { api('/api/notify/meta').then((m) => { setNmeta(m); setAuto(!!m.auto?.enabled) }).catch(() => {}) }, [])
  useEffect(() => { const m = window.location.hash.match(/stay=(\d+)/); if (m) setSel(Number(m[1])) }, [])
  useEffect(() => { if (sel) api(`/api/icu/stay/${sel}`).then(setStay).catch(() => setStay(null)) }, [sel])
  const beds = useMemo(() => (b?.beds || []).filter((x) => (filter === 'all' || x.level === filter) && (!q || String(x.stay_id).includes(q) || String(x.bed) === q.trim() || x.risks.some((r) => r.name.toLowerCase().includes(q.toLowerCase())))), [b, filter, q])
  const open = useCallback((id) => { setSel(id); setTimeout(() => detailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50) }, [])
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest?.('input, textarea, select') || !beds.length) return
      const i = beds.findIndex((x) => x.stay_id === sel)
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); setSel(beds[Math.min(beds.length - 1, i + 1)].stay_id) }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); setSel(beds[Math.max(0, i - 1)].stay_id) }
    }
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey)
  }, [beds, sel])
  if (!b) return <Shell><Loading error={err} /></Shell>
  const cur = b.beds.find((x) => x.stay_id === sel)
  const counts = Object.fromEntries(LEVELS.map((k) => [k, b.beds.filter((x) => x.level === k).length]))
  const setAutoOn = async (v) => {
    setAuto(v)
    try { const r = await api('/api/notify/auto', { enabled: v }); show(v ? `Auto-alert on${r.sent_now ? ` – ${r.sent_now} alert${r.sent_now === 1 ? '' : 's'} sent now` : ''}` : 'Auto-alert off'); load() } catch (e) { show(e.message, 'err'); setAuto(!v) }
  }
  const onFalse = async (r, reason, note) => {
    try {
      await api('/api/icu/verdict', { stay_id: cur.stay_id, risk: r.domain, verdict: 'false_alarm', role, reason_code: reason, note })
      show(`False alarm logged for bed ${cur.bed} – ${r.name}`); load(); return true
    } catch (e) { show(e.message, 'err'); return false }
  }
  const tr = TREND[cur?.trend] || TREND.steady

  return (
    <Shell right={<>
      <label className="flex items-center gap-2 text-xs text-muted" title="Who is entering false alarms and alerts">
        <select className="rounded-md border border-line bg-bg px-2 py-1 text-sm text-text" value={role} onChange={(e) => setRole(e.target.value)}><option>nurse</option><option>doctor</option></select>
      </label>
      <AutoSwitch value={auto} onChange={setAutoOn} />
      <Counter c={b.counter} label="False alarms" />
    </>}>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-black">ICU risk</h1>
        <span className={`rounded px-2 py-0.5 text-xs font-semibold ${/real/i.test(b.data_note) ? 'bg-emerald-500/15 text-emerald-300' : 'bg-panel-2 text-muted'}`} title={b.source?.note || ''}>{b.data_note}</span>
        {b.source?.missing?.length > 0 && <span className="text-xs text-muted">not recorded in this data: {b.source.missing.join(', ')}</span>}
        <span className="ml-auto flex items-center gap-1.5 text-xs text-emerald-300"><span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />live</span>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-3 md:grid-cols-4">
        {LEVELS.map((k) => (
          <button key={k} onClick={() => setFilter(filter === k ? 'all' : k)}
                  className={`rounded-xl border bg-panel px-3 py-2 text-left transition hover:bg-panel-2 ${filter === k ? 'ring-2' : ''}`} style={{ borderColor: `${LEVEL[k].color}55`, '--tw-ring-color': LEVEL[k].color }}>
            <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wide" style={{ color: LEVEL[k].color }}><span>{LEVEL[k].icon} {k}</span>{filter === k && <span>filter on</span>}</div>
            <div className="text-2xl font-black">{counts[k]} <span className="text-xs font-normal text-muted">patients</span></div>
          </button>
        ))}
      </div>

      <section className="mb-4 rounded-xl border border-line bg-panel p-3">
        <div className="mb-2 flex items-center gap-2">
          <input className={`${inputCls} max-w-xs py-1.5 text-sm`} placeholder="Search bed, patient # or risk…" value={q} onChange={(e) => setQ(e.target.value)} />
          <span className="text-xs text-muted">{beds.length} beds · sickest first · click a bed or use the arrow keys</span>
        </div>
        <div className="grid max-h-[46vh] content-start gap-2 overflow-auto pr-1" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(168px, 1fr))' }}>
          {beds.map((x) => <BedTile key={x.stay_id} b={x} active={x.stay_id === sel} onOpen={open} />)}
          {!beds.length && <p className="p-4 text-sm text-muted">No beds match.</p>}
        </div>
      </section>

      {cur && (
        <section ref={detailRef} key={cur.stay_id} className="scroll-mt-16 rounded-xl border border-line bg-panel p-4 anim-fade-up">
          <div className="flex flex-wrap items-center gap-4">
            <Gauge value={cur.criticality} level={cur.level} />
            <div className="flex-1">
              <div className="text-xs text-muted">Bed {cur.bed} · patient #{cur.stay_id} · hour {num(cur.hour)} in ICU</div>
              <div className="text-2xl font-black" style={{ color: LEVEL[cur.level]?.color }}>{cur.level}</div>
              <div className={`text-sm ${tr[1]}`}>{tr[0]} {tr[2]}{cur.change_6h != null ? ` · ${cur.change_6h > 0 ? '+' : ''}${num(cur.change_6h)} in the last 6 h` : ''}</div>
            </div>
            <div className="grid justify-items-end gap-1">
              <button onClick={() => setComposing(true)} disabled={!nmeta} className="flex items-center gap-2 rounded-lg bg-sky-500 px-3 py-2 text-sm font-bold text-white shadow-lg shadow-sky-500/20 transition hover:bg-sky-600">🔔 Notify doctor</button>
              <span className="text-xs text-muted">{cur.risks.length} active risk{cur.risks.length === 1 ? '' : 's'} · ranked by severity</span>
            </div>
          </div>
          {cur.notified && (
            <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-sky-400/30 bg-sky-500/10 px-3 py-2 text-sm">
              <span>🔔</span><span className={`rounded px-1.5 text-xs font-bold ${URG[cur.notified.urgency]?.cls || ''}`}>{URG[cur.notified.urgency]?.label}</span>
              <span className="text-xs text-muted">{cur.notified.auto ? 'auto-alert' : 'alert'} · last sent {String(cur.notified.ts).slice(11, 16)} UTC</span>
              <div className="flex w-full flex-wrap gap-x-4 gap-y-1">
                {(cur.notified.doctors || []).map((d) => (
                  <span key={d.doctor}><b>{d.doctor}</b>: <span className={(STATUS[d.status] || [])[1]}>{(STATUS[d.status] || [d.status])[0]}</span>{d.ack_ts ? <span className="text-xs text-muted"> {String(d.ack_ts).slice(11, 16)}</span> : ''}</span>
                ))}
              </div>
            </div>
          )}
          <div className="mt-4 grid gap-2">
            {cur.risks.length ? cur.risks.map((r) => (
              <RiskCard key={r.domain} r={r} reasons={b.reasons} onFalse={onFalse} done={cur.verdicts?.some((v) => v.risk === r.domain && v.verdict === 'false_alarm')} />
            )) : <p className="text-sm text-muted">No active risk for this patient.</p>}
          </div>
          <div className="mt-5 mb-2 flex items-center justify-between"><h3 className="text-sm font-bold">Vital signs</h3><span className="flex items-center gap-1.5 text-xs text-emerald-300"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />live</span></div>
          <LiveVitals bed={cur} series={stay?.stay_id === cur.stay_id ? stay.vitals : null} />
          <div className="mt-5 mb-1 flex items-center justify-between"><h3 className="text-sm font-bold">Overall criticality during the stay</h3><span className="text-xs text-muted">hover for details</span></div>
          <Timeline points={stay?.stay_id === cur.stay_id ? stay.timeline : null} />
        </section>
      )}

      <section className="mt-4 rounded-xl border border-line bg-panel p-4">
        <h3 className="mb-2 text-sm font-bold">False-alarm log <span className="font-normal text-muted">· every entry teaches the model at {b.counter.threshold}</span></h3>
        {log.length ? (
          <div className="max-h-56 overflow-auto"><table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-panel text-muted"><tr><th className="py-1">Time</th><th>Patient</th><th>Risk</th><th>What it was</th><th>By</th></tr></thead>
            <tbody>{log.map((l, i) => <tr key={i} className="border-t border-line"><td className="py-1.5 pr-2 whitespace-nowrap text-muted">{l.ts?.slice(5, 16)}</td><td className="pr-2">#{l.stay_id}</td><td className="pr-2">{l.risk_name}</td><td className="pr-2">{l.reason}{l.note ? <span className="text-muted"> – {l.note}</span> : ''}</td><td>{l.staff_role}</td></tr>)}</tbody>
          </table></div>
        ) : <p className="text-sm text-muted">No false alarms logged yet.</p>}
      </section>
      {composing && cur && nmeta && <NotifyPanel bed={cur} meta={nmeta} role={role} onClose={() => setComposing(false)} onSent={(r) => { setComposing(false); show(`Sent to ${r.to.join(' + ')}`); load() }} />}
      <Toast {...(toast || {})} />
    </Shell>
  )
}
