import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, num } from './api'
import { Counter, LEVEL, Loading, Shell, Toast, inputCls, useToast } from './ui'
import { STATUS, Sbar, URG } from './notify'
import { playChime, unlockAudio } from '../lib/sound'

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

function PatientRow({ b, active, onClick }) {
  const l = LEVEL[b.level] || {}, t = TREND[b.trend] || TREND.steady, top = b.risks[0]
  return (
    <button onClick={onClick} className={`group grid w-full grid-cols-[2.6rem_1fr_auto] items-center gap-3 rounded-lg border px-3 py-2 text-left transition ${active ? 'border-sky-400/70 bg-sky-500/10' : 'border-transparent hover:border-line hover:bg-panel-2'}`}>
      <div className="relative grid h-10 w-10 place-items-center rounded-lg text-base font-black" style={{ background: `${l.color}22`, color: l.color }}>
        {num(b.criticality)}
      </div>
      <div className="min-w-0">
        <div className="flex items-center gap-2 text-sm"><b>Bed {b.bed}</b><span className="text-xs text-muted">#{b.stay_id}</span><span className={`text-xs ${t[1]}`}>{t[0]} {t[2]}</span>{b.notified && <span title={b.notified.any_open ? 'doctor alerted, waiting for reply' : 'doctor responded'} className={b.notified.any_open ? 'text-amber-300' : 'text-emerald-300'}>🔔</span>}</div>
        <div className="truncate text-xs text-muted">{top ? <>{DOMAIN_ICON[top.domain]} {top.name} · {top.evidence}</> : 'No active risk'}</div>
      </div>
      <div className="flex gap-1">{b.risks.map((r) => <span key={r.domain} title={`${r.name}: ${r.tier}`} className="h-6 w-1.5 rounded-full" style={{ background: LEVEL[r.tier]?.color }} />)}</div>
    </button>
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


const RANK = { Low: 0, Watch: 1, High: 2, Critical: 3 }
function stateAt(points, h) {
  let cur = null
  for (const p of points) { if (p[0] <= h) cur = p; else break }
  return cur ? { score: cur[1], level: cur[2], domain: cur[3], tier: cur[4] } : { score: 0, level: 'Low', domain: '', tier: 0 }
}

function WallView({ beds, onOpen }) {
  const [tl, setTl] = useState(null)
  const [mode, setMode] = useState('now')            // 'now' = latest values, 'replay' = play the stays hour by hour
  const [hour, setHour] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2)
  const [sound, setSound] = useState(true)
  const [feed, setFeed] = useState([])
  const prev = useRef({})
  const box = useRef(null)
  useEffect(() => { api('/api/icu/timelines').then(setTl).catch(() => setTl({})) }, [])
  const maxH = useMemo(() => Math.min(120, Math.max(24, ...Object.values(tl || {}).map((p) => p.at(-1)?.[0] || 0))), [tl])
  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => setHour((h) => { if (h >= maxH) { setPlaying(false); return h } return h + 1 }), 1000 / speed)
    return () => clearInterval(t)
  }, [playing, speed, maxH])
  const tiles = useMemo(() => beds.map((b) => {
    if (mode === 'now' || !tl) return { ...b, score: b.criticality, lvl: b.level, dom: b.risks[0]?.domain, domName: b.risks[0]?.name }
    const s = stateAt(tl[String(b.stay_id)] || [], hour)
    return { ...b, score: s.score, lvl: s.level, dom: s.domain, domName: DOMAIN_NAME[s.domain] }
  }).sort((a, b) => (RANK[b.lvl] - RANK[a.lvl]) || (b.score - a.score)), [beds, tl, mode, hour])
  // new escalations -> alert feed (+ chime for Critical)
  useEffect(() => {
    if (mode !== 'replay') { prev.current = {}; return }
    const ev = []
    for (const t of tiles) {
      const before = prev.current[t.stay_id]
      if (before !== undefined && RANK[t.lvl] > RANK[before] && RANK[t.lvl] >= 2) ev.push({ id: `${t.stay_id}-${hour}`, hour, bed: t.bed, stay_id: t.stay_id, lvl: t.lvl, dom: t.domName, score: t.score })
      prev.current[t.stay_id] = t.lvl
    }
    if (ev.length) { setFeed((f) => [...ev, ...f].slice(0, 40)); if (sound && ev.some((e) => e.lvl === 'Critical')) playChime() }
  }, [tiles, mode, hour, sound])
  const counts = Object.fromEntries(['Critical', 'High', 'Watch', 'Low'].map((k) => [k, tiles.filter((t) => t.lvl === k).length]))
  const startReplay = () => { unlockAudio(); setMode('replay'); setFeed([]); prev.current = {}; setHour(0); setPlaying(true) }
  const full = () => (document.fullscreenElement ? document.exitFullscreen() : box.current?.requestFullscreen?.())
  useEffect(() => {
    const k = (e) => { if (e.target.closest?.('input, textarea, select')) return
      if (e.key === ' ' && mode === 'replay') { e.preventDefault(); setPlaying((p) => !p) }
      if (e.key === 'f') full() }
    window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k)
  })
  return (
    <div ref={box} className="rounded-xl border border-line bg-bg p-3">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="flex rounded-lg bg-panel p-0.5 text-sm font-bold">
          <button onClick={() => { setMode('now'); setPlaying(false) }} className={`rounded-md px-3 py-1 ${mode === 'now' ? 'bg-sky-500 text-white' : 'text-muted'}`}>Now</button>
          <button onClick={startReplay} className={`rounded-md px-3 py-1 ${mode === 'replay' ? 'bg-sky-500 text-white' : 'text-muted'}`}>Replay stays</button>
        </div>
        {mode === 'replay' && <>
          <button onClick={() => setPlaying((p) => !p)} className="rounded-md bg-panel px-3 py-1 text-sm font-bold" aria-label={playing ? 'pause' : 'play'}>{playing ? '❚❚ Pause' : '▶ Play'}</button>
          <input type="range" min="0" max={maxH} value={hour} onChange={(e) => { setHour(Number(e.target.value)); setPlaying(false) }} className="w-48 accent-sky-400" aria-label="hour" />
          <span className="w-28 text-sm"><b>hour {hour}</b><span className="text-muted"> / {maxH}</span></span>
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="rounded-md border border-line bg-panel px-2 py-1 text-sm">{[1, 2, 4, 8].map((x) => <option key={x} value={x}>{x} h/s</option>)}</select>
        </>}
        <div className="ml-auto flex items-center gap-3 text-sm">
          {['Critical', 'High', 'Watch', 'Low'].map((k) => <span key={k} className="font-bold" style={{ color: LEVEL[k].color }}>{LEVEL[k].icon} {counts[k]}</span>)}
          <button onClick={() => setSound((x) => !x)} className="rounded-md bg-panel px-2 py-1" title="chime on new Critical">{sound ? '🔊' : '🔈'}</button>
          <button onClick={full} className="rounded-md bg-panel px-2 py-1" title="full screen (F)">⛶</button>
        </div>
      </div>
      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_280px]">
        <div className="grid content-start gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}>
          {tiles.map((t) => {
            const l = LEVEL[t.lvl] || LEVEL.Low
            return (
              <button key={t.stay_id} onClick={() => onOpen(t.stay_id)} title={`patient #${t.stay_id}`}
                      className={`rounded-lg border-2 p-2 text-left transition hover:brightness-125 ${t.lvl === 'Critical' ? 'anim-soft-pulse' : ''}`}
                      style={{ borderColor: `${l.color}${t.lvl === 'Low' ? '44' : 'cc'}`, background: `${l.color}${t.lvl === 'Low' ? '0d' : '1f'}` }}>
                <div className="flex items-center justify-between text-xs"><b>Bed {t.bed}</b>{t.notified && <span>🔔</span>}</div>
                <div className="text-3xl font-black leading-tight" style={{ color: t.lvl === 'Low' ? '#a6b2c8' : l.color }}>{num(t.score)}</div>
                <div className="truncate text-xs text-muted">{t.dom ? `${DOMAIN_ICON[t.dom] || ''} ${t.domName || ''}` : 'stable'}</div>
              </button>
            )
          })}
        </div>
        <aside className="rounded-lg border border-line bg-panel p-3">
          <h3 className="mb-2 text-sm font-bold">Alerts {mode === 'replay' ? `· replay` : ''}</h3>
          {mode === 'now' ? <p className="text-xs text-muted">Press "Replay stays" to play the ICU stays hour by hour. Patients who escalate to High or Critical appear here, with a chime for Critical. Space = pause, F = full screen.</p>
            : feed.length ? <div className="grid max-h-[60vh] gap-1.5 overflow-auto">{feed.map((e) => (
              <button key={e.id} onClick={() => onOpen(e.stay_id)} className="anim-fade-up rounded-md border-l-4 bg-panel-2 px-2 py-1.5 text-left text-xs" style={{ borderColor: LEVEL[e.lvl].color }}>
                <b>Bed {e.bed}</b> → <b style={{ color: LEVEL[e.lvl].color }}>{e.lvl}</b> {num(e.score)}<br /><span className="text-muted">hour {e.hour} · {e.dom || 'combined'}</span>
              </button>))}</div> : <p className="text-xs text-muted">No escalations yet.</p>}
        </aside>
      </div>
    </div>
  )
}

function NotifyPanel({ bed, meta, role, onClose, onSent }) {
  const [docs, setDocs] = useState([])
  const [urg, setUrg] = useState(bed.level === 'Critical' ? 'emergency' : bed.level === 'High' ? 'urgent' : 'routine')
  const [risks, setRisks] = useState(bed.risks.map((r) => r.domain))
  const [note, setNote] = useState('')
  const [preview, setPreview] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  useEffect(() => {
    const t = setTimeout(() => api(`/api/notify/preview?stay_id=${bed.stay_id}&urgency=${urg}&risks=${risks.join(',')}&note=${encodeURIComponent(note)}`)
      .then((p) => { setPreview(p.message); setDocs((d) => (d.length ? d : p.suggested)) }).catch(() => {}), 200)
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
          <div><div className="mb-1.5 text-sm font-bold">Message preview <span className="font-normal text-muted">(SBAR, measured values only)</span></div><Sbar text={preview} /></div>
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
  const [view, setView] = useState(() => (window.location.hash.includes('view=wall') ? 'wall' : 'list'))
  const [auto, setAuto] = useState(false)
  useEffect(() => { api('/api/notify/meta').then((m) => { setNmeta(m); setAuto(!!m.auto?.enabled) }).catch(() => {}) }, [])
  useEffect(() => { const m = window.location.hash.match(/stay=(\d+)/); if (m) setSel(Number(m[1])) }, [])
  const setAutoOn = async (v) => {
    setAuto(v)
    try { const r = await api('/api/notify/auto', { enabled: v }); show(v ? `Auto-notify on${r.sent_now ? ` – ${r.sent_now} alert${r.sent_now === 1 ? '' : 's'} sent now` : ''}` : 'Auto-notify off'); load() } catch (e) { show(e.message, 'err'); setAuto(!v) }
  }
  const load = useCallback(() => {
    api('/api/icu/board').then((d) => { setB(d); setSel((s) => s ?? d.beds[0]?.stay_id) }).catch((e) => setErr(e.message))
    api('/api/icu/false-alarms').then(setLog).catch(() => {})
  }, [])
  useEffect(() => { load(); const t = setInterval(load, 10000); return () => clearInterval(t) }, [load])
  useEffect(() => { if (sel) api(`/api/icu/stay/${sel}`).then(setStay).catch(() => setStay(null)) }, [sel])
  const beds = useMemo(() => (b?.beds || []).filter((x) => (filter === 'all' || x.level === filter) && (!q || String(x.stay_id).includes(q) || String(x.bed) === q.trim() || x.risks.some((r) => r.name.toLowerCase().includes(q.toLowerCase())))), [b, filter, q])
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest?.('input, textarea, select') || !beds.length) return
      const i = beds.findIndex((x) => x.stay_id === sel)
      if (e.key === 'ArrowDown' || e.key === 'j') { e.preventDefault(); setSel(beds[Math.min(beds.length - 1, i + 1)].stay_id) }
      if (e.key === 'ArrowUp' || e.key === 'k') { e.preventDefault(); setSel(beds[Math.max(0, i - 1)].stay_id) }
    }
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey)
  }, [beds, sel])
  if (!b) return <Shell><Loading error={err} /></Shell>
  const cur = b.beds.find((x) => x.stay_id === sel)
  const counts = Object.fromEntries(LEVELS.map((k) => [k, b.beds.filter((x) => x.level === k).length]))
  const onFalse = async (r, reason, note) => {
    try {
      await api('/api/icu/verdict', { stay_id: cur.stay_id, risk: r.domain, verdict: 'false_alarm', role, reason_code: reason, note })
      show(`False alarm logged for bed ${cur.bed} – ${r.name}`); load(); return true
    } catch (e) { show(e.message, 'err'); return false }
  }
  const t = TREND[cur?.trend] || TREND.steady

  return (
    <Shell right={<>
      <label className="flex items-center gap-2 text-xs text-muted" title="Who is entering false alarms and alerts">
        <select className="rounded-md border border-line bg-bg px-2 py-1 text-sm text-text" value={role} onChange={(e) => setRole(e.target.value)}><option>nurse</option><option>doctor</option></select>
      </label>
      <AutoSwitch value={auto} onChange={setAutoOn} />
      <Counter c={b.counter} label="False alarms" />
    </>}>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div><h1 className="text-2xl font-black">ICU risk <span className={`ml-2 rounded px-2 py-0.5 align-middle text-xs font-semibold ${/real/i.test(b.data_note) ? 'bg-emerald-500/15 text-emerald-300' : 'bg-panel-2 text-muted'}`}
              title={[b.source?.hospital && `Hospital ${b.source.hospital}`, b.source?.note].filter(Boolean).join(' · ')}>{b.data_note}</span>
          {b.source?.missing?.length > 0 && <span className="ml-2 align-middle text-xs font-normal text-muted" title={b.source.note}>not recorded in this data: {b.source.missing.join(', ')}</span>}</h1>
          <p className="text-sm text-muted">{view === 'list' ? 'Sickest first · ↑/↓ to move between patients · each warning lists the vitals and labs behind it' : 'Central station: every bed at a glance · click a bed to open it'}</p></div>
        <div className="flex rounded-lg border border-line bg-panel p-0.5 text-sm font-bold">
          {[['list', '☰ Patients'], ['wall', '▦ Wall']].map(([k, l]) => <button key={k} onClick={() => setView(k)} className={`rounded-md px-3 py-1.5 ${view === k ? 'bg-sky-500 text-white' : 'text-muted hover:text-text'}`}>{l}</button>)}
        </div>
      </div>

      {view === 'wall' ? <WallView beds={b.beds} onOpen={(id) => { setSel(id); setView('list') }} /> : <>
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        {LEVELS.map((k) => (
          <button key={k} onClick={() => setFilter(filter === k ? 'all' : k)}
                  className={`rounded-xl border bg-panel p-3 text-left transition hover:bg-panel-2 ${filter === k ? 'ring-2' : ''}`} style={{ borderColor: `${LEVEL[k].color}55`, '--tw-ring-color': LEVEL[k].color }}>
            <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wide" style={{ color: LEVEL[k].color }}><span>{LEVEL[k].icon} {k}</span>{filter === k && <span>filter on</span>}</div>
            <div className="text-3xl font-black">{counts[k]}</div>
            <div className="text-xs text-muted">patients</div>
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(320px,400px)_minmax(0,1fr)]">
        <section className="flex max-h-[calc(100vh-240px)] min-h-[420px] flex-col rounded-xl border border-line bg-panel">
          <div className="flex items-center gap-2 border-b border-line p-2">
            <input className={`${inputCls} py-1.5 text-sm`} placeholder="Search bed, patient # or risk…" value={q} onChange={(e) => setQ(e.target.value)} />
            <span className="shrink-0 text-xs text-muted">{beds.length}</span>
          </div>
          <div className="flex-1 space-y-0.5 overflow-auto p-1.5">
            {beds.map((x) => <PatientRow key={x.stay_id} b={x} active={x.stay_id === sel} onClick={() => setSel(x.stay_id)} />)}
            {!beds.length && <p className="p-4 text-sm text-muted">No patients match.</p>}
          </div>
        </section>

        <div className="grid content-start gap-4">
          {cur ? (
            <section key={cur.stay_id} className="rounded-xl border border-line bg-panel p-4 anim-fade-up">
              <div className="flex flex-wrap items-center gap-4">
                <Gauge value={cur.criticality} level={cur.level} />
                <div className="flex-1">
                  <div className="text-xs text-muted">Bed {cur.bed} · patient #{cur.stay_id} · hour {num(cur.hour)} in ICU</div>
                  <div className="text-2xl font-black" style={{ color: LEVEL[cur.level]?.color }}>{cur.level}</div>
                  <div className={`text-sm ${t[1]}`}>{t[0]} {t[2]}{cur.change_6h != null ? ` · ${cur.change_6h > 0 ? '+' : ''}${num(cur.change_6h)} in the last 6 h` : ''}</div>
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
              <div className="mt-5 mb-1 flex items-center justify-between"><h3 className="text-sm font-bold">Overall criticality during the stay</h3><span className="text-xs text-muted">hover for details</span></div>
              <Timeline points={stay?.stay_id === cur.stay_id ? stay.timeline : null} />
            </section>
          ) : <section className="rounded-xl border border-line bg-panel p-6 text-muted">Select a patient.</section>}

          <section className="rounded-xl border border-line bg-panel p-4">
            <h3 className="mb-2 text-sm font-bold">False-alarm log <span className="font-normal text-muted">· every entry teaches the model at {b.counter.threshold}</span></h3>
            {log.length ? (
              <div className="max-h-56 overflow-auto"><table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-panel text-muted"><tr><th className="py-1">Time</th><th>Patient</th><th>Risk</th><th>What it was</th><th>By</th></tr></thead>
                <tbody>{log.map((l, i) => <tr key={i} className="border-t border-line"><td className="py-1.5 pr-2 whitespace-nowrap text-muted">{l.ts?.slice(5, 16)}</td><td className="pr-2">#{l.stay_id}</td><td className="pr-2">{l.risk_name}</td><td className="pr-2">{l.reason}{l.note ? <span className="text-muted"> – {l.note}</span> : ''}</td><td>{l.staff_role}</td></tr>)}</tbody>
              </table></div>
            ) : <p className="text-sm text-muted">No false alarms logged yet.</p>}
          </section>
        </div>
      </div>
      </>}
      {composing && cur && nmeta && <NotifyPanel bed={cur} meta={nmeta} role={role} onClose={() => setComposing(false)} onSent={(r) => { setComposing(false); show(`Sent to ${r.to.join(' + ')}`); load() }} />}
      <Toast {...(toast || {})} />
    </Shell>
  )
}
