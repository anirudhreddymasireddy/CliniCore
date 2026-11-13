import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import TopBar, { BarButton } from '../components/TopBar'
import BedTile from '../components/BedTile'
import NotificationStack from '../components/NotificationStack'
import PatientDashboard from '../components/PatientDashboard'
import Toast from '../components/Toast'
import { getJSON } from '../lib/api'
import { playChime, unlockAudio } from '../lib/sound'
import { sinceText } from '../lib/clinical'
import { riskLine, useWardActions } from './useWardActions'
import { toggleFullscreen } from './Presentation'

const STEP_MS = 2000

// Replay over ICU hours 0-72 with data from the backend (live Exasol, or the snapshot when CAREPRIORITY_DATA=snapshot).
export default function Replay() {
  const [ward, setWard] = useState('A')
  const [data, setData] = useState(null)
  const [meta, setMeta] = useState(null)
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(null)
  const [hour, setHour] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [acks, setAcks] = useState({})
  const [openBed, setOpenBed] = useState(null)
  const [patients, setPatients] = useState({})
  const [muted, setMuted] = useState(false)
  const prevHour = useRef(0)

  useEffect(() => { getJSON('/api/meta').then(setMeta).catch((e) => setError(String(e))); getJSON('/api/health').then(setHealth).catch(() => {}) }, [])
  useEffect(() => {
    setData(null); setOpenBed(null); setAcks({}); setHour(0); setPlaying(false); prevHour.current = 0
    getJSON(`/api/ward/${ward}`).then(setData).catch((e) => setError(String(e)))
  }, [ward])
  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => setHour((h) => (h >= 72 ? 72 : h + 1)), STEP_MS)
    return () => clearInterval(t)
  }, [playing])
  useEffect(() => { if (hour >= 72) setPlaying(false) }, [hour])

  const alertsAt = useCallback((h) => (data?.beds || []).flatMap((b) => {
    const r = b.hours[String(h)]
    return r && r.alert_fired === 1 && ['MEDIUM', 'CRITICAL'].includes(r.alert_level)
      ? [{ id: `${ward}|${b.bed}|${h}`, bed: b.bed, token: b.token, anonId: b.anon_id, source: b.source, level: r.alert_level,
           components: r.trigger_components, text: r.alert_text, alertHour: h }] : []
  }), [data, ward])

  // chime once for a NEW critical alert when the replay steps forward by one hour
  useEffect(() => {
    if (data && hour === prevHour.current + 1 && !muted && alertsAt(hour).some((a) => a.level === 'CRITICAL')) playChime()
    prevHour.current = hour
  }, [hour, data, muted, alertsAt])

  const visible = useMemo(() => {
    const out = []
    for (let h = Math.max(0, hour - 5); h <= hour; h++) out.push(...alertsAt(h))
    return out.filter((n) => { const a = acks[n.id]; return !(a?.ack || (a?.until && a.until > hour * 60)) })
      .sort((a, b) => (a.level === b.level ? b.alertHour - a.alertHour : a.level === 'CRITICAL' ? -1 : 1))
      .map((n) => ({ ...n, since: sinceText(hour - n.alertHour) }))
  }, [alertsAt, acks, hour])

  const openPatient = useCallback((bed) => {
    const b = data.beds.find((x) => x.bed === bed)
    setOpenBed(bed)
    if (!patients[b.token]) getJSON(`/api/patient/${b.token}?upto=72`).then((p) => setPatients((ps) => ({ ...ps, [b.token]: p })))
  }, [data, patients])

  const ackBed = useCallback((ctx) => setAcks((a) => {
    const next = { ...a }
    for (const n of visible) if (n.bed === ctx.bed) next[n.id] = { ack: true }
    return next
  }), [visible])
  const { toast, act, refreshKey } = useWardActions({ onReturnToWard: () => setOpenBed(null), onAcknowledged: ackBed })
  const log = (n, action) => act(action, { ward, bed: n.bed, anon_id: n.anonId, source: n.source, replay_hour: hour,
    alert_hour: n.alertHour, tier: n.level, components: n.components || '' })

  if (error) return <p className="p-8 text-2xl">Backend not available: {error}</p>
  const bedObj = data?.beds.find((b) => b.bed === openBed)
  return (
    <div className="flex min-h-screen flex-col gap-4 p-4">
      <TopBar title={<a href="#/">CliniCore · ICU Central Monitor</a>}
              subtitle={`Replay mode · ICU hours 0–72 · data: ${health?.data_mode === 'snapshot' ? 'local snapshot' : 'live Exasol (read-only)'}`}>
        <BarButton active={ward === 'A'} onClick={() => setWard('A')} testid="ward-A">Ward A · eICU</BarButton>
        <BarButton active={ward === 'B'} onClick={() => setWard('B')} testid="ward-B">Ward B · MIMIC</BarButton>
        <BarButton onClick={() => setMuted((m) => !m)} testid="sound-toggle">{muted ? '🔕 Muted' : '🔔 Sound on'}</BarButton>
        <BarButton onClick={toggleFullscreen} testid="fullscreen">⛶</BarButton>
      </TopBar>
      <div className="flex flex-wrap items-center gap-4 rounded-2xl border border-line bg-panel px-5 py-3">
        <BarButton active={playing} testid="play" onClick={() => { unlockAudio(); if (hour >= 72) setHour(0); setPlaying((p) => !p) }}>
          {playing ? '⏸ Pause' : '▶ Play'}
        </BarButton>
        <BarButton testid="restart" onClick={() => { setPlaying(false); setHour(0); setAcks({}) }}>⟲ Restart</BarButton>
        <span className="min-w-44 text-2xl font-black tabular-nums" data-testid="clock">ICU hour {hour}</span>
        <input type="range" min="0" max="72" step="1" value={hour} onChange={(e) => setHour(Number(e.target.value))}
               aria-label="ICU hour" data-testid="slider" className="h-3 min-w-64 flex-1 cursor-pointer accent-blue-400" />
      </div>
      <div className="flex flex-1 items-start gap-4">
        <main className="min-w-0 flex-1">
          {!data || !meta ? <p className="text-xl text-muted">Loading ward…</p> : openBed && bedObj ? (
            patients[bedObj.token]
              ? <PatientDashboard patient={patients[bedObj.token]} meta={meta} nowHour={hour} anchorLabel="now" ward={ward} bed={openBed}
                                  onBack={() => setOpenBed(null)} onAction={act} refreshKey={refreshKey} />
              : <p className="text-xl text-muted">Loading patient…</p>
          ) : (
            <section>
              <h2 className="mb-3 text-xl font-extrabold">{data.name} <span className="text-base font-medium text-muted">· {data.selection}</span></h2>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(250px,1fr))] gap-4" data-testid="ward-grid">
                {data.beds.map((b) => {
                  const r = b.hours[String(hour)]
                  return <BedTile key={b.token} bed={b.bed} anonId={b.anon_id} tier={r?.tier}
                                  vitals={{ hr: r?.heart_rate, map: r?.map_mmhg, rr: r?.resp_rate, spo2: r?.spo2 }}
                                  risks={riskLine(r)} note="Not scored this hour (in crisis, discharged or no recent vitals)"
                                  highlight={r?.tier === 'CRITICAL'} onOpen={() => openPatient(b.bed)} />
                })}
              </div>
            </section>
          )}
        </main>
        <NotificationStack items={visible} onOpen={(n) => openPatient(n.bed)}
                           onAck={(n) => { setAcks((a) => ({ ...a, [n.id]: { ack: true } })); log(n, 'acknowledge (notification)') }}
                           onSnooze={(n) => { setAcks((a) => ({ ...a, [n.id]: { until: hour * 60 + 30 } })); log(n, 'snooze 30 min') }} />
      </div>
      <Toast toast={toast} />
    </div>
  )
}
