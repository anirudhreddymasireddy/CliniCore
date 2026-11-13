import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import TopBar, { BarButton } from '../components/TopBar'
import BedTile from '../components/BedTile'
import NotificationStack from '../components/NotificationStack'
import PatientDashboard from '../components/PatientDashboard'
import Toast from '../components/Toast'
import { getJSON } from '../lib/api'
import { playChime, unlockAudio } from '../lib/sound'
import { sinceText } from '../lib/clinical'
import { latestVitals, riskLine, useWardActions } from './useWardActions'

const SNOOZE_MS = 30 * 60 * 1000

export function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen?.()
  else document.documentElement.requestFullscreen?.().catch(() => {})
}

// Scripted pitch demo from the local snapshot only (GET /api/presentation is always snapshot-backed).
export default function Presentation() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [stateHour, setStateHour] = useState({})
  const [notes, setNotes] = useState([])
  const [openToken, setOpenToken] = useState(null)
  const [muted, setMuted] = useState(false)
  const [now, setNow] = useState(Date.now())
  const timers = useRef([])
  const utcNow = () => new Date().toISOString().replace('T', ' ').slice(0, 19)
  const [sessionStart, setSessionStart] = useState(utcNow)
  const mutedRef = useRef(muted)
  mutedRef.current = muted

  useEffect(() => { getJSON('/api/presentation').then(setData).catch((e) => setError(String(e))) }, [])
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 15000); return () => clearInterval(t) }, [])
  const initialHours = useMemo(() => Object.fromEntries((data?.beds || []).map((b) => [b.token, b.display_hour])), [data])
  useEffect(() => { setStateHour(initialHours) }, [initialHours])

  const ackToken = useCallback((ctx) => setNotes((ns) => ns.map((n) => (n.bed === ctx.bed ? { ...n, ack: true } : n))), [])
  const { toast, act, refreshKey, reset: resetToast } = useWardActions({ onReturnToWard: () => setOpenToken(null), onAcknowledged: ackToken })

  const trigger = useCallback((role) => {
    if (!data) return
    const bed = data.beds.find((b) => b.role === role)
    const p = data.patients[bed.token]
    const row = p.hours.find((h) => h.hour_index === bed.alert_hour)
    setStateHour((s) => ({ ...s, [bed.token]: bed.alert_hour }))
    setNotes((ns) => (ns.some((n) => n.id === role) ? ns : [...ns, {
      id: role, token: bed.token, bed: bed.bed, anonId: p.anon_id, level: row.alert_level, components: row.trigger_components,
      text: row.alert_text, alertHour: bed.alert_hour, createdAt: Date.now(), ack: false, snoozedUntil: 0,
    }]))
    if (row.alert_level === 'CRITICAL' && !mutedRef.current) playChime()
  }, [data])

  const reset = useCallback(() => {
    timers.current.forEach(clearTimeout); timers.current = []
    setStateHour(initialHours); setNotes([]); setOpenToken(null); resetToast(); setSessionStart(utcNow())
  }, [initialHours, resetToast])

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest?.('select, input, textarea') || e.metaKey || e.ctrlKey || e.altKey) return
      const k = e.key.toLowerCase()
      if (k === 'n') {
        unlockAudio()
        if (notes.some((n) => n.id === 'critical') || timers.current.length) return
        timers.current.push(setTimeout(() => { timers.current = []; trigger('critical') }, (data?.trigger_delay_seconds ?? 2) * 1000))
      } else if (k === 'm') trigger('medium')
      else if (k === 'r') reset()
      else if (k === 'f') toggleFullscreen()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [trigger, reset, notes, data])

  if (error) return <p className="p-8 text-2xl">Snapshot not available: {error}. Run ui/backend export_snapshot.py first.</p>
  if (!data) return <p className="p-8 text-2xl text-muted">Loading demo ward…</p>

  const visible = notes.filter((n) => !n.ack && n.snoozedUntil <= now)
    .sort((a, b) => (a.level === b.level ? b.createdAt - a.createdAt : a.level === 'CRITICAL' ? -1 : 1))
    .map((n) => ({ ...n, since: sinceText(null, (now - n.createdAt) / 60000) }))
  const openBed = data.beds.find((b) => b.token === openToken)
  const log = (n, action) => act(action, { ward: 'P', bed: n.bed, anon_id: n.anonId, source: data.patients[n.token].source,
    replay_hour: n.alertHour, alert_hour: n.alertHour, tier: n.level, components: n.components || '' })

  return (
    <div className="flex min-h-screen flex-col gap-4 p-4">
      <TopBar title={<a href="#/">CliniCore · ICU Central Monitor</a>} subtitle={`${data.ward_name} · presentation mode · local snapshot`}>
        <span className="hidden text-sm text-muted lg:inline">Keys: N critical · M medium · R reset · F full screen</span>
        <BarButton onClick={() => setMuted((m) => !m)} testid="sound-toggle" label="Toggle sound">{muted ? '🔕 Muted' : '🔔 Sound on'}</BarButton>
        <BarButton onClick={toggleFullscreen} testid="fullscreen" label="Full screen">⛶ Full screen</BarButton>
      </TopBar>
      <div className="flex flex-1 items-start gap-4">
        <main className="min-w-0 flex-1">
          {openBed ? (
            <PatientDashboard patient={data.patients[openBed.token]} meta={data.meta} nowHour={stateHour[openBed.token]}
                              anchorLabel={stateHour[openBed.token] === openBed.alert_hour ? 'alert' : 'now'} ward="P" bed={openBed.bed}
                              onBack={() => setOpenToken(null)} onAction={act} refreshKey={refreshKey} vitalsWindowH={8} actionsSince={sessionStart} />
          ) : (
            <section>
              <h2 className="mb-3 text-xl font-extrabold">Ward overview <span className="text-base font-medium text-muted">· {data.beds.length} beds · anonymized</span></h2>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(250px,1fr))] gap-4" data-testid="ward-grid">
                {data.beds.map((b) => {
                  const p = data.patients[b.token]
                  const h = stateHour[b.token] ?? b.display_hour
                  const row = p.hours.find((x) => x.hour_index === h)
                  return <BedTile key={b.token} bed={b.bed} anonId={p.anon_id} tier={row?.tier}
                                  vitals={latestVitals(p.vitals.points, (h + 1) * 60, row)} risks={riskLine(row)}
                                  highlight={row?.tier === 'CRITICAL'} onOpen={() => setOpenToken(b.token)} />
                })}
              </div>
            </section>
          )}
        </main>
        <NotificationStack items={visible} onOpen={(n) => setOpenToken(n.token)}
                           onAck={(n) => { setNotes((ns) => ns.map((x) => (x.id === n.id ? { ...x, ack: true } : x))); log(n, 'acknowledge (notification)') }}
                           onSnooze={(n) => { setNotes((ns) => ns.map((x) => (x.id === n.id ? { ...x, snoozedUntil: Date.now() + SNOOZE_MS } : x))); log(n, 'snooze 30 min') }} />
      </div>
      <Toast toast={toast} />
    </div>
  )
}
