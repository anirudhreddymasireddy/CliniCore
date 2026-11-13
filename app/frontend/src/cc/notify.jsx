import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { playChime, unlockAudio } from '../lib/sound'

// Who is looking at this screen: 'nurse' or an on-call doctor id. Stored per browser, chosen on the Doctor inbox page.
const KEY = 'clinicore.viewer'
const SEEN = 'clinicore.seen'
const read = (k, d) => { try { return localStorage.getItem(k) ?? d } catch { return d } }
const write = (k, v) => { try { localStorage.setItem(k, v) } catch { /* private mode */ } }

export function useViewer() {
  const [v, setV] = useState(() => read(KEY, 'nurse'))
  useEffect(() => { const on = () => setV(read(KEY, 'nurse')); window.addEventListener('clinicore-viewer', on); return () => window.removeEventListener('clinicore-viewer', on) }, [])
  return [v, (x) => { write(KEY, x); window.dispatchEvent(new Event('clinicore-viewer')) }]
}

export const URG = {
  emergency: { label: 'Emergency', cls: 'bg-red-500 text-white', ring: '#F65353' },
  urgent: { label: 'Urgent', cls: 'bg-amber-400 text-black', ring: '#FFA511' },
  routine: { label: 'Routine', cls: 'bg-sky-500 text-white', ring: '#38bdf8' },
}
export const STATUS = { sent: ['waiting for doctor', 'text-amber-200'], acknowledged: ['acknowledged', 'text-emerald-300'], on_my_way: ['doctor on the way', 'text-emerald-300'], declined: ['declined', 'text-red-300'] }

export function Sbar({ text }) {
  return (
    <div className="grid gap-1 rounded-lg bg-bg p-3 font-mono text-xs leading-relaxed">
      {String(text || '').split('\n').map((l, i) => {
        const m = l.match(/^([SBAR]):\s?(.*)$/)
        return m ? <div key={i}><b className="text-sky-300">{m[1]}</b> {m[2]}</div> : <div key={i} className="pl-4">{l}</div>
      })}
    </div>
  )
}

export async function ack(n, doctor, action) {
  await api('/api/notify/ack', { notify_id: n.notify_id, doctor, action })
}

// Polls the inbox of the doctor using this screen; pops up every new notification once (with a chime for emergencies).
export function useInbox(viewer, every = 4000) {
  const [items, setItems] = useState([])
  const [popups, setPopups] = useState([])
  const first = useRef(true)
  const load = useCallback(async () => {
    if (!viewer || viewer === 'nurse') { setItems([]); return }
    try {
      const d = await api(`/api/notify?doctor=${viewer}&limit=100`)
      setItems(d)
      const seen = Number(read(`${SEEN}.${viewer}`, '0'))
      const fresh = d.filter((n) => n.notify_id > seen && n.status === 'sent')
      if (fresh.length) {
        write(`${SEEN}.${viewer}`, String(Math.max(...d.map((n) => n.notify_id))))
        if (!first.current || fresh.length <= 3) {
          setPopups((p) => [...fresh.filter((f) => !p.some((x) => x.notify_id === f.notify_id)), ...p].slice(0, 4))
          if (fresh.some((n) => n.urgency === 'emergency')) playChime()
        }
      }
      first.current = false
    } catch { /* backend restarting */ }
  }, [viewer])
  useEffect(() => { first.current = true; load(); const t = setInterval(load, every); return () => clearInterval(t) }, [load, every])
  return { items, popups, dismiss: (id) => setPopups((p) => p.filter((x) => x.notify_id !== id)), reload: load }
}

export function DoctorPopups() {
  const [viewer] = useViewer()
  const { popups, dismiss, reload } = useInbox(viewer)
  useEffect(() => { const u = () => unlockAudio(); window.addEventListener('pointerdown', u, { once: true }); return () => window.removeEventListener('pointerdown', u) }, [])
  if (!popups.length) return null
  const act = async (n, a) => { try { await ack(n, viewer, a) } finally { dismiss(n.notify_id); reload() } }
  return (
    <div className="fixed top-16 right-4 z-50 grid w-[440px] max-w-[calc(100vw-2rem)] gap-3" role="alert" aria-live="assertive">
      {popups.map((n) => {
        const u = URG[n.urgency] || URG.routine
        return (
          <div key={n.notify_id} className="anim-slide-in rounded-xl border-2 bg-panel p-4 shadow-2xl" style={{ borderColor: u.ring }}>
            <div className="mb-2 flex items-center gap-2">
              <span className={`rounded px-2 py-0.5 text-xs font-extrabold ${u.cls}`}>{u.label}</span>
              <b>Bed {n.bed} · #{n.stay_id}</b>
              <span className="ml-auto text-xs text-muted">{n.auto ? 'auto-alert' : `from ${n.sent_by}`} · {String(n.ts).slice(11, 16)} UTC</span>
            </div>
            <Sbar text={n.message} />
            <div className="mt-3 flex flex-wrap gap-2">
              <button onClick={() => act(n, 'on_my_way')} className="rounded-md bg-emerald-500 px-3 py-1.5 text-sm font-bold text-black hover:bg-emerald-400">On my way</button>
              <button onClick={() => act(n, 'acknowledged')} className="rounded-md bg-panel-2 px-3 py-1.5 text-sm font-bold hover:bg-line">Acknowledge</button>
              <a href={`#/icu?stay=${n.stay_id}`} onClick={() => dismiss(n.notify_id)} className="rounded-md px-3 py-1.5 text-sm text-sky-300 hover:bg-panel-2">Open patient</a>
              <button onClick={() => dismiss(n.notify_id)} className="ml-auto rounded-md px-2 py-1.5 text-sm text-muted hover:bg-panel-2">Later</button>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function InboxBell() {
  const [viewer] = useViewer()
  const { items } = useInbox(viewer, 6000)
  const open = items.filter((n) => n.status === 'sent').length
  if (viewer === 'nurse') return <a href="#/inbox" className="rounded-md px-2 py-1 text-xs text-muted hover:bg-panel-2" title="Switch this screen to a doctor">👤 nurse</a>
  return (
    <a href="#/inbox" className="relative rounded-md px-2 py-1 text-sm hover:bg-panel-2" title="Doctor inbox">
      🔔{open > 0 && <span className="absolute -top-1 -right-1 grid h-5 min-w-5 place-items-center rounded-full bg-red-500 px-1 text-[11px] font-bold text-white">{open}</span>}
    </a>
  )
}
