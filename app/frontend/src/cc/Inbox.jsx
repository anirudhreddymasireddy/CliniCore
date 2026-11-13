import { useEffect, useState } from 'react'
import { api } from './api'
import { Shell } from './ui'
import { STATUS, Sbar, URG, ack, useInbox, useViewer } from './notify'

export default function Inbox() {
  const [meta, setMeta] = useState(null)
  const [viewer, setViewer] = useViewer()
  const [show, setShow] = useState('open')
  const { items, reload } = useInbox(viewer, 4000)
  useEffect(() => { api('/api/notify/meta').then(setMeta).catch(() => {}) }, [])
  const list = items.filter((n) => show === 'all' || n.status === 'sent')
  const act = async (n, a) => { await ack(n, viewer, a); reload() }
  return (
    <Shell title="Doctor inbox" subtitle="Risk alerts sent by ICU nurses or by auto-notify. New alerts also pop up on any CliniCore page while this screen is signed in as a doctor.">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="text-sm text-muted">This screen is:</span>
        <button onClick={() => setViewer('nurse')} className={`rounded-md px-3 py-1.5 text-sm font-semibold ${viewer === 'nurse' ? 'bg-sky-500/20 text-sky-300' : 'bg-panel-2'}`}>Nurse (no pop-ups)</button>
        {(meta?.doctors || []).map((d) => (
          <button key={d.id} onClick={() => setViewer(d.id)} className={`rounded-md px-3 py-1.5 text-sm font-semibold ${viewer === d.id ? 'bg-sky-500/20 text-sky-300 ring-1 ring-sky-400' : 'bg-panel-2 hover:bg-line'}`}>{d.label}</button>
        ))}
      </div>
      {viewer === 'nurse' ? <p className="rounded-xl border border-line bg-panel p-6 text-muted">Choose which on-call doctor uses this screen. Tip for the demo: open a second browser window, sign it in as "ICU attending on call", and send an alert from the ICU risk board in the first window.</p> : (
        <section className="rounded-xl border border-line bg-panel p-4">
          <div className="mb-3 flex items-center gap-2">
            {[['open', `Open (${items.filter((n) => n.status === 'sent').length})`], ['all', `All (${items.length})`]].map(([k, l]) => <button key={k} onClick={() => setShow(k)} className={`rounded-md px-3 py-1 text-sm font-bold ${show === k ? 'bg-sky-500 text-white' : 'bg-panel-2'}`}>{l}</button>)}
          </div>
          <div className="grid gap-3">
            {list.map((n) => {
              const u = URG[n.urgency] || URG.routine, s = STATUS[n.status] || [n.status, '']
              return (
                <article key={n.notify_id} className="rounded-lg border border-line bg-panel-2/50 p-3" style={{ borderLeft: `4px solid ${u.ring}` }}>
                  <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
                    <span className={`rounded px-2 py-0.5 text-xs font-extrabold ${u.cls}`}>{u.label}</span>
                    <b>Bed {n.bed} · #{n.stay_id}</b><span className="text-muted">{n.risk_level || n.level} {Math.round(n.criticality)}/100</span>
                    <span className={`ml-auto text-xs ${s[1]}`}>{s[0]}{n.ack_ts ? ` · ${String(n.ack_ts).slice(11, 16)}` : ''}</span>
                  </div>
                  <Sbar text={n.message} />
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted">
                    <span>{n.auto ? 'auto-notify' : `sent by ${n.sent_by}`} · {n.ts} UTC</span>
                    <a href={`#/icu?stay=${n.stay_id}`} className="text-sky-300 hover:underline">open patient</a>
                    {n.status === 'sent' && <span className="ml-auto flex gap-2">
                      <button onClick={() => act(n, 'on_my_way')} className="rounded-md bg-emerald-500 px-3 py-1 text-sm font-bold text-black">On my way</button>
                      <button onClick={() => act(n, 'acknowledged')} className="rounded-md bg-panel-2 px-3 py-1 text-sm font-bold">Acknowledge</button>
                      <button onClick={() => act(n, 'declined')} className="rounded-md px-3 py-1 text-sm text-red-300 hover:bg-panel-2">Decline</button>
                    </span>}
                  </div>
                </article>
              )
            })}
            {!list.length && <p className="text-sm text-muted">No {show === 'open' ? 'open ' : ''}alerts for this doctor.</p>}
          </div>
        </section>
      )}
    </Shell>
  )
}
