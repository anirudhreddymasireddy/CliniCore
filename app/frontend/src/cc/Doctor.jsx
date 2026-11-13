import { useCallback, useEffect, useState } from 'react'
import { api, num, pct } from './api'
import { Btn, Counter, Loading, Panel, Shell, Toast, inputCls, useToast } from './ui'

function CaseRow({ c, active, onClick }) {
  return (
    <button onClick={onClick} className={`w-full min-w-0 rounded-lg border px-3 py-2 text-left ${active ? 'border-blue-400 bg-blue-500/15' : 'border-line hover:bg-panel-2'}`}>
      <div className="flex min-w-0 items-center gap-2">
        <b className="truncate">{c.source === 'live intake' ? '● New intake' : 'Visit'} {c.visit_id}</b>
        <span className="ml-auto shrink-0 rounded bg-bg px-1.5 text-xs text-muted">{c.age != null ? `${num(c.age)} y · ` : ''}{pct(c.confidence)}</span>
      </div>
      <div className="truncate text-sm text-muted">{c.complaint || 'no complaint recorded'}</div>
      {c.flags?.length > 0 && <div className="text-sm text-amber-300">⚠ unusual case</div>}
    </button>
  )
}

function Detail({ c, groups, onSaved, role, show }) {
  const [pick, setPick] = useState(c.top?.[0]?.group || '')
  const [detail, setDetail] = useState('')
  const [q, setQ] = useState('')
  const [openList, setOpenList] = useState(false)
  const [busy, setBusy] = useState(false)
  useEffect(() => { setPick(c.top?.[0]?.group || ''); setDetail(''); setQ('') }, [c.visit_id, c.top])
  const v = c.vitals || {}
  const save = async () => {
    setBusy(true)
    try {
      const r = await api('/api/ed/label', { visit_id: c.visit_id, label: pick, detail, role })
      show(r.model_was_right ? 'Saved – model agreed' : 'Saved – correction will teach the model'); onSaved()
    } catch (e) { show(e.message, 'err') }
    setBusy(false)
  }
  return (
    <Panel title={`${c.source === 'live intake' ? 'New intake' : 'ED visit'} ${c.visit_id}`} right={<span className="text-sm text-muted">{c.source}</span>}>
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <p><b>{num(c.age)} y, {c.sex}</b>{c.ambulance ? ' · arrived by ambulance' : ''}</p>
          <p className="mt-1"><b>Complaint:</b> {c.complaint || '–'}</p>
          {c.chronic && <p className="mt-1"><b>Known conditions:</b> {c.chronic}</p>}
          {c.history_note && <p className="mt-1"><b>Notes:</b> {c.history_note}</p>}
          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            {[['Temp', v.temp_c, '°C', 1], ['HR', v.hr, '/min'], ['RR', v.rr, '/min'], ['SpO₂', v.spo2, '%'], ['BP', v.sbp != null ? `${num(v.sbp)}/${num(v.dbp)}` : null, ''], ['Pain', v.pain, '/10']].map(([l, x, u, d]) => (
              <div key={l} className="rounded-lg bg-panel-2 p-2"><div className="text-sm text-muted">{l}</div><div className="text-xl font-bold">{typeof x === 'string' ? x : num(x, d || 0)}<span className="text-sm text-muted">{x != null ? u : ''}</span></div></div>
            ))}
          </div>
        </div>
        <div>
          <p className="mb-2"><b>Why it is here:</b> {c.why}</p>
          {c.flags?.length > 0 && <p className="mb-2 rounded-lg bg-amber-400/10 p-2 text-amber-200">⚠ {c.flags.join('; ')}</p>}
          <p className="mb-1 font-bold">Model's top guesses</p>
          {c.top.map((t) => (
            <button key={t.group} onClick={() => setPick(t.group)} className={`mb-1 flex w-full items-center justify-between rounded-lg border px-3 py-2 ${pick === t.group ? 'border-blue-400 bg-blue-500/20' : 'border-line hover:bg-panel-2'}`}>
              <span>{t.label}</span><b>{pct(t.prob)}</b>
            </button>
          ))}
          {c.plausible?.length > 0 && <p className="mt-1 text-sm text-muted">Also possible: {c.plausible.map((x) => x.label).join(' · ')}</p>}
          <div className="relative mt-3">
            <label className="text-sm font-bold" htmlFor="grp">Diagnosis group <span className="font-normal text-muted">(type to search all {groups.length})</span></label>
            <input id="grp" className={inputCls} autoComplete="off" value={openList ? q : (groups.find((g) => g.id === pick)?.label || '')}
                   onFocus={() => { setOpenList(true); setQ('') }} onBlur={() => setTimeout(() => setOpenList(false), 150)} onChange={(e) => setQ(e.target.value)}
                   placeholder="e.g. kidney, injury, breathing…" />
            {openList && (
              <div className="absolute z-20 mt-1 max-h-64 w-full overflow-auto rounded-lg border border-line bg-panel shadow-2xl">
                {groups.filter((g) => g.label.toLowerCase().includes(q.toLowerCase())).map((g) => (
                  <button key={g.id} onMouseDown={() => { setPick(g.id); setOpenList(false) }} className={`block w-full px-3 py-1.5 text-left text-sm hover:bg-panel-2 ${g.id === pick ? 'text-sky-300' : ''}`}>{g.label}</button>
                ))}
              </div>
            )}
          </div>
          <label className="mt-3 block text-sm font-bold" htmlFor="det">Specific diagnosis or note <span className="font-normal text-muted">(optional, your own words)</span></label>
          <textarea id="det" rows="2" maxLength="300" className={`${inputCls} text-sm`} value={detail} onChange={(e) => setDetail(e.target.value)}
                    placeholder="e.g. community-acquired pneumonia, right lower lobe" />
          <p className="mt-1 text-xs text-muted">The model learns the group; your text is saved with the case and shown to the patient after review.</p>
          {c.labelled ? <p className="mt-3 text-emerald-300">Already labelled: {groups.find((g) => g.id === c.labelled)?.label}</p>
            : <Btn kind="primary" className="mt-3 w-full" disabled={!pick || busy} onClick={save}>Confirm diagnosis group</Btn>}
        </div>
      </div>
    </Panel>
  )
}

export default function Doctor() {
  const [q, setQ] = useState(null)
  const [meta, setMeta] = useState(null)
  const [err, setErr] = useState(null)
  const [sel, setSel] = useState(null)
  const [tab, setTab] = useState('live')
  const [role, setRole] = useState('doctor')
  const [toast, show] = useToast()
  const load = useCallback(() => api('/api/ed/queue?limit=60').then(setQ).catch((e) => setErr(e.message)), [])
  useEffect(() => { load(); api('/api/ed/meta').then(setMeta).catch((e) => setErr(e.message)); const t = setInterval(load, 8000); return () => clearInterval(t) }, [load])
  if (!q || !meta) return <Shell><Loading error={err} /></Shell>
  const lists = { live: q.live, waiting: q.waiting, audit: q.audit }
  const list = lists[tab]
  const cur = sel && [...q.live, ...q.waiting, ...q.audit].find((c) => c.visit_id === sel)
  return (
    <Shell title="Doctor queue" subtitle="Cases the model was not sure about, unusual cases, and a 5% random re-check of confident answers."
           right={<><select className="w-auto rounded-md border border-line bg-bg px-2 py-1 text-sm" value={role} onChange={(e) => setRole(e.target.value)}><option>doctor</option><option>nurse practitioner</option></select><Counter c={q.counter} label="Doctor labels" /></>}>
      <div className="grid gap-5 lg:grid-cols-[380px_minmax(0,1fr)]">
        <Panel>
          <div className="mb-3 grid grid-cols-3 gap-1">
            {[['live', `New (${q.live_total})`], ['waiting', `Unsure (${q.waiting_total})`], ['audit', 'Re-check 5%']].map(([k, l]) => (
              <button key={k} onClick={() => setTab(k)} className={`rounded-lg px-2 py-1.5 text-sm font-bold ${tab === k ? 'bg-blue-500 text-white' : 'bg-panel-2'}`}>{l}</button>
            ))}
          </div>
          <div className="grid max-h-[70vh] gap-2 overflow-auto pr-1">
            {list.length ? list.map((c) => <CaseRow key={c.visit_id} c={c} active={c.visit_id === sel} onClick={() => setSel(c.visit_id)} />)
              : <p className="text-muted">{tab === 'live' ? 'No new intakes waiting. Add one on the ED intake page.' : 'Nothing here.'}</p>}
          </div>
        </Panel>
        <Toast {...(toast || {})} />
        <div>{cur ? <Detail c={cur} groups={meta.groups} role={role} show={show} onSaved={async () => { await load(); setSel(null) }} /> : <Panel><p className="text-muted">Pick a case on the left. Least certain cases are listed first.</p></Panel>}</div>
      </div>
    </Shell>
  )
}
