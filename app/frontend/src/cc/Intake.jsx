import { useEffect, useMemo, useState } from 'react'
import { api, pct } from './api'
import { Btn, Loading, Panel, Shell, inputCls } from './ui'

const VITALS = [
  ['temp_c', 'Temperature', '°C', 0.1], ['hr', 'Heart rate', '/min', 1], ['rr', 'Breathing rate', '/min', 1],
  ['spo2', 'Oxygen saturation', '%', 1], ['sbp', 'Blood pressure systolic', 'mmHg', 1], ['dbp', 'Blood pressure diastolic', 'mmHg', 1],
  ['pain', 'Pain (0–10)', '', 1],
]
const EMPTY = { age: '', sex: 'female', ambulance: false, temp_c: '', hr: '', rr: '', spo2: '', sbp: '', dbp: '', pain: '', acuity: '', reasons: [], chronic: [], history_note: '' }
const EXAMPLES = {
  'Fever + cough, 67 y': { age: 67, sex: 'male', ambulance: true, temp_c: 38.9, hr: 118, rr: 26, spo2: 91, sbp: 98, dbp: 60, pain: 3, reasons: [14400, 10100], chronic: ['COPD'] },
  'Abdominal pain, 34 y': { age: 34, sex: 'female', temp_c: 37.1, hr: 88, rr: 16, spo2: 99, sbp: 124, dbp: 78, pain: 7, reasons: [15451] },
  'Ankle injury, 25 y': { age: 25, sex: 'male', temp_c: 36.8, hr: 76, rr: 14, spo2: 99, sbp: 128, dbp: 80, pain: 5, reasons: [55450] },
}

function ReasonPicker({ reasons, value, onChange }) {
  const [q, setQ] = useState('')
  const hits = useMemo(() => {
    const s = q.trim().toLowerCase()
    return (s ? reasons.filter((r) => r.label.toLowerCase().includes(s)) : reasons).slice(0, 8)
  }, [q, reasons])
  const label = (c) => reasons.find((r) => r.code === c)?.label || `code ${c}`
  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-2">
        {value.map((c, i) => (
          <span key={c} className="flex items-center gap-2 rounded-lg bg-blue-500/20 px-2 py-1">
            <b>{i === 0 ? 'Main:' : `${i + 1}.`}</b> {label(c)}
            <button type="button" aria-label="remove" className="text-muted hover:text-text" onClick={() => onChange(value.filter((x) => x !== c))}>✕</button>
          </span>
        ))}
        {!value.length && <span className="text-muted">No reason chosen yet (at least one is required)</span>}
      </div>
      {value.length < 3 && (
        <>
          <input className={inputCls} placeholder="Type a complaint, e.g. chest pain, cough, fever…" value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="mt-1 grid gap-1">
            {hits.map((r) => (
              <button type="button" key={r.code} disabled={value.includes(r.code)} onClick={() => { onChange([...value, r.code]); setQ('') }}
                      className="rounded-lg px-2 py-1 text-left hover:bg-panel-2 disabled:opacity-40">{r.label}</button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function Result({ res }) {
  const c = res.clinician
  const p = res.patient
  return (
    <div className="grid gap-4 anim-fade-up">
      <Panel title="What the patient sees">
        <div className={`rounded-xl p-4 ${p.kind === 'answer' ? 'bg-emerald-500/15' : 'bg-blue-500/15'}`}>
          <div className="text-2xl font-black">{p.headline}</div>
          <p className="mt-2 text-muted">{p.detail}</p>
        </div>
        <p className="mt-2 text-sm text-muted">Visit number {res.visit_id}. The patient never sees a guess when the system is unsure.</p>
      </Panel>
      <Panel title="What the clinician sees" right={<span className={`rounded-md px-2 py-0.5 text-sm font-bold ${c.route === 'PATIENT_VIEW' ? 'bg-emerald-500 text-black' : 'bg-blue-500 text-white'}`}>{c.route === 'PATIENT_VIEW' ? 'answered automatically' : 'sent to doctor queue'}</span>}>
        <p className="mb-3"><b>Why:</b> {c.route_reason}</p>
        <div className="mb-3 grid gap-2">
          {c.top.map((t, i) => (
            <div key={t.group} className="flex items-center gap-3">
              <span className="w-6 text-muted">{i + 1}.</span>
              <span className="w-80">{t.label}</span>
              <div className="h-3 flex-1 rounded bg-line"><div className="h-3 rounded bg-blue-400" style={{ width: pct(t.prob) }} /></div>
              <span className="w-12 text-right font-bold">{pct(t.prob)}</span>
            </div>
          ))}
        </div>
        {c.plausible_set?.length > 0 && <p className="mb-2"><b>Also possible:</b> {c.plausible_set.map((x) => x.label).join(' · ')}</p>}
        {c.flags?.length > 0 && <div className="rounded-lg border border-amber-400/60 bg-amber-400/10 p-2"><b>Check:</b> {c.flags.join('; ')}</div>}
        {res.audit && <p className="mt-2 text-amber-300">Picked for the 5% random doctor re-check.</p>}
      </Panel>
    </div>
  )
}

export default function Intake() {
  const [meta, setMeta] = useState(null)
  const [err, setErr] = useState(null)
  const [f, setF] = useState(EMPTY)
  const [res, setRes] = useState(null)
  const [problems, setProblems] = useState(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => { api('/api/ed/meta').then(setMeta).catch((e) => setErr(e.message)) }, [])
  if (!meta) return <Shell><Loading error={err} /></Shell>
  const set = (k, v) => setF((s) => ({ ...s, [k]: v }))

  const submit = async (e) => {
    e.preventDefault(); setBusy(true); setProblems(null)
    const body = { ...f }
    for (const k of ['age', 'temp_c', 'hr', 'rr', 'spo2', 'sbp', 'dbp', 'pain', 'acuity']) body[k] = body[k] === '' ? null : Number(body[k])
    try { setRes(await api('/api/ed/intake', body)) } catch (x) { setProblems(x.message); setRes(null) }
    setBusy(false)
  }

  return (
    <Shell title="ED intake" subtitle="Enter what the patient tells you and the triage vital signs. The assistant suggests a diagnosis group or hands the case to a doctor."
           right={<span className="text-sm text-muted">model v{meta.model_version}</span>}>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <form onSubmit={submit} className="grid gap-4">
          <Panel title="Patient" right={<div className="flex flex-wrap gap-1">{Object.entries(EXAMPLES).map(([k, v]) => <button type="button" key={k} className="rounded-md bg-panel-2 px-2 py-1 text-sm hover:bg-line" onClick={() => { setF({ ...EMPTY, ...v }); setRes(null) }}>{k}</button>)}</div>}>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <label>Age<input className={inputCls} type="number" min="18" max="110" required value={f.age} onChange={(e) => set('age', e.target.value)} /></label>
              <label>Sex<select className={inputCls} value={f.sex} onChange={(e) => set('sex', e.target.value)}><option value="female">female</option><option value="male">male</option><option value="other">other</option></select></label>
              <label>Triage level<select className={inputCls} value={f.acuity} onChange={(e) => set('acuity', e.target.value)}><option value="">not triaged</option>{[1, 2, 3, 4, 5].map((x) => <option key={x} value={x}>{x} {x === 1 ? '(immediate)' : x === 5 ? '(non-urgent)' : ''}</option>)}</select></label>
              <label className="flex items-end gap-2 pb-2"><input type="checkbox" className="h-5 w-5" checked={f.ambulance} onChange={(e) => set('ambulance', e.target.checked)} />Arrived by ambulance</label>
            </div>
          </Panel>
          <Panel title="Reason for visit (up to 3)"><ReasonPicker reasons={meta.reasons} value={f.reasons} onChange={(v) => set('reasons', v)} /></Panel>
          <Panel title="Vital signs (leave empty if not measured)">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {VITALS.map(([k, l, u, st]) => <label key={k} className="text-sm">{l} {u && <span className="text-muted">({u})</span>}<input className={inputCls} type="number" step={st} value={f[k]} onChange={(e) => set(k, e.target.value)} /></label>)}
            </div>
          </Panel>
          <Panel title="History">
            <div className="mb-3 flex flex-wrap gap-2">
              {meta.chronic.map((c) => {
                const on = f.chronic.includes(c.id)
                return <button type="button" key={c.id} onClick={() => set('chronic', on ? f.chronic.filter((x) => x !== c.id) : [...f.chronic, c.id])}
                               className={`rounded-lg border px-2 py-1 ${on ? 'border-blue-400 bg-blue-500/30' : 'border-line'}`}>{c.label}</button>
              })}
            </div>
            <textarea className={inputCls} rows="2" maxLength="500" placeholder="Other history (shown to the doctor, not used by the model)" value={f.history_note} onChange={(e) => set('history_note', e.target.value)} />
          </Panel>
          {problems && <div className="rounded-xl border border-red-400 bg-red-500/15 p-3"><b>Please fix:</b> {problems}</div>}
          <div className="flex gap-3"><Btn kind="primary" type="submit" disabled={busy}>{busy ? 'Assessing…' : 'Assess'}</Btn><Btn onClick={() => { setF(EMPTY); setRes(null); setProblems(null) }}>Clear</Btn></div>
        </form>
        <div>{res ? <Result res={res} /> : <Panel title="Result"><p className="text-muted">Fill in the form and press Assess. Confident answers (≥ {pct(meta.model?.tau)} and a clear gap to the 2nd guess, nothing unusual) go to the patient as a group, never a diagnosis. Everything else goes to a doctor.</p></Panel>}</div>
      </div>
    </Shell>
  )
}
