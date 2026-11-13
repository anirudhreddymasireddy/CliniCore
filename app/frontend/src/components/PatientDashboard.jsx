import { useEffect, useMemo, useState } from 'react'
import TierBadge from './TierBadge'
import MonitorVitals from './MonitorVitals'
import Waveforms from './Waveforms'
import RiskChart from './RiskChart'
import { getJSON } from '../lib/api'
import { HORIZON, LABS, RISK, TIER, componentLevel, labFlag, orderedTests, riskTypes, topReasons, trendArrow } from '../lib/clinical'

export const FOOTER = 'Clinical decision support – not a diagnosis. Feedback is reviewed by the clinical team; the model is not retrained automatically.'
const FALSE_ALARM_REASONS = ['Artifact', 'Already known & treated', 'Not clinically relevant']
const SEX = { M: 'male', F: 'female' }

function Panel({ title, children, className = '', testid }) {
  return (
    <section className={`rounded-2xl border border-line bg-panel p-4 ${className}`} data-testid={testid}>
      <h3 className="mb-3 text-xl font-extrabold">{title}</h3>
      {children}
    </section>
  )
}

function RiskCard({ comp, hours, meta }) {
  const k = RISK[comp]
  const last = hours[hours.length - 1]
  const p = last?.[k.key]
  const level = componentLevel(meta, comp, p)
  const { arrow, note } = trendArrow(hours, k.key)
  return (
    <div className="min-w-40 rounded-xl bg-panel-2 px-4 py-3" style={{ borderLeft: `8px solid ${TIER[level].color}` }} data-testid={`risk-${comp}`}>
      <div className="text-base font-bold text-muted">{k.label}</div>
      <div className="flex items-baseline gap-2">
        <span className="text-4xl font-black tabular-nums">{p == null ? '–' : `${(100 * p).toFixed(0)}%`}</span>
        <span className="text-3xl font-black" aria-label={note}>{arrow}</span>
      </div>
      <TierBadge tier={level} size="sm" />
      <div className="mt-1 text-sm text-muted">{note}</div>
    </div>
  )
}

function LabPanel({ hours, meta }) {
  const last = hours[hours.length - 1]
  const window = hours.filter((h) => h.hour_index >= last.hour_index - 6)
  const newResults = LABS.filter(([key]) => window.some((h, i) => i > 0 && h[key] != null && h[key] !== window[i - 1][key])).map((l) => l[1])
  if (last.troponin_ordered_flag === 1 && !newResults.includes('Troponin')) newResults.push('Troponin')
  const ordered = orderedTests(last.alert_text)
  const cells = LABS.filter(([key]) => last[key] != null)
  return (
    <div data-testid="lab-panel">
      <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-2">
        {cells.map(([key, label, unit, dec, rkey]) => {
          const flag = labFlag(meta.ranges, rkey, last[key], dec)
          const color = { high: '#F65353', low: '#FFA511', artifact: '#a6b2c8' }[flag] || '#eef2fa'
          const mark = { high: '↑ HIGH', low: '↓ LOW', artifact: 'likely artifact' }[flag]
          return (
            <div key={key} className="rounded-xl bg-panel-2 px-3 py-2" style={{ border: `2px solid ${flag === 'high' || flag === 'low' ? color : 'transparent'}` }}
                 data-flag={flag || 'normal'}>
              <div className="text-sm font-semibold text-muted">{label}</div>
              <div className="text-2xl font-extrabold tabular-nums" style={{ color }}>{Number(last[key]).toFixed(dec)}
                <span className="ml-1 text-xs font-medium text-muted">{unit}</span></div>
              {mark && <div className="text-xs font-bold" style={{ color }}>{mark}</div>}
            </div>
          )
        })}
      </div>
      <p className="mt-3 text-base"><b>Recently ordered (model signal):</b> {ordered.length ? ordered.join(', ') : 'none in the current alert'}</p>
      <p className="text-base"><b>New results in the last 6 h:</b> {newResults.length ? newResults.join(', ') : 'none'}</p>
      <p className="mt-1 text-xs text-muted">Latest value per test (carried forward up to 24 h); new results are detected from value changes.</p>
    </div>
  )
}

function HistoryPanel({ hours, actions }) {
  const alerts = hours.filter((h) => h.alert_fired === 1 && TIER[h.alert_level]).reverse()
  return (
    <div data-testid="history-panel">
      <div className="mb-1 text-base font-extrabold">Previous alerts</div>
      <ul className="mb-3 space-y-1.5">
        {alerts.length ? alerts.map((a) => (
          <li key={a.hour_index} className="flex items-center gap-2 text-base">
            <TierBadge tier={a.alert_level} size="sm" /> ICU hour {a.hour_index} · {riskTypes(a.trigger_components).join(', ')}
          </li>
        )) : <li className="text-muted">No alerts so far.</li>}
      </ul>
      <div className="mb-1 text-base font-extrabold">Nurse actions</div>
      <ul className="space-y-1">
        {actions.length ? [...actions].reverse().map((a, i) => (
          <li key={i} className="text-base">{a.logged_at_utc.slice(11, 16)} UTC · {a.action}{a.reason ? ` (${a.reason})` : ''}</li>
        )) : <li className="text-muted">No actions logged.</li>}
      </ul>
    </div>
  )
}

export default function PatientDashboard({ patient, meta, nowHour, anchorLabel, ward, bed, onBack, onAction, refreshKey, vitalsWindowH = 12, actionsSince = null }) {
  const hours = useMemo(() => patient.hours.filter((h) => h.hour_index <= nowHour), [patient, nowHour])
  const [actions, setActions] = useState([])
  const [reason, setReason] = useState('')
  useEffect(() => {
    getJSON(`/api/feedback?anon_id=${encodeURIComponent(patient.anon_id)}&ward=${ward}`).then(setActions).catch(() => setActions([]))
  }, [patient.anon_id, ward, refreshKey])
  if (!hours.length) {
    return <div className="rounded-2xl border border-line bg-panel p-6 text-xl">Bed {bed}: no scored hour up to ICU hour {nowHour}.
      <button onClick={onBack} className="ml-4 min-h-12 rounded-xl border-2 border-line px-4 font-bold">◀ Back to ward</button></div>
  }
  const last = hours[hours.length - 1]
  const tier = last.tier
  const anchorHour = last.hour_index
  const anchorMin = (anchorHour + 1) * 60
  const alerts = hours.filter((h) => h.alert_fired === 1 && TIER[h.alert_level])
    .map((h) => ({ t: (h.hour_index + 1) * 60, color: TIER[h.alert_level].color }))
  const hrNow = [...patient.vitals.points].reverse().find((p) => p.t <= anchorMin && p.hr != null)?.hr
  const reasons = topReasons(last.alert_text)
  const lastAlert = [...hours].reverse().find((h) => h.alert_fired === 1)
  const ctx = { ward, bed, anon_id: patient.anon_id, source: patient.source, replay_hour: nowHour,
                alert_hour: lastAlert ? lastAlert.hour_index : null, tier, components: last.trigger_components || '' }

  return (
    <div className="anim-fade-up overflow-hidden rounded-3xl border-2 bg-bg" style={{ borderColor: TIER[tier].color }} data-testid="patient-dashboard">
      <div className="flex items-center justify-between border-b border-line bg-panel px-5 py-3">
        <div className="flex gap-2" aria-hidden>
          {['#F65353', '#FFA511', '#7EC636'].map((c) => <span key={c} className="h-4 w-4 rounded-full" style={{ background: c }} />)}
        </div>
        <button onClick={onBack} className="min-h-12 rounded-xl border-2 border-line bg-panel-2 px-5 text-lg font-bold hover:bg-line">◀ Back to ward</button>
      </div>

      <div className="flex flex-wrap items-start justify-between gap-5 border-b border-line px-6 py-5">
        <div>
          <div className="text-base font-bold text-muted">Patient Dashboard</div>
          <h2 className="text-4xl font-black" data-testid="patient-title">Bed {bed} · {patient.anon_id}</h2>
          <div className="mt-1 text-lg">{patient.age != null ? `${patient.age} y` : 'age n/a'} · {SEX[patient.sex] || 'sex n/a'} · {patient.unit_type || 'ICU'} · {patient.source === 'EICU' ? 'eICU' : patient.source}</div>
          <div className="text-base text-muted">ICU admission reason: {patient.admission_reason || 'not available in the model data'}</div>
          <div className="mt-3"><TierBadge tier={tier} predicted size="lg" /></div>
        </div>
        <div className="rounded-2xl border border-line bg-panel p-4" data-testid="risk-score">
          <div className="mb-2 text-lg font-extrabold">Risk score · next 6 h</div>
          <div className="flex flex-wrap gap-3">{Object.keys(RISK).map((c) => <RiskCard key={c} comp={c} hours={hours} meta={meta} />)}</div>
        </div>
      </div>

      <div className="grid gap-4 p-5 xl:grid-cols-[1.25fr_1fr]">
        <div className="flex min-w-0 flex-col gap-4">
          <Panel title="Vitals" testid="vitals-panel">
            <div className="flex flex-col gap-2">
              <Waveforms hr={hrNow} />
              <MonitorVitals points={patient.vitals.points} anchorMin={anchorMin} alerts={alerts} anchorLabel={anchorLabel} windowH={vitalsWindowH}
                             resolution={patient.vitals.resolution} artifactTimes={patient.vitals.artifact_times || []} />
            </div>
          </Panel>
          <Panel title="Lab"><LabPanel hours={hours} meta={meta} /></Panel>
          <Panel title="History"><HistoryPanel hours={hours} actions={actionsSince ? actions.filter((a) => a.logged_at_utc >= actionsSince) : actions} /></Panel>
        </div>
        <Panel title="Prediction" className="min-w-0 self-start" testid="prediction-panel">
          <div className="mb-3 text-lg font-extrabold text-sky-300">Prediction horizon: next 6 h</div>
          {tier === 'NORMAL' ? (
            <p className="text-lg">No active prediction: all three risks are below their alert thresholds.</p>
          ) : (
            <div>
              <div className="flex flex-wrap items-center gap-3">
                <TierBadge tier={tier} predicted />
                <span className="text-xl font-extrabold">{riskTypes(last.trigger_components).join(', ')}</span>
              </div>
              <div className="mt-4 text-base font-bold text-muted">Top drivers</div>
              <ol className="mt-1 list-decimal space-y-1 pl-7 text-xl leading-snug" data-testid="drivers">
                {(reasons.length ? reasons : ['pattern-based risk (no single value out of range)']).map((r) => <li key={r}>{r}</li>)}
              </ol>
            </div>
          )}
          <div className="mt-5 text-base font-bold text-muted">Risk curves of the three models · last 12 h</div>
          <RiskChart hours={hours} anchorHour={anchorHour} thresholds={meta.thresholds.crit} anchorLabel={anchorLabel}
                     alerting={(last.trigger_components || '').split(',')} />
          {last.alert_text && <p className="mt-3 text-sm text-muted">Alert text: {last.alert_text}</p>}
        </Panel>
      </div>

      <div className="flex flex-wrap items-center justify-end gap-3 border-t border-line px-5 py-4" data-testid="actions">
        <select value={reason} onChange={(e) => setReason(e.target.value)} aria-label="False alarm reason"
                className="min-h-12 rounded-xl border-2 border-line bg-panel-2 px-3 text-lg font-semibold">
          <option value="">False alarm – reason (required)</option>
          {FALSE_ALARM_REASONS.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
        <button disabled={!reason} onClick={() => { onAction('false alarm', { ...ctx, reason }); setReason('') }}
                className="min-h-14 rounded-xl border-2 border-line bg-panel-2 px-6 text-xl font-bold enabled:hover:bg-line disabled:opacity-40">
          ✕ False alarm
        </button>
        <button onClick={() => onAction('acknowledge', ctx)} className="min-h-14 rounded-xl bg-blue-500 px-6 text-xl font-bold text-white hover:bg-blue-400">
          ✓ Acknowledge
        </button>
        <button onClick={() => onAction('call doctor', ctx)} className="min-h-14 rounded-xl px-7 text-xl font-black" style={{ background: '#F65353', color: '#0b0f19' }}>
          ☎ Call doctor
        </button>
      </div>
      <footer className="border-t border-line bg-panel px-5 py-3 text-base font-bold" data-testid="footer">{FOOTER}</footer>
    </div>
  )
}
