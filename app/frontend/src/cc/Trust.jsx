import { useEffect, useState } from 'react'
import { api, num, pct } from './api'
import { Counter, Loading, Panel, Shell } from './ui'

const Stat = ({ label, value, hint }) => (
  <div className="rounded-xl bg-panel-2 p-3"><div className="text-sm text-muted">{label}</div><div className="text-3xl font-black">{value}</div>{hint && <div className="text-sm text-muted">{hint}</div>}</div>
)

function Table({ rows, cols }) {
  if (!rows?.length) return <p className="text-muted">Not run yet.</p>
  return (
    <div className="overflow-auto"><table className="w-full text-left text-sm">
      <thead className="text-muted"><tr>{cols.map(([k, l]) => <th key={k} className="pr-3 pb-1">{l}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i} className="border-t border-line">{cols.map(([k, , f]) => <td key={k} className="py-1 pr-3">{f ? f(r[k], r) : r[k]}</td>)}</tr>)}</tbody>
    </table></div>
  )
}

export default function Trust() {
  const [t, setT] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => { api('/api/trust').then(setT).catch((e) => setErr(e.message)) }, [])
  if (!t) return <Shell><Loading error={err} /></Shell>
  const s1 = t.stage1, best = s1 ? s1[s1.chosen_on_validation] : null, te = best?.test_2022
  return (
    <Shell title="Model & trust" subtitle="Every number here is measured on data the model never saw in training.">
      <div className="grid gap-5">
        <Panel title="How CliniCore earns trust">
          <ul className="grid list-disc gap-1 pl-6">
            <li>Patients only see an answer when the model is confident (≥ {pct(t.rules.tau)} and a clear gap to the 2nd guess) <b>and</b> the case looks like the training data. Otherwise: "A doctor will review your case".</li>
            <li>Patients see a diagnosis <b>group</b>, never a diagnosis. {pct(t.rules.audit_rate)} of confident answers are re-checked by a doctor at random.</li>
            <li>Retraining starts automatically after {t.rules.retrain_after} doctor labels (ED) or {t.rules.retrain_after} false alarms with a reason (ICU), and goes live only if it passes a safety check on frozen data.</li>
            <li>ICU warnings list vitals, labs and treatments only; staff confirm or reject each one.</li>
          </ul>
          <div className="mt-3 flex flex-wrap gap-6"><Counter c={t.counters.stage1} label="ED doctor labels" /><Counter c={t.counters.stage3} label="ICU false alarms" /></div>
        </Panel>

        <Panel title="Stage 1 · ED diagnosis group (real data: NHAMCS, trained 2018–2021, tested 2022)">
          {te ? (
            <>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
                <Stat label="Accuracy (17 groups)" value={pct(te.accuracy)} hint={`majority-class baseline ${pct(s1.majority_class_accuracy_2022)}`} />
                <Stat label="Right group in top 3" value={pct(te.top3_accuracy)} />
                <Stat label="Answered automatically" value={pct(te.auto_share)} />
                <Stat label="…of those correct" value={pct(te.auto_accuracy)} />
                <Stat label="Calibration error (ECE)" value={num(te.ece, 3)} hint="0 = perfect" />
                <Stat label="90% plausible set" value={pct(s1.conformal?.coverage_2022)} hint={`contains the true group; avg ${num(s1.conformal?.mean_set_size, 1)} groups`} />
              </div>
              <h3 className="mt-4 mb-1 font-bold">The confidence cut-off is a policy choice (2022 test year)</h3>
              <Table rows={s1.policy_sweep} cols={[['tau', 'cut-off', (v) => pct(v)], ['auto_share_test_2022', 'answered automatically', (v) => pct(v)], ['auto_accuracy_test_2022', 'correct when answered', (v) => pct(v)]]} />
              <h3 className="mt-4 mb-1 font-bold">Where it is strong and weak (2022)</h3>
              <Table rows={t.stage1_groups} cols={[['group', 'group'], ['mean', 'accuracy', (v) => pct(v)], ['size', 'visits']]} />
            </>
          ) : <p className="text-muted">Run nhamcs_stage1.py to see real-data results.</p>}
        </Panel>

        <Panel title="Stage 3 · sepsis early warning, model comparison (real data: PhysioNet 2019, trained hospital A, tested hospital B)">
          <Table rows={t.sepsis_benchmark} cols={[['model', 'model'], ['auprc', 'AUPRC', (v) => num(v, 3)], ['auroc', 'AUROC', (v) => num(v, 3)], ['utility', 'utility (approx.)', (v) => num(v, 3)],
            ['septic_patients_alerted', 'septic patients alerted', (v) => pct(v)], ['false_alert_hours_per_24h', 'false-alert h / 24 h', (v) => num(v, 2)]]} />
          <p className="mt-2 text-sm text-muted">Decision rule: keep the simplest model (LightGBM) unless another beats it on both AUPRC and utility with 95% confidence. Utility is our re-implementation of the PhysioNet score; the split is not the official hidden test set.</p>
        </Panel>

        <Panel title="Stage 3 · four ICU risks (synthetic test data – pipeline check only)">
          <Table rows={(t.stage3 || []).filter((r) => r.set === 'B_shifted_hospital')} cols={[['risk', 'risk'], ['standard', 'standard behind the tiers'], ['events', 'events'], ['auroc', 'AUROC', (v) => num(v, 3)], ['auprc', 'AUPRC', (v) => num(v, 3)]]} />
        </Panel>

        <Panel title="Retraining log">
          <Table rows={t.registry} cols={[['ts', 'time'], ['stage', 'stage'], ['trigger', 'trigger'], ['n_new', 'new examples'], ['promoted', 'went live', (v) => (String(v) === 'true' || v === 1 || v === true ? 'yes' : 'no – kept old model')], ['version_after', 'version']]} />
        </Panel>
      </div>
    </Shell>
  )
}
