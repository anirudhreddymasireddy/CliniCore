import { useEffect, useState } from 'react'
import { api, pct } from './api'
import { Shell } from './ui'

const TILES = [
  ['#/intake', 'ED intake', 'Stage 1', 'Enter history, complaint and vitals. Confident answers go to the patient as a diagnosis group; everything else goes to a doctor.'],
  ['#/doctor', 'Doctor queue', 'Stage 1 · human in the loop', 'Uncertain and unusual cases plus a 5% random re-check. Every label teaches the model; it retrains at 50.'],
  ['#/icu', 'ICU risk', 'Stage 3', 'Sepsis, respiratory failure, kidney injury and shock ranked by criticality, with the vitals and labs behind each warning. Wall view for the central station, with replay.'],
  ['#/inbox', 'Doctor inbox', 'Stage 3 · escalation', 'Nurses send a risk to the right on-call doctor as an SBAR message; it pops up on the doctor\'s screen. Optional auto-alert for rising Critical patients.'],
]

export default function Home() {
  const [h, setH] = useState(null)
  useEffect(() => { api('/api/clinicore/health').then(setH).catch(() => setH({ error: true })) }, [])
  return (
    <Shell>
      <div className="mx-auto max-w-5xl py-6 text-center">
        <h1 className="text-5xl font-black">CliniCore</h1>
        <p className="mt-2 text-xl text-muted">From the ED front door to the ICU bed: predictions that show their evidence, flag their uncertainty and keep clinicians in charge.</p>
      </div>
      <div className="mx-auto grid max-w-6xl gap-4 md:grid-cols-2">
        {TILES.map(([href, t, tag, d]) => (
          <a key={href} href={href} className="rounded-2xl border border-line bg-panel p-5 hover:border-blue-400 hover:bg-panel-2">
            <div className="text-sm font-bold text-blue-300">{tag}</div><div className="text-2xl font-black">{t}</div><p className="mt-2 text-muted">{d}</p>
          </a>
        ))}
      </div>
      <p className="mt-6 text-center text-sm text-muted">
        {h?.error ? 'Backend not reachable.' : h ? `${h.icu_patients} ICU patients · ED model ${h.stage1_model}` : ''}
      </p>
    </Shell>
  )
}
