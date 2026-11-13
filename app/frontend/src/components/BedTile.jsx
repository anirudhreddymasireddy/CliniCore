import TierBadge from './TierBadge'
import { TIER } from '../lib/clinical'

function Num({ label, value, unit, color, dec = 0 }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs font-semibold text-muted">{label}</span>
      <span className="font-extrabold tabular-nums leading-none" style={{ color, fontSize: '1.55rem' }}>
        {value == null ? '–' : Number(value).toFixed(dec)}
        <span className="ml-0.5 text-[0.65rem] font-medium text-muted">{unit}</span>
      </span>
    </div>
  )
}

export default function BedTile({ bed, anonId, tier, vitals, risks, note, onOpen, highlight }) {
  const color = tier ? TIER[tier].color : '#3a4767'
  return (
    <div className={`flex flex-col gap-3 rounded-2xl border-2 bg-panel p-4 transition-colors duration-700 ${highlight ? 'anim-soft-pulse' : ''}`}
         style={{ borderColor: color, borderTopWidth: 10 }} data-testid={`bed-${bed}`} data-tier={tier || 'NOT_SCORED'}>
      <div className="flex items-baseline justify-between">
        <span className="text-2xl font-black">Bed {bed}</span>
        <span className="text-sm text-muted">{anonId}</span>
      </div>
      <div>{tier ? <TierBadge tier={tier} size="sm" /> : <span className="text-sm font-bold text-muted">NOT SCORED THIS HOUR</span>}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 rounded-xl bg-black/60 p-3">
        <Num label="HR" value={vitals.hr} unit="bpm" color="var(--color-mon-hr)" />
        <Num label="MAP" value={vitals.map} unit="mmHg" color="var(--color-mon-bp)" />
        <Num label="SpO₂" value={vitals.spo2} unit="%" color="var(--color-mon-spo2)" />
        <Num label="RR" value={vitals.rr} unit="/min" color="var(--color-mon-rr)" />
      </div>
      <div className="min-h-5 text-sm text-muted">{risks || note}</div>
      <button onClick={onOpen} className="min-h-12 rounded-xl border-2 border-line bg-panel-2 text-lg font-bold hover:bg-line">
        Open
      </button>
    </div>
  )
}
