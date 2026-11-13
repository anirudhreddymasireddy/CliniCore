import { TIER, HORIZON } from '../lib/clinical'

// Colour is never the only signal: every badge carries an icon and a text label.
export default function TierBadge({ tier, predicted = false, size = 'md' }) {
  const t = TIER[tier] || TIER.NORMAL
  const sz = { sm: 'text-sm px-2 py-0.5', md: 'text-base px-3 py-1', lg: 'text-xl px-4 py-1.5' }[size]
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-lg font-extrabold tracking-wide whitespace-nowrap ${sz}`}
          style={{ background: t.color, color: '#0b0f19' }}>
      <span aria-hidden>{t.icon}</span>{t.label}{predicted && tier !== 'NORMAL' ? ` · ${HORIZON}` : ''}
    </span>
  )
}
