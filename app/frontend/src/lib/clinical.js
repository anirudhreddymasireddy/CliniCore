export const TIER = {
  CRITICAL: { color: '#F65353', icon: '⚠', label: 'CRITICAL' },
  MEDIUM: { color: '#FFA511', icon: '▲', label: 'MEDIUM' },
  NORMAL: { color: '#7EC636', icon: '●', label: 'NORMAL' },
}
export const RISK = {
  shock: { label: 'Shock', key: 'p_shock', color: '#ff7b8a' },
  resp: { label: 'Respiratory', key: 'p_resp', color: '#b69cff' },
  renal: { label: 'Renal / metabolic', key: 'p_renal', color: '#4cc3ff' },
}
export const HORIZON = 'PREDICTED (next 6 h)'

export function riskTypes(components) {
  if (!components) return []
  return components.split(',').map((c) => (c === 'NEWS2>=7' ? 'NEWS2 ≥ 7 safety net' : RISK[c]?.label || c))
}

// ALERT_TEXT: "<TIER> – <comp> risk p%: d1, d2, d3; recent ... ordered | <comp> ..."
export function topReasons(text, k = 3) {
  if (!text || !text.includes('–')) return []
  const out = []
  text.split('–').slice(1).join('–').split(' | ').forEach((part) => {
    const drivers = part.includes(':') ? part.split(':').slice(1).join(':') : part
    drivers.split(';')[0].split(', ').map((d) => d.trim()).filter(Boolean).forEach((d) => out.push(d))
  })
  return out.slice(0, k)
}

export function orderedTests(text) {
  const out = []
  for (const m of (text || '').matchAll(/recent ([^|;]+?) ordered/g)) {
    m[1].split(',').map((t) => t.trim()).filter(Boolean).forEach((t) => { if (!out.includes(t)) out.push(t) })
  }
  return out
}

// Current score vs the deployed alert threshold (setting A: shock -> CRITICAL, resp / renal -> MEDIUM).
export function componentLevel(meta, comp, p) {
  if (p == null) return 'NORMAL'
  if (p >= meta.thresholds.crit[comp]) return comp === 'shock' ? 'CRITICAL' : 'MEDIUM'
  return 'NORMAL'
}

export function trendArrow(hours, key) {
  if (!hours.length) return { arrow: '→', note: 'no trend yet' }
  const last = hours[hours.length - 1]
  const past = [...hours].reverse().find((h) => h.hour_index <= last.hour_index - 3) || hours[0]
  const diff = 100 * ((last[key] ?? 0) - (past[key] ?? 0))
  if (diff >= 1) return { arrow: '↑', note: `+${diff.toFixed(0)} pts in 3 h` }
  if (diff <= -1) return { arrow: '↓', note: `${diff.toFixed(0)} pts in 3 h` }
  return { arrow: '→', note: 'stable over 3 h' }
}

export const LABS = [
  ['lactate', 'Lactate', 'mmol/L', 1, 'LACTATE'], ['creatinine', 'Creatinine', 'mg/dL', 2, 'CREATININE'],
  ['potassium', 'Potassium', 'mmol/L', 1, 'K'], ['sodium', 'Sodium', 'mmol/L', 0, 'NA'],
  ['bicarbonate', 'Bicarbonate', 'mmol/L', 0, 'HCO3'], ['glucose', 'Glucose', 'mg/dL', 0, 'GLU'],
  ['ph_arterial', 'Arterial pH', '', 2, 'PH'], ['pao2', 'PaO₂', 'mmHg', 0, 'PAO2'], ['paco2', 'PaCO₂', 'mmHg', 0, 'PACO2'],
  ['wbc', 'WBC', 'K/µL', 1, 'WBC'], ['hemoglobin', 'Hemoglobin', 'g/dL', 1, 'HB'], ['platelets', 'Platelets', 'K/µL', 0, 'PLT'],
  ['troponin', 'Troponin', '× ULN', 1, 'TROPONIN'], ['bnp', 'BNP', 'pg/mL', 0, 'BNP'], ['bun', 'BUN', 'mg/dL', 0, 'BUN'],
  ['magnesium', 'Magnesium', 'mg/dL', 1, 'MG'], ['bilirubin_total', 'Bilirubin', 'mg/dL', 1, 'BILI'], ['albumin', 'Albumin', 'g/dL', 1, 'ALB'],
]

export function labFlag(ranges, rkey, value, dec) {
  const r = ranges[rkey]
  if (!r || value == null) return null
  const v = Number(value.toFixed(dec))
  if ((r.cap_high != null && v >= r.cap_high) || (r.cap_low != null && v <= r.cap_low)) return 'artifact'
  if (r.low != null && v < r.low) return 'low'
  if (r.high != null && v > r.high) return 'high'
  return null
}

export function sinceText(hoursAgo, minutesAgo) {
  if (minutesAgo != null) return minutesAgo < 1 ? 'just now' : `${Math.round(minutesAgo)} min ago`
  return hoursAgo <= 0 ? 'just now' : `${hoursAgo} h ago`
}
