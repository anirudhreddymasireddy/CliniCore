export async function api(path, body) {
  const r = await fetch(path, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  let data = null
  try { data = await r.json() } catch { /* empty */ }
  if (!r.ok) {
    const d = data?.detail
    const msg = Array.isArray(d) ? d.map((x) => (typeof x === 'string' ? x : `${(x.loc || []).slice(-1)[0]}: ${x.msg}`)).join(' · ') : (d || r.statusText)
    throw new Error(msg)
  }
  return data
}

export const pct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`)
export const num = (v, d = 0) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d))
