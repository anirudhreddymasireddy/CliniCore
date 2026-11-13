export async function getJSON(path) {
  const r = await fetch(path)
  if (!r.ok) {
    let detail = r.statusText
    try { detail = (await r.json()).detail || detail } catch { /* not JSON */ }
    throw new Error(`${r.status}: ${detail}`)
  }
  return r.json()
}

export async function postFeedback(body) {
  const r = await fetch('/api/feedback', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`feedback not logged (${r.status})`)
  return r.json()
}
