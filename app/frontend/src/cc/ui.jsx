import { useEffect, useState } from 'react'
import { DoctorPopups, InboxBell } from './notify'

export const LEVEL = {
  Critical: { color: '#F65353', icon: '▲▲' }, High: { color: '#FFA511', icon: '▲' },
  Watch: { color: '#E8D44D', icon: '●' }, Low: { color: '#7EC636', icon: '✓' },
}

const NAV = [
  ['#/', 'Home'], ['#/intake', 'ED intake'], ['#/doctor', 'Doctor queue'], ['#/icu', 'ICU risk'], ['#/inbox', 'Doctor inbox'],
]

export function Shell({ title, subtitle, children, right }) {
  const here = (window.location.hash || '#/').split('?')[0]
  // CliniCore pages use a compact desk-screen size; the wall monitor (presentation mode) keeps its large type.
  useEffect(() => { const h = document.documentElement; const old = h.style.fontSize; h.style.fontSize = '14.5px'; return () => { h.style.fontSize = old } }, [])
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-30 flex items-center gap-5 border-b border-line bg-panel/95 px-5 py-2 backdrop-blur">
        <a href="#/" className="flex shrink-0 items-center gap-2 text-lg font-black tracking-tight">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-sky-400 to-blue-600 text-sm text-white">C+</span>CliniCore
        </a>
        <nav className="flex gap-0.5 overflow-x-auto">
          {NAV.slice(1).map(([h, l]) => (
            <a key={h} href={h} className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-semibold transition ${here === h ? 'bg-blue-500/20 text-sky-300' : 'text-muted hover:bg-panel-2 hover:text-text'}`}>{l}</a>
          ))}
        </nav>
        <div className="ml-auto flex shrink-0 items-center gap-4">{right}<InboxBell /></div>
      </header>
      <div className="border-b border-amber-300/20 bg-amber-300/10 px-5 py-1 text-xs text-amber-200">Clinical decision support – not a diagnosis. Staff always decide.</div>
      <main className="mx-auto w-full max-w-[1600px] flex-1 p-5">
        {title && <div className="mb-4"><h1 className="text-2xl font-black">{title}</h1>{subtitle && <p className="text-sm text-muted">{subtitle}</p>}</div>}
        {children}
      </main>
      <DoctorPopups />
    </div>
  )
}

export function Panel({ title, children, className = '', right }) {
  return (
    <section className={`rounded-xl border border-line bg-panel p-4 ${className}`}>
      {(title || right) && <div className="mb-3 flex items-center justify-between gap-3"><h2 className="text-base font-bold">{title}</h2>{right}</div>}
      {children}
    </section>
  )
}

export function Badge({ level, children }) {
  const l = LEVEL[level] || { color: '#a6b2c8', icon: '' }
  return (
    <span className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-sm font-extrabold whitespace-nowrap" style={{ background: l.color, color: '#0b0f19' }}>
      <span aria-hidden>{l.icon}</span>{children || level}
    </span>
  )
}

export function Counter({ c, label }) {
  if (!c) return null
  const w = Math.min(100, (100 * c.n) / c.threshold)
  return (
    <div className="w-44" title="When this reaches the threshold the model retrains automatically and only goes live if it passes the safety check.">
      <div className="flex justify-between text-xs text-muted"><span>{label}</span><span><b className="text-text">{c.n}</b>/{c.threshold}</span></div>
      <div className="mt-1 h-1.5 rounded bg-line"><div className="h-1.5 rounded bg-sky-400 transition-all" style={{ width: `${w}%` }} /></div>
    </div>
  )
}

export function Toast({ msg, kind = 'ok' }) {
  if (!msg) return null
  return <div className={`fixed bottom-6 left-1/2 z-50 -translate-x-1/2 rounded-lg px-4 py-2.5 text-sm font-bold shadow-2xl anim-fade-up ${kind === 'err' ? 'bg-red-500 text-white' : 'bg-emerald-400 text-black'}`}>{msg}</div>
}

export function useToast() {
  const [t, setT] = useState(null)
  useEffect(() => { if (!t) return; const id = setTimeout(() => setT(null), 3200); return () => clearTimeout(id) }, [t])
  return [t, (msg, kind = 'ok') => setT({ msg, kind })]
}

export function Btn({ children, onClick, kind = 'default', disabled, type = 'button', className = '' }) {
  const k = {
    default: 'border-line bg-panel-2 hover:bg-line', primary: 'border-blue-400 bg-blue-500 text-white hover:bg-blue-600',
    danger: 'border-red-400 bg-red-500/90 text-white hover:bg-red-600', ok: 'border-emerald-400 bg-emerald-500 text-black hover:bg-emerald-400',
  }[kind]
  return <button type={type} disabled={disabled} onClick={onClick} className={`min-h-9 rounded-lg border px-3 text-sm font-bold transition disabled:opacity-40 ${k} ${className}`}>{children}</button>
}

export const inputCls = 'w-full rounded-lg border border-line bg-bg px-3 py-2 text-text placeholder:text-muted focus:border-blue-400'

export function Loading({ error }) {
  return <p className="p-6 text-lg text-muted">{error ? `Could not load: ${error}` : 'Loading…'}</p>
}
