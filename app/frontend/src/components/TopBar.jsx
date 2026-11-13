export default function TopBar({ title, subtitle, children }) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-line bg-panel px-5 py-3">
      <div>
        <h1 className="text-2xl font-black tracking-tight">{title}</h1>
        <p className="text-base text-muted">{subtitle}</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">{children}</div>
    </header>
  )
}

export function BarButton({ children, onClick, active, testid, label }) {
  return (
    <button onClick={onClick} data-testid={testid} aria-label={label}
            className={`min-h-12 rounded-xl border-2 px-4 text-lg font-bold ${active ? 'border-blue-400 bg-blue-500 text-white' : 'border-line bg-panel-2 hover:bg-line'}`}>
      {children}
    </button>
  )
}
