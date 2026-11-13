export default function Toast({ toast }) {
  if (!toast) return null
  return (
    <div role="status" aria-live="polite" data-testid="toast"
         className="anim-fade-up fixed bottom-8 left-1/2 z-50 -translate-x-1/2 rounded-2xl border-2 border-normal bg-panel-2 px-7 py-4 text-2xl font-extrabold shadow-2xl shadow-black/50">
      ✓ {toast}
    </div>
  )
}
