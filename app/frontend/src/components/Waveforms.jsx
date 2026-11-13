import { useEffect, useRef } from 'react'

// Simulated ECG and pleth sweep whose rhythm matches the patient's current heart rate.
// Clearly labelled: this is NOT patient data.
function ecg(phase) {
  const g = (mu, s, a) => a * Math.exp(-((phase - mu) ** 2) / (2 * s * s))
  return g(0.18, 0.025, 0.12) - g(0.36, 0.008, 0.12) + g(0.38, 0.01, 1.0) - g(0.405, 0.01, 0.25) + g(0.62, 0.05, 0.28)
}
function pleth(phase) {
  const p = (phase + 0.85) % 1
  return Math.exp(-((p - 0.22) ** 2) / 0.012) * 0.9 + Math.exp(-((p - 0.45) ** 2) / 0.02) * 0.35
}

export default function Waveforms({ hr }) {
  const ref = useRef(null)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const c = canvas.getContext('2d')
    let raf, last = performance.now(), x = 0, beat = 0
    const prev = { e: null, p: null }
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const draw = (now) => {
      const w = canvas.width, h = canvas.height, rate = Math.max(30, Math.min(200, hr || 70))
      const dt = Math.min(0.05, (now - last) / 1000); last = now
      const steps = Math.max(1, Math.round(dt * 220))
      for (let i = 0; i < steps; i++) {
        beat = (beat + (rate / 60) / 220) % 1
        x = (x + w / 1320) % w
        c.fillStyle = '#000'; c.fillRect(x, 0, 10, h)
        const ye = h * 0.32 - ecg(beat) * h * 0.26, yp = h * 0.92 - pleth(beat) * h * 0.3
        c.lineWidth = 2
        if (prev.e && x > prev.e[0]) {
          c.strokeStyle = '#3dff7a'; c.beginPath(); c.moveTo(...prev.e); c.lineTo(x, ye); c.stroke()
          c.strokeStyle = '#33e1ff'; c.beginPath(); c.moveTo(...prev.p); c.lineTo(x, yp); c.stroke()
        }
        prev.e = [x, ye]; prev.p = [x, yp]
      }
      if (!reduce) raf = requestAnimationFrame(draw)
    }
    c.fillStyle = '#000'; c.fillRect(0, 0, canvas.width, canvas.height)
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [hr])
  return (
    <div className="relative overflow-hidden rounded-xl bg-black ring-1 ring-white/10" data-testid="waveforms">
      <canvas ref={ref} width={1200} height={150} className="block h-[110px] w-full" aria-hidden />
      <span className="absolute right-2 top-1.5 rounded bg-black/70 px-2 py-0.5 text-xs font-bold text-white/80">
        Illustrative waveform – not patient data · rate set to current HR {hr ? Math.round(hr) : '–'} bpm
      </span>
    </div>
  )
}
