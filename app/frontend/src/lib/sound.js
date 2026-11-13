// One soft chime (two gentle sine tones, low volume). Only used for NEW CRITICAL notifications; never repeats.
let ctx = null

export function unlockAudio() {
  const AC = window.AudioContext || window.webkitAudioContext
  if (!AC) return
  if (!ctx) ctx = new AC()
  if (ctx.state === 'suspended') ctx.resume()
}

export function playChime() {
  try {
    unlockAudio()
    if (!ctx) return
    const t = ctx.currentTime
    ;[[784, 0], [1046, 0.18]].forEach(([freq, delay]) => {
      const o = ctx.createOscillator(), g = ctx.createGain()
      o.type = 'sine'
      o.frequency.setValueAtTime(freq, t + delay)
      g.gain.setValueAtTime(0.0001, t + delay)
      g.gain.exponentialRampToValueAtTime(0.06, t + delay + 0.04)
      g.gain.exponentialRampToValueAtTime(0.0001, t + delay + 0.9)
      o.connect(g); g.connect(ctx.destination)
      o.start(t + delay); o.stop(t + delay + 0.95)
    })
    window.__cpChimes = (window.__cpChimes || 0) + 1
  } catch (e) {
    window.__cpChimeError = String(e)
  }
}
