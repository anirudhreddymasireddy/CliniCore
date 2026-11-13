import { useEffect, useState } from 'react'
import Presentation from './modes/Presentation'
import Replay from './modes/Replay'
import Home from './cc/Home'
import Intake from './cc/Intake'
import Doctor from './cc/Doctor'
import Icu from './cc/Icu'
import Trust from './cc/Trust'
import Inbox from './cc/Inbox'

function useHash() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const on = () => setHash(window.location.hash)
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return hash
}

export default function App() {
  const hash = useHash()
  if (hash.startsWith('#/present')) return <Presentation />
  if (hash.startsWith('#/replay')) return <Replay />
  if (hash.startsWith('#/intake')) return <Intake />
  if (hash.startsWith('#/doctor')) return <Doctor />
  if (hash.startsWith('#/icu')) return <Icu />
  if (hash.startsWith('#/trust')) return <Trust />
  if (hash.startsWith('#/inbox')) return <Inbox />
  return <Home />
}
