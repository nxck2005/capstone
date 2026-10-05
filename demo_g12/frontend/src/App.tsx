import { lazy, Suspense, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { Check, CircleAlert, ImageOff, LoaderCircle, RotateCcw, X } from 'lucide-react'
import { api, isValidChart, isValidImages, isValidInfer, isValidMetadata } from './api'
import type { ChartResponse, ChartSeries, DemoImage, ImagesResponse, InferResponse, InferenceArm, Metadata } from './api'

const Chart = lazy(() => import('./Chart'))

const OUTAGES = ['decode_failure', 'codec_infeasibility', 'structural_infeasibility']
// Plain names for visitors; the two curves match the two live receivers.
const SHOWN: { id: string; name: string }[] = [{ id: 'learned', name: 'AI link' }, { id: 'classical_adaptive', name: 'Digital link' }]

function errorMessage(error: unknown) { return error instanceof Error ? error.message : 'Unexpected response from the local demo service.' }
function formatSnr(n: number) { return `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n)} dB` }
function formatPercent(n: number | null) { return n === null ? '—' : `${(n * 100).toFixed(1)}%` }
function signalWords(snr: number) {
  if (snr <= -5) return 'Very weak signal'
  if (snr <= 0) return 'Weak signal'
  if (snr <= 7) return 'Moderate signal'
  return 'Strong signal'
}
function pointAt(series: ChartSeries | undefined, snr: number) { return series?.points.find((point) => point.snr_db === snr) }

/** One short sentence, built only from the measured curves at this SNR. */
function narrative(series: ChartSeries[], snr: number): string | null {
  const ai = pointAt(series.find((s) => s.id === 'learned'), snr)
  const digital = pointAt(series.find((s) => s.id.startsWith('classical')), snr)
  if (!ai || !digital) return null
  const coverage = digital.coverage ?? null
  // Quote the measured value: at 1/24 the AI link is below half at the noisiest points.
  if (coverage === 0) return `Too noisy for the digital link: nothing gets through. The AI link still gets ${Math.round(ai.accuracy * 100)}% of pictures right.`
  if (coverage !== null && coverage < 1) return `Only ${Math.round(coverage * 100)}% of digital transmissions get through at this level.`
  if (digital.accuracy > ai.accuracy) return 'Both links get through, and the digital link is more accurate here.'
  return 'Both links get through, but the AI link is more accurate here.'
}

function Visual({ src, alt, fallback }: { src?: string | null; alt: string; fallback?: ReactNode }) {
  const [broken, setBroken] = useState(false)
  useEffect(() => setBroken(false), [src])
  return src && !broken ? <img src={src} alt={alt} onError={() => setBroken(true)} /> : <div className="visual-empty">{fallback ?? <><ImageOff size={22} /><span>Image unavailable</span></>}</div>
}

function Receiver({ kind, result, pending, truth }: { kind: 'classical' | 'learned'; result?: InferenceArm; pending: boolean; truth?: string | null }) {
  const isLearned = kind === 'learned'
  const outage = !isLearned && !!result && OUTAGES.includes(result.status)
  const delivered = result?.status === 'delivered'
  const unavailable = result?.status === 'unavailable'
  const verdict = delivered && truth && result?.predicted_label ? result.predicted_label === truth : null
  const empty = pending ? <><LoaderCircle className="spin" size={22} /><span>Sending…</span></>
    : outage ? <><ImageOff size={28} /><strong>No image received</strong></>
      : unavailable ? <><CircleAlert size={22} /><span>Unavailable</span></>
        : <><ImageOff size={22} /><span>Waiting</span></>

  return <article className={`receiver receiver-${kind}`} aria-label={isLearned ? 'Semantic DJSCC' : 'Classical digital'}>
    <header className="card-head">
      <h3>{isLearned ? 'AI link' : 'Digital link'}</h3>
      <span>{isLearned ? 'Deep JSCC' : 'JPEG 2000 + 5G'}</span>
    </header>
    <div className={`visual ${outage ? 'is-outage' : ''}`}>
      <Visual src={pending ? null : result?.image_url} alt={`${isLearned ? 'AI link' : 'Digital link'} output`} fallback={empty} />
    </div>
    <div className="result">
      <span className="result-kicker">{outage ? 'Fallback guess (no image received)' : 'Prediction'}</span>
      <div className="result-line">
        <strong className={outage ? 'is-fallback' : ''}>{pending ? '…' : result?.predicted_label || 'Not available'}</strong>
        {!pending && verdict !== null && <span className={`verdict ${verdict ? 'yes' : 'no'}`}>{verdict ? <Check size={14} /> : <X size={14} />}{verdict ? 'Correct' : 'Wrong'}</span>}
        {!pending && delivered && result?.confidence != null && <span className="confidence" aria-label="Confidence">{Math.round(result.confidence * 100)}% sure</span>}
        {!pending && outage && <span className="confidence" aria-label="Confidence">Confidence N/A</span>}
      </div>
      {outage && !pending && <p className="result-note">“{result.predicted_label}” is the fixed guess used whenever nothing arrives. It is not a prediction from this picture.</p>}
      {unavailable && !pending && <p className="result-note">{result.detail || 'This result is not available on this laptop. Nothing was substituted.'}</p>}
    </div>
  </article>
}

function Channel({ snr }: { snr: number | null }) {
  // Visual only: the grain gets heavier as the signal gets weaker.
  const level = snr === null ? 0.5 : Math.min(1, Math.max(0, (18 - snr) / 26))
  return <div className="channel" aria-hidden="true">
    <svg className="channel-noise" preserveAspectRatio="none">
      <filter id="channel-grain"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="7" /><feColorMatrix type="saturate" values="0" /></filter>
      <rect width="100%" height="100%" filter="url(#channel-grain)" style={{ opacity: 0.1 + level * 0.8 }} />
    </svg>
    <span className="channel-label">Noise</span>
    <span className="channel-arrow">→</span>
  </div>
}

export default function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [images, setImages] = useState<DemoImage[]>([])
  const [ratio, setRatio] = useState('')
  const [snr, setSnr] = useState<number | null>(null)
  const [imageId, setImageId] = useState('')
  const [chart, setChart] = useState<ChartResponse | null>(null)
  const [infer, setInfer] = useState<InferResponse | null>(null)
  const [bootstrapError, setBootstrapError] = useState('')
  const [imageError, setImageError] = useState('')
  const [chartError, setChartError] = useState('')
  const [inferError, setInferError] = useState('')
  const [pending, setPending] = useState(false)
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    api.metadata(controller.signal).then((data) => {
      if (controller.signal.aborted) return
      if (!isValidMetadata(data)) throw new Error('The API returned an invalid SNR grid or ratio list.')
      setMetadata(data)
      // Optional deep link, e.g. ?snr=18&ratio=r_1_24, limited to measured values.
      const params = new URLSearchParams(window.location.search)
      const linkedSnr = Number(params.get('snr'))
      const linkedRatio = params.get('ratio')
      setRatio(data.ratios.some((item) => item.id === linkedRatio) ? linkedRatio! : data.ratios[0].id)
      setSnr(params.has('snr') && data.snr_grid_db.includes(linkedSnr) ? linkedSnr : data.default_snr_db)
      setBootstrapError('')
    }).catch((error) => { if (!controller.signal.aborted) setBootstrapError(errorMessage(error)) })
    api.images(controller.signal).then((data: ImagesResponse) => {
      if (controller.signal.aborted) return
      if (!isValidImages(data)) throw new Error('The API must provide four unique bundled training images with local URLs.')
      setImages(data.images)
      setImageId(data.images[0]?.id ?? '')
      setImageError('')
    }).catch((error) => { if (!controller.signal.aborted) setImageError(errorMessage(error)) })
    return () => controller.abort()
  }, [retry])

  useEffect(() => {
    if (!metadata || !ratio) return
    const controller = new AbortController()
    setChart(null)
    setChartError('')
    api.chart(ratio, controller.signal).then((data) => {
      if (controller.signal.aborted) return
      if (!isValidChart(data, ratio, metadata.snr_grid_db)) throw new Error('The API returned an incomplete or unmeasured accuracy curve.')
      setChart(data)
    }).catch((error) => { if (!controller.signal.aborted) setChartError(errorMessage(error)) })
    return () => controller.abort()
  }, [metadata, ratio, retry])

  useEffect(() => {
    if (!metadata || !ratio || !imageId || snr === null) return
    const controller = new AbortController()
    setPending(true)
    setInfer(null)
    setInferError('')
    const timer = window.setTimeout(() => {
      api.infer(imageId, snr, ratio, controller.signal).then((data) => {
        if (controller.signal.aborted) return
        if (!isValidInfer(data, imageId, snr, ratio)) throw new Error('The API returned a result for another image, ratio, or SNR.')
        setInfer(data)
      }).catch((error) => { if (!controller.signal.aborted) setInferError(errorMessage(error)) })
        .finally(() => { if (!controller.signal.aborted) setPending(false) })
    }, 180)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [metadata, ratio, imageId, snr, retry])

  const grid = metadata?.snr_grid_db ?? []
  const selectedImage = images.find((image) => image.id === imageId)
  const index = snr === null ? 0 : Math.max(0, grid.indexOf(snr))
  const maxIndex = Math.max(0, grid.length - 1)
  const problem = bootstrapError || imageError || chartError || inferError
  const unavailable = [infer?.classical.status, infer?.learned.status].filter((status) => status === 'unavailable').length
  const service = problem ? 'Service error'
    : !metadata || !images.length ? 'Connecting…'
      : metadata.inference_available === false || unavailable === 2 ? 'Inference unavailable'
        : unavailable > 0 || !!metadata.inference_reason ? 'Partial inference' : 'Local API ready'
  const shown = SHOWN.flatMap(({ id, name }) => { const found = chart?.series.find((s) => s.id === id || (id.startsWith('classical') && s.id.startsWith('classical'))); return found ? [{ ...found, label: name }] : [] })
  const presets = [grid[0], -6, -4, grid[maxIndex]].filter((value, i, all) => value !== undefined && grid.includes(value) && all.indexOf(value) === i)
  const story = snr === null || !shown.length ? null : narrative(shown, snr)
  const stepSnr = (delta: number) => { if (snr !== null && grid.length) setSnr(grid[Math.min(maxIndex, Math.max(0, index + delta))]) }

  return <div className="app">
    <header className="topbar">
      <div className="brand"><span className="brand-mark" aria-hidden="true" />Signal Lab</div>
      <span className={`status ${service === 'Local API ready' ? 'is-ok' : ''}`} role="status" aria-label="Local service status">{service}</span>
    </header>

    <main className="page">
      <h1>Send a picture through a noisy channel</h1>

      {problem && <div className="alert" role="alert"><CircleAlert size={17} /><span>{problem}</span><button onClick={() => setRetry((n) => n + 1)}><RotateCcw size={14} /> Retry</button></div>}

      <section className="stage" aria-label="Interactive comparison">
        <article className="sender">
          <header className="card-head"><h3>Picture</h3><span>{selectedImage?.truth_label ?? ''}</span></header>
          <div className="visual"><Visual src={selectedImage?.thumbnail_url} alt={selectedImage ? `Input: ${selectedImage.label}` : 'Input image'} /></div>
          <div className="thumbs" role="group" aria-label="Choose a training image">
            {images.map((image) => <button type="button" key={image.id} aria-label={image.label} aria-pressed={imageId === image.id} className={imageId === image.id ? 'thumb is-chosen' : 'thumb'} onClick={() => setImageId(image.id)}>
              <Visual src={image.thumbnail_url} alt="" />
            </button>)}
          </div>
          <span className="sr-only">{selectedImage?.label}</span>
        </article>
        <Channel snr={snr} />
        <Receiver kind="learned" result={infer?.learned} pending={pending} truth={selectedImage?.truth_label} />
        <Receiver kind="classical" result={infer?.classical} pending={pending} truth={selectedImage?.truth_label} />
      </section>

      <section className="instrument" aria-label="Signal strength">
        <div className="controls">
          <span className="label">Signal strength</span>
          <strong className="snr-value">{snr === null ? '—' : <>{formatSnr(snr).replace(' dB', '')}<small>dB</small></>}</strong>
          <span className="snr-words">{snr === null ? '' : signalWords(snr)}</span>
          <div className="stepper">
            <button type="button" onClick={() => stepSnr(-1)} disabled={!metadata || index === 0} aria-label="Weaker signal">−</button>
            <input className="snr-range" type="range" aria-label="Channel SNR" min={0} max={maxIndex} step={1} value={index} disabled={!metadata}
              onChange={(event) => setSnr(grid[Number(event.target.value)])}
              style={{ '--progress': `${maxIndex ? (index / maxIndex) * 100 : 0}%` } as React.CSSProperties} />
            <button type="button" onClick={() => stepSnr(1)} disabled={!metadata || index === maxIndex} aria-label="Stronger signal">+</button>
          </div>
          <div className="presets">
            {presets.map((value) => <button type="button" key={value} className={snr === value ? 'is-on' : ''} onClick={() => setSnr(value)} aria-label={formatSnr(value)}>{formatSnr(value).replace(' dB', '')}</button>)}
          </div>
        </div>

        <div className="evidence">
          <span className="evidence-title">Final test · {(metadata?.n_images ?? 3925).toLocaleString('en-US')} pictures</span>
          <div className="evidence-head">
            <p className="story" aria-live="polite">{story ?? 'Loading…'}</p>
            {snr !== null && shown.length > 0 && <div className="legend">
              {shown.map((item) => <span key={item.id} className={`legend-item legend-${item.id.startsWith('classical') ? 'digital' : 'ai'}`}><i />{item.label}<b>{formatPercent(pointAt(item, snr)?.accuracy ?? null)}</b></span>)}
              <span className="legend-note">correct</span>
            </div>}
          </div>
          {shown.length && metadata && snr !== null
            ? <Suspense fallback={<div className="chart-empty">Loading…</div>}><Chart series={shown} snr={snr} grid={grid} onSelect={setSnr} /></Suspense>
            : <div className="chart-empty" role={chartError ? 'alert' : 'status'}>{chartError ? 'Chart unavailable' : 'Loading…'}</div>}
        </div>
      </section>
    </main>
  </div>
}
