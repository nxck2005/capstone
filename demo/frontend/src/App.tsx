import { lazy, Suspense, useEffect, useState } from 'react'
import { Activity, ArrowDownRight, ArrowRight, Check, ChevronDown, CircleAlert, Cpu, Database, Image as ImageIcon, LockKeyhole, MoveRight, Radio, RotateCcw, SlidersHorizontal, Sparkles, Waves } from 'lucide-react'
import { api, isValidChart, isValidImages, isValidInfer, isValidMetadata } from './api'
import type { ChartResponse, DemoImage, ImagesResponse, InferResponse, InferenceArm, Metadata } from './api'

const Chart = lazy(() => import('./Chart'))

function errorMessage(error: unknown) { return error instanceof Error ? error.message : 'Unexpected response from the local demo service.' }
function formatSnr(n: number) { return `${n > 0 ? '+' : ''}${n} dB` }
function formatPercent(n: number | null) { return n === null ? '—' : `${(n * 100).toFixed(1)}%` }

function Visual({ src, alt, fallback }: { src?: string | null; alt: string; fallback?: string }) {
  const [broken, setBroken] = useState(false)
  useEffect(() => setBroken(false), [src])
  return src && !broken ? <img src={src} alt={alt} onError={() => setBroken(true)} /> : <div className="visual-placeholder"><ImageIcon size={25} strokeWidth={1.5} /><span>{fallback || 'Image unavailable'}</span></div>
}

function ArmCard({ kind, result, pending, hasRequest }: { kind: 'classical' | 'learned'; result?: InferenceArm; pending: boolean; hasRequest: boolean }) {
  const isLearned = kind === 'learned'
  const label = isLearned ? 'Semantic DJSCC' : 'Classical digital'
  const detail = isLearned ? 'TASK-AWARE · END TO END' : 'JPEG 2000 · LDPC · ADAPTIVE'
  return <article className={`arm-card ${isLearned ? 'arm-learned' : 'arm-classical'}`} aria-label={label}>
    <div className="arm-heading">
      <span className={`arm-icon ${isLearned ? 'arm-icon-coral' : 'arm-icon-teal'}`}>{isLearned ? <Sparkles size={21} /> : <Radio size={21} />}</span>
      <div><div className="arm-title">{label}</div><div className="arm-detail">{detail}</div></div>
      <span className={`arm-status ${pending ? 'status-pending' : result?.status === 'delivered' ? 'status-good' : 'status-muted'}`}>
        {pending ? 'RUNNING' : !hasRequest ? 'WAITING' : result?.status === 'delivered' ? (isLearned ? 'OUTPUT READY' : 'DELIVERED') : result ? result.status.replaceAll('_', ' ').toUpperCase() : 'UNAVAILABLE'}
      </span>
    </div>
    <div className="arm-visual"><Visual src={result?.image_url} alt={`${label} output`} fallback={pending ? 'Processing image…' : result?.status === 'unavailable' ? 'Inference unavailable' : result && !isLearned ? 'No decoded image at this SNR' : result ? 'Reconstruction unavailable' : 'Awaiting local inference'} /></div>
    <div className="arm-prediction">
      <div><span className="field-label">PREDICTED LABEL</span><strong>{pending ? 'Computing…' : result?.predicted_label || 'Not available'}</strong></div>
      <div className="confidence"><span className="field-label">CONFIDENCE</span><strong>{pending ? '—' : formatPercent(result?.confidence ?? null)}</strong></div>
    </div>
    {result?.detail && !pending && <p className="arm-note">{result.detail}</p>}
    {result && !pending && ['decode_failure', 'codec_infeasibility', 'structural_infeasibility'].includes(result.status) && <p className="arm-note">An outage prediction may still be returned by the frozen fallback policy. A fallback is not a delivered transmission.</p>}
    {result?.status === 'unavailable' && !pending && !result.detail && <p className="arm-note">Image-level inference is not available from this local service. No prediction was substituted.</p>}
  </article>
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
  const [galleryOpen, setGalleryOpen] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    api.metadata(controller.signal).then((data) => {
      if (controller.signal.aborted) return
      if (!isValidMetadata(data)) throw new Error('The API returned an invalid SNR grid or ratio list.')
      setMetadata(data)
      setRatio(data.ratios[0].id)
      setSnr(data.default_snr_db)
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

  const selectedImage = images.find((image) => image.id === imageId)
  const index = snr === null ? 0 : metadata?.snr_grid_db.indexOf(snr) ?? 0
  const maxIndex = (metadata?.snr_grid_db.length ?? 1) - 1
  const ratioInfo = metadata?.ratios.find((item) => item.id === ratio)
  const hasServiceError = !!(bootstrapError || imageError || chartError || inferError)
  const unavailable = [infer?.classical.status, infer?.learned.status].filter((status) => status === 'unavailable').length
  const serviceLabel = hasServiceError ? 'SERVICE ERROR'
    : !metadata || !images.length ? 'CONNECTING TO LOCAL API'
      : metadata.inference_available === false || unavailable === 2 ? 'INFERENCE UNAVAILABLE'
        : unavailable > 0 || !!metadata.inference_reason ? 'PARTIAL INFERENCE' : 'LOCAL API READY'
  const indicatorClass = hasServiceError ? 'indicator-error'
    : serviceLabel === 'INFERENCE UNAVAILABLE' || serviceLabel === 'PARTIAL INFERENCE' ? 'indicator-limited'
      : serviceLabel === 'LOCAL API READY' ? 'indicator-ready' : 'indicator-loading'

  return <div className="app-shell">
    <aside className="rail" aria-label="Exhibit identity"><div className="rail-logo"><Waves size={26} strokeWidth={2.5} /></div><div className="rail-wordmark">SIGNAL / LAB</div><div className="rail-bottom"><span className="rail-line" /><span>01 — 04</span></div></aside>
    <div className="main-shell">
      <header className="topbar"><div className="topbar-location"><span className="topbar-dot" /> INTERACTIVE EXHIBIT <span className="topbar-divider">/</span> SEMANTIC COMMUNICATION</div><div className="topbar-right" role="status" aria-label="Local service status"><span className={`online-indicator ${indicatorClass}`} /> {serviceLabel} <span className="topbar-divider">/</span> VALIDATION CHART</div></header>
      <main>
        <section className="hero" aria-labelledby="hero-title">
          <div className="hero-content"><div className="hero-kicker"><span>THE EXPERIMENT</span><span className="hero-kicker-line" /></div><h1 id="hero-title">What if we sent<br /><em>meaning</em> instead of pixels?</h1><p>Put two communication systems on the same image and channel. Move the noise level. Watch what survives.</p><a className="hero-link" href="#workspace">EXPLORE THE EXPERIMENT <ArrowDownRight size={18} /></a></div>
          <div className="hero-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit orbit-three" /><div className="hero-core"><Waves size={64} strokeWidth={1.2} /></div><span className="hero-art-label art-label-one">SOURCE / IMAGE</span><span className="hero-art-label art-label-two">CHANNEL / NOISE</span><span className="hero-art-label art-label-three">RECEIVER / TASK</span><span className="art-node node-one" /><span className="art-node node-two" /><span className="art-node node-three" /></div>
          <div className="hero-corner">01 <span>/</span> 04</div>
        </section>

        <section id="workspace" className="workspace" aria-label="Interactive comparison">
          <div className="section-heading"><div><span className="eyebrow">01 / THE TEST BENCH</span><h2>Set the conditions<span className="accent-dot">.</span></h2><p>Live demo: bundled training images. Published chart: separate validation aggregates.</p></div><span className="section-badge"><LockKeyhole size={14} /> TEST SPLIT SEALED</span></div>
          {bootstrapError && <div className="alert" role="alert"><CircleAlert size={18} /><span>{bootstrapError}</span><button onClick={() => setRetry((n) => n + 1)}><RotateCcw size={14} /> Retry</button></div>}
          <div className="controls">
            <div className="control-intro"><span className="control-icon"><SlidersHorizontal size={20} /></span><div><strong>Experiment controls</strong><small>Adjust a variable to update both pipelines</small></div></div>
            <div className="control-fields">
              <label className="ratio-field"><span className="field-label">BANDWIDTH BUDGET</span><span className="select-wrap"><select aria-label="Bandwidth budget" value={ratio} disabled={!metadata} onChange={(event) => setRatio(event.target.value)}><option value="" disabled>Loading…</option>{metadata?.ratios.map((item) => <option key={item.id} value={item.id}>{item.label} · {item.channel_uses.toLocaleString()} uses</option>)}</select><ChevronDown size={16} /></span></label>
              <div className="snr-field"><div className="snr-head"><label htmlFor="snr-control" className="field-label">CHANNEL SNR</label><strong>{snr === null ? '—' : formatSnr(snr)}</strong></div><input id="snr-control" type="range" min={0} max={maxIndex} step={1} value={Math.max(0, index)} disabled={!metadata} onChange={(event) => setSnr(metadata!.snr_grid_db[Number(event.target.value)])} style={{ '--range-progress': `${maxIndex > 0 ? (Math.max(0, index) / maxIndex) * 100 : 0}%` } as React.CSSProperties} /><div className="snr-bounds"><span>{metadata ? formatSnr(metadata.snr_grid_db[0]) : '—'}</span><span>MORE NOISE <MoveRight size={12} className="bound-arrow" /> LESS NOISE</span><span>{metadata ? formatSnr(metadata.snr_grid_db[maxIndex]) : '—'}</span></div></div>
            </div>
          </div>

          <div className="section-heading compare-heading"><div><span className="eyebrow">02 / SIDE BY SIDE</span><h2>See what gets through<span className="accent-dot">.</span></h2></div><span className="comparison-context"><Activity size={15} /> {snr === null ? 'AWAITING API' : `AT ${formatSnr(snr).toUpperCase()}`} <span>·</span> {ratioInfo?.label ?? '—'}</span></div>
          <div className="input-strip"><div className="input-strip-title"><span className="input-icon"><ImageIcon size={18} /></span><div><span className="field-label">INPUT IMAGE · TRAIN SPLIT</span><strong>{selectedImage?.label ?? (imageError || 'Awaiting local image catalog')}</strong></div></div><div className="input-preview"><Visual src={selectedImage?.thumbnail_url} alt={selectedImage ? `Input: ${selectedImage.label}` : 'Input image'} /></div><div className="input-strip-right"><span>{selectedImage?.truth_label ? `TRAIN LABEL  /  ${selectedImage.truth_label}` : 'BUNDLED TRAIN IMAGE'}</span><button type="button" disabled={!images.length} onClick={() => setGalleryOpen((v) => !v)} aria-expanded={galleryOpen} aria-controls="image-gallery">CHOOSE IMAGE <ArrowRight size={15} /></button></div></div>
          {galleryOpen && <div id="image-gallery" className="gallery" aria-label="Choose a training image">{images.map((image) => <button type="button" key={image.id} onClick={() => { setImageId(image.id); setGalleryOpen(false) }} className={imageId === image.id ? 'chosen' : ''}><div className="gallery-image"><Visual src={image.thumbnail_url} alt="" /></div><span>{image.label}</span>{imageId === image.id && <Check size={16} />}</button>)}</div>}
          <div className="comparison-grid"><ArmCard kind="classical" result={infer?.classical} pending={pending} hasRequest={!!metadata && !!imageId} /><ArmCard kind="learned" result={infer?.learned} pending={pending} hasRequest={!!metadata && !!imageId} /></div>
          {(inferError || metadata?.inference_reason || infer?.reason) && <div className="alert subtle-alert" role="status"><CircleAlert size={17} />{inferError || infer?.reason || metadata?.inference_reason}<button onClick={() => setRetry((n) => n + 1)}><RotateCcw size={14} /> Retry</button></div>}
          <p className="comparison-footnote"><LockKeyhole size={14} /> Training-image outputs come from the local inference API; confidence is an uncalibrated softmax score. The chart uses separate frozen validation aggregates. No prediction is invented if inference is unavailable.</p>

          <div className="section-heading chart-heading"><div><span className="eyebrow">03 / THE BIGGER PICTURE</span><h2>Beyond one image<span className="accent-dot">.</span></h2><p>Frozen, aggregate validation accuracy at each measured channel condition.</p></div><span className="section-badge muted-badge"><Database size={14} /> PUBLISHED EVIDENCE</span></div>
          {chart && metadata && snr !== null ? <Suspense fallback={<div className="chart-empty">Loading chart module…</div>}><Chart series={chart.series} snr={snr} grid={metadata.snr_grid_db} /></Suspense> : <div className="chart-empty" role={chartError ? 'alert' : 'status'}><Activity size={25} /><strong>{chartError ? 'Unable to load the measured curves' : 'Loading published measurements'}</strong><span>{chartError || 'Connecting to the local evidence service…'}</span>{chartError && <button onClick={() => setRetry((n) => n + 1)}>Retry</button>}</div>}

          <div className="closing-grid"><article className="insight-card"><span className="eyebrow">04 / THE TAKEAWAY</span><h3>Different strengths.<br /><em>Different failure modes.</em></h3><p>At the 1/6 budget, the learned task head keeps predicting as noise rises; the digital image can instead hit a decoding cliff. Once LDPC succeeds and JPEG 2000 pixels arrive, the artifact-aware classical scorer can use more image detail and becomes more accurate. The 10% outage-policy score is not successful delivery.</p><div className="insight-link"><Waves size={17} /> A SYSTEM-LEVEL COMPARISON, NOT A CHANNEL-CODE-ONLY CLAIM</div></article><article className="method-card"><div className="method-header"><Cpu size={20} /><span>READ THE EVIDENCE</span></div><div className="method-row"><span>01</span><p>Chart: validation only · one frozen seed cell · 1,000 images per point. Test sealed.</p></div><div className="method-row"><span>02</span><p>Learned uses its own task head; classical uses the artifact-finetuned classifier.</p></div><div className="method-row"><span>03</span><p>Demo images come from the train split. Live inference is separate from published validation evidence.</p></div><div className="method-source">SOURCE <span>{metadata?.evidence_label || 'W10 published validation closeout'}</span></div></article></div>
        </section>
      </main>
      <footer className="footer"><div className="footer-brand"><Waves size={20} /> SIGNAL / LAB <span>—</span> SEMANTIC COMMUNICATION</div><span>RESEARCH EXHIBIT <span className="footer-dot">·</span> SIMULATED AWGN <span className="footer-dot">·</span> TEST SEALED</span></footer>
    </div>
  </div>
}
