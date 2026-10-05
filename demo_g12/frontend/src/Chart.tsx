import { useLayoutEffect, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import type { ChartSeries } from './api'

// Same colorblind-safe assignments used in the W10 results figures and the paper.
const palette: Record<string, string> = {
  learned: '#0072B2',
  classical: '#D55E00',
  classical_adaptive: '#D55E00',
  learned_randomized: '#009E73',
  learned_snr_randomised: '#009E73',
  er9_digital: '#CC79A7',
  classical_fixed_mcs: '#E69F00',
}
export const seriesColor = (item: ChartSeries, index: number) =>
  item.color || palette[item.id] || ['#0072B2', '#D55E00', '#009E73', '#CC79A7'][index]
const percent = (value: number) => `${(value * 100).toFixed(1)}%`
const signed = (n: number) => `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n)}`

function useWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(fallback)
  useLayoutEffect(() => {
    const node = ref.current
    if (!node) return
    const measure = () => { const w = node.getBoundingClientRect().width; if (w > 0) setWidth(Math.max(320, Math.floor(w))) }
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

interface ChartProps {
  series: ChartSeries[]
  snr: number
  grid: number[]
  onSelect?: (snr: number) => void
}

export default function Chart({ series, snr, grid, onSelect }: ChartProps) {
  const active = series.slice(0, 4)
  const [wrapRef, width] = useWidth<HTMLDivElement>(820)
  const [hover, setHover] = useState<number | null>(null)
  const dragging = useRef(false)

  const height = 190
  const pad = { left: 44, right: 14, top: 14, bottom: 30 }
  const plotW = width - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom
  // Each measured SNR gets equal width. The grid is 1 dB apart at the noisy end and 2–3 dB apart
  // at the clean end, so the noisy end is wider. Every tick carries its real value, and a break
  // mark shows where the step size changes.
  const step = (plotW) / Math.max(1, grid.length - 1)
  const x = (value: number) => pad.left + grid.indexOf(value) * step
  const y = (value: number) => pad.top + (1 - value) * plotH
  const nearest = (clientX: number, rect: DOMRect) => {
    const index = Math.round((clientX - rect.left - pad.left) / step)
    return grid[Math.min(grid.length - 1, Math.max(0, index))]
  }
  const baseStep = grid.length > 1 ? grid[1] - grid[0] : 1
  const breakAt = grid.findIndex((value, i) => i > 0 && value - grid[i - 1] > baseStep)
  const pick = (event: ReactPointerEvent<SVGSVGElement>) => nearest(event.clientX, event.currentTarget.getBoundingClientRect())

  const classical = active.find((item) => item.id.startsWith('classical'))
  const coverage = classical?.points.map((point) => point.coverage ?? null) ?? []
  // Shade only the run of measured points, from the noisy end, where no digital packet arrived.
  let lastZero = -1
  while (coverage[lastZero + 1] === 0) lastZero += 1
  const description = `Accuracy versus measured SNR; selected ${snr} dB. ${active.map((item) => `${item.label}: ${percent(item.points.find((p) => p.snr_db === snr)!.accuracy)}`).join('; ')}`

  return <div className="chart" ref={wrapRef}>
    <svg
      className="chart-svg"
      width={width}
      height={height}
      role="img"
      aria-label={description}
      onPointerDown={(event) => { if (!onSelect) return; dragging.current = true; event.currentTarget.setPointerCapture?.(event.pointerId); onSelect(pick(event)) }}
      onPointerMove={(event) => { const value = pick(event); setHover(value); if (dragging.current && onSelect && value !== snr) onSelect(value) }}
      onPointerUp={() => { dragging.current = false }}
      onPointerLeave={() => { setHover(null) }}
    >
      {lastZero >= 0 && <g className="chart-dead-zone">
        <rect x={x(grid[0])} y={pad.top} width={Math.max(0, x(grid[lastZero]) - x(grid[0]))} height={plotH} />
        <text x={x(grid[0]) + 8} y={pad.top + 16}>Only the AI link gets through</text>
      </g>}
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => <g key={tick} className="chart-grid">
        <line x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} />
        <text x={pad.left - 8} y={y(tick) + 4} textAnchor="end">{Math.round(tick * 100)}%</text>
      </g>)}
      <g className="chart-floor">
        <line x1={pad.left} x2={width - pad.right} y1={y(0.1)} y2={y(0.1)} />
        <text x={width - pad.right - 4} y={y(0.1) - 6} textAnchor="end">Fixed guess (≈10%)</text>
      </g>
      {grid.map((value) => <line key={value} className="chart-tick" x1={x(value)} x2={x(value)} y1={pad.top + plotH} y2={pad.top + plotH + 5} />)}
      {grid.map((value) => <text key={value} className="chart-axis" x={x(value)} y={height - 8} textAnchor="middle">{signed(value)}</text>)}
      {breakAt > 0 && <g className="chart-break" transform={`translate(${(x(grid[breakAt - 1]) + x(grid[breakAt])) / 2}, ${pad.top + plotH})`}>
        <rect x={-5} y={-5} width={10} height={10} /><path d="M -5 4 L -1 -4 M 1 4 L 5 -4" />
      </g>}
      {hover !== null && hover !== snr && <line className="chart-hover" x1={x(hover)} x2={x(hover)} y1={pad.top} y2={pad.top + plotH} />}
      <line className="chart-marker" data-testid="marker" data-snr={snr} x1={x(snr)} x2={x(snr)} y1={pad.top - 4} y2={pad.top + plotH} />
      {active.map((item, index) => ({ item, index })).sort((a, b) => Number(a.item.id === 'learned') - Number(b.item.id === 'learned')).map(({ item, index }) => {
        const stroke = seriesColor(item, index)
        const line = item.points.map((p) => `${x(p.snr_db)},${y(p.accuracy)}`).join(' ')
        const isAi = item.id === 'learned'
        // 95% interval band, drawn only when the API supplies one for every point (multi-seed curves).
        const banded = item.points.every((p) => p.ci_low !== undefined && p.ci_high !== undefined)
        const band = banded ? [...item.points.map((p) => `${x(p.snr_db)},${y(p.ci_high!)}`), ...[...item.points].reverse().map((p) => `${x(p.snr_db)},${y(p.ci_low!)}`)].join(' ') : ''
        return <g key={item.id} data-testid="series" data-stroke={stroke} data-series={item.id}>
          {isAi && <polygon className="chart-area" points={`${x(grid[0])},${y(0)} ${line} ${x(grid[grid.length - 1])},${y(0)}`} fill={stroke} />}
          {banded && <polygon className="chart-band" data-testid="band" points={band} fill={stroke} />}
          <polyline className={isAi ? 'chart-line chart-line-ai' : 'chart-line'} points={line} stroke={stroke} />
          {item.points.map((p) => <circle key={p.snr_db} data-testid="point" data-snr={p.snr_db} cx={x(p.snr_db)} cy={y(p.accuracy)} r={p.snr_db === snr ? 6 : 2.6} fill={stroke} className={p.snr_db === snr ? 'chart-dot chart-dot-active' : 'chart-dot'} />)}
        </g>
      })}
    </svg>
    <div className="chart-caption" aria-hidden="true"><span>← noisier</span><span>signal strength (dB)</span><span>cleaner →</span></div>
  </div>
}
