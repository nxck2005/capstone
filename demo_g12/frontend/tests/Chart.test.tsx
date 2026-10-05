import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import Chart from '../src/Chart'

const grid = [-8, -4, 18]
const points = (values: number[], coverage?: number[]) => grid.map((snr_db, index) => ({ snr_db, accuracy: values[index], coverage: coverage?.[index] }))
const series = [
  { id: 'learned', label: 'Learned', points: points([.728, .794, .834], [1, 1, 1]) },
  { id: 'classical_adaptive', label: 'Classical', points: points([.1, .834, .893], [0, 1, 1]) },
  { id: 'learned_randomized', label: 'Randomized', points: points([.77, .824, .839], [1, 1, 1]) },
]

afterEach(() => cleanup())

it('plots only measured samples, one equal step per measured SNR, labelled with real values, in the W10 colors', () => {
  render(<Chart series={series} snr={18} grid={grid} />)
  const curves = screen.getAllByTestId('series')
  const strokes = Object.fromEntries(curves.map((item) => [item.getAttribute('data-series'), item.getAttribute('data-stroke')]))
  expect(strokes).toEqual({ learned: '#0072B2', classical_adaptive: '#D55E00', learned_randomized: '#009E73' })
  for (const curve of curves) {
    const dots = curve.querySelectorAll('[data-testid="point"]')
    expect([...dots].map((dot) => Number(dot.getAttribute('data-snr')))).toEqual(grid)
  }
  // Equal width per measured point; the jump from −4 to +18 is shown by a break mark, not hidden.
  const xs = [...curves[0].querySelectorAll('[data-testid="point"]')].map((dot) => Number(dot.getAttribute('cx')))
  expect(xs[1] - xs[0]).toBeCloseTo(xs[2] - xs[1], 5)
  for (const label of ['−8', '−4', '+18']) expect(screen.getByText(label)).toBeInTheDocument()
  expect(document.querySelector('.chart-break')).not.toBeNull()
  expect(screen.getByTestId('marker')).toHaveAttribute('data-snr', '18')
  expect(screen.getByLabelText(/selected 18 dB/)).toBeInTheDocument()
})

it('selects the nearest measured SNR when the chart is clicked', () => {
  const onSelect = vi.fn()
  render(<Chart series={series} snr={-8} grid={grid} onSelect={onSelect} />)
  const svg = screen.getByRole('img')
  svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 820, height: 190, right: 820, bottom: 190, x: 0, y: 0, toJSON: () => ({}) })
  fireEvent.pointerDown(svg, { clientX: 815, pointerId: 1 })
  expect(onSelect).toHaveBeenLastCalledWith(18)
})

it('draws a 95% interval band only for curves that supply one', () => {
  const withBand = [{ ...series[0], points: series[0].points.map((p) => ({ ...p, ci_low: p.accuracy - .01, ci_high: p.accuracy + .01 })) }, series[1]]
  render(<Chart series={withBand} snr={-8} grid={grid} />)
  const bands = screen.getAllByTestId('band')
  expect(bands).toHaveLength(1)
  expect(bands[0].closest('[data-testid="series"]')).toHaveAttribute('data-series', 'learned')
  // Upper edge left to right, then lower edge back: two vertices per measured SNR.
  expect(bands[0].getAttribute('points')!.split(' ')).toHaveLength(2 * grid.length)
})

it('marks the region where no digital transmission got through', () => {
  render(<Chart series={series} snr={-8} grid={grid} />)
  expect(screen.getByText('Only the AI link gets through')).toBeInTheDocument()
})
