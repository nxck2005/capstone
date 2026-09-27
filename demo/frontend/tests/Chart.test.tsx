import { expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import Chart from '../src/Chart'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  LineChart: ({ children, data }: { children: ReactNode; data: unknown }) => <div data-testid="rows" data-rows={JSON.stringify(data)}>{children}</div>,
  CartesianGrid: () => null,
  XAxis: ({ type, domain }: { type: string; domain: number[] }) => <span data-testid="x-axis" data-type={type} data-domain={JSON.stringify(domain)} />,
  YAxis: () => null,
  Tooltip: () => null,
  Line: ({ stroke }: { stroke: string }) => <span data-testid="curve" data-stroke={stroke} />,
  ReferenceLine: ({ x }: { x: number }) => <span data-testid="marker" data-x={x} />,
}))

it('plots only measured samples at their numeric SNR positions, using W10 colors and selected marker', () => {
  const grid = [-8, -4, 18]
  const points = (values: number[]) => grid.map((snr_db, index) => ({ snr_db, accuracy: values[index] }))
  const series = [
    { id: 'learned', label: 'Learned', points: points([.728, .794, .834]) },
    { id: 'classical_adaptive', label: 'Classical', points: points([.1, .834, .893]) },
    { id: 'learned_randomized', label: 'Randomized', points: points([.77, .824, .839]) },
  ]
  render(<Chart series={series} snr={18} grid={grid} />)
  expect(screen.getByTestId('x-axis')).toHaveAttribute('data-type', 'number')
  expect(screen.getByTestId('x-axis')).toHaveAttribute('data-domain', '[-8,18]')
  expect(screen.getByTestId('marker')).toHaveAttribute('data-x', '18')
  expect(JSON.parse(screen.getByTestId('rows').getAttribute('data-rows')!)).toEqual([
    { snr_db: -8, arm0: .728, arm1: .1, arm2: .77 },
    { snr_db: -4, arm0: .794, arm1: .834, arm2: .824 },
    { snr_db: 18, arm0: .834, arm1: .893, arm2: .839 },
  ])
  expect(screen.getAllByTestId('curve').map((item) => item.getAttribute('data-stroke'))).toEqual(['#0072B2', '#D55E00', '#009E73'])
  expect(screen.getByLabelText(/selected 18 dB/)).toBeInTheDocument()
  cleanup()
})
