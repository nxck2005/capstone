import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ChartSeries } from './api'

// Same colorblind-safe assignments used in the W10 results figures.
const palette: Record<string, string> = {
  learned: '#0072B2',
  classical: '#D55E00',
  classical_adaptive: '#D55E00',
  learned_randomized: '#009E73',
  er9_digital: '#CC79A7',
  classical_fixed_mcs: '#E69F00',
}
const color = (item: ChartSeries, index: number) => item.color || palette[item.id] || ['#0072B2', '#D55E00', '#009E73', '#CC79A7'][index]
const percent = (value: number) => `${(value * 100).toFixed(1)}%`

export default function Chart({ series, snr, grid }: { series: ChartSeries[]; snr: number; grid: number[] }) {
  const active = series.slice(0, 4)
  // Each row is an actual measured SNR. The chart never generates intermediate samples.
  const rows = grid.map((snr_db, index) => ({
    snr_db,
    ...Object.fromEntries(active.map((item, arm) => [`arm${arm}`, item.points[index].accuracy])),
  }))

  return <div className="chart-wrap">
    <div className="chart-topline">
      <div><span className="eyebrow">Measured response</span><h2>Accuracy across the channel</h2></div>
      <span className="chart-tag">VALIDATION · 1,000 IMAGES / POINT</span>
    </div>
    <div className="chart-legend" aria-label="Chart series">
      {active.map((item, index) => <span key={item.id} className="legend-item"><i style={{ background: color(item, index) }} />{item.label}</span>)}
    </div>
    <div className="chart-plot" role="img" aria-label={`Accuracy versus measured SNR; selected ${snr} dB. ${active.map((item) => `${item.label}: ${percent(item.points.find((p) => p.snr_db === snr)!.accuracy)}`).join('; ')}`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 19, right: 18, left: -15, bottom: 3 }}>
          <CartesianGrid stroke="#2a4254" strokeDasharray="3 5" vertical={false} />
          <XAxis dataKey="snr_db" type="number" domain={[-8, 18]} allowDataOverflow ticks={grid.filter((_, index) => index % 4 === 0 || index === grid.length - 1)} tickFormatter={(n: number) => n > 0 ? `+${n}` : String(n)} tick={{ fill: '#a9c0cb', fontSize: 11 }} tickLine={false} axisLine={{ stroke: '#405364' }} dy={9} />
          <YAxis type="number" domain={[0, 1]} ticks={[0, .25, .5, .75, 1]} tickFormatter={(n: number) => `${Math.round(n * 100)}%`} tick={{ fill: '#a9c0cb', fontSize: 11 }} tickLine={false} axisLine={false} />
          <Tooltip labelFormatter={(n) => `${Number(n) > 0 ? '+' : ''}${n} dB · measured`} formatter={(value, _name, item) => [percent(Number(value)), active.find((_, index) => `arm${index}` === item.dataKey)?.label ?? 'Accuracy']} contentStyle={{ background: '#192c3e', border: '1px solid #395166', borderRadius: 8, color: '#f0f6f6', fontSize: 12 }} labelStyle={{ color: '#dce8e9' }} />
          {active.map((item, index) => <Line key={item.id} type="linear" dataKey={`arm${index}`} name={item.label} stroke={color(item, index)} strokeWidth={2.8} dot={{ r: 2.5, fill: color(item, index), stroke: '#142637', strokeWidth: 1 }} activeDot={{ r: 5 }} isAnimationActive={false} connectNulls={false} />)}
          <ReferenceLine x={snr} stroke="#d5e1e4" strokeWidth={1.5} strokeDasharray="5 5" ifOverflow="extendDomain" />
        </LineChart>
      </ResponsiveContainer>
    </div>
    <div className="chart-footer"><span>CHANNEL SNR (dB) · SELECTED {snr > 0 ? '+' : ''}{snr} dB</span><span>Lines connect measured points for readability; no intermediate SNR was measured.</span></div>
    <div className="reading-grid" aria-label="Accuracy at selected SNR">
      {active.map((item, index) => <div key={item.id} className="reading"><span className="reading-label"><i style={{ background: color(item, index) }} />{item.label}</span><strong>{percent(item.points.find((p) => p.snr_db === snr)!.accuracy)}</strong>{item.id === 'classical_adaptive' && item.points.find((p) => p.snr_db === snr)?.coverage != null && <small>{percent(item.points.find((p) => p.snr_db === snr)!.coverage!)} delivered</small>}</div>)}
    </div>
  </div>
}
