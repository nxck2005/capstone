import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import App from '../src/App'

const metadata = {
  dataset: 'Imagenette-160 validation',
  evidence_label: 'W10 v11 closeout · validation only',
  default_snr_db: -8,
  snr_grid_db: [-8, -4, 18],
  ratios: [{ id: 'r_1_6', label: '1/6', channel_uses: 12800 }, { id: 'r_1_24', label: '1/24', channel_uses: 3200 }],
}
const images = { images: [
  { id: 'image-a', label: 'Train image A', thumbnail_url: '/api/assets/a.png', truth_label: 'tench', split: 'train' },
  { id: 'image-b', label: 'Train image B', thumbnail_url: '/api/assets/b.png', split: 'train' },
  { id: 'image-c', label: 'Train image C', thumbnail_url: '/api/assets/c.png', split: 'train' },
  { id: 'image-d', label: 'Train image D', thumbnail_url: '/api/assets/d.png', split: 'train' },
] }
const chart = (ratio = 'r_1_6') => ({ ratio, series: [
  { id: 'learned', label: 'Learned DJSCC', points: [{ snr_db: -8, accuracy: .728 }, { snr_db: -4, accuracy: .794 }, { snr_db: 18, accuracy: .834 }] },
  { id: 'classical', label: 'Adaptive classical', points: [{ snr_db: -8, accuracy: .1 }, { snr_db: -4, accuracy: .834 }, { snr_db: 18, accuracy: .893 }] },
] })
const infer = (image_id = 'image-a', snr_db = -8, ratio = 'r_1_6') => ({
  image_id, snr_db, ratio, input_image_url: '/api/assets/a.png',
  learned: { status: 'delivered', predicted_label: 'tench', confidence: .82, image_url: '/api/assets/learned.png' },
  classical: { status: 'decode_failure', predicted_label: 'cassette player', confidence: .1, image_url: null },
})
const response = (body: unknown) => ({ ok: true, json: async () => body }) as Response

function mockApi() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (url === '/api/metadata') return response(metadata)
    if (url === '/api/images') return response(images)
    if (url.startsWith('/api/chart')) return response(chart(new URL(url, 'http://localhost').searchParams.get('ratio')!))
    if (url === '/api/infer') {
      const body = JSON.parse(String(init?.body)) as { image_id: string; snr_db: number; ratio: string }
      return response(infer(body.image_id, body.snr_db, body.ratio))
    }
    throw new Error(`Unexpected request ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

beforeEach(() => { mockApi() })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('exhibit', () => {
  it('loads only local API resources and displays measured values without a test claim', async () => {
    const fetchMock = mockApi()
    render(<App />)
    expect((await screen.findAllByText('72.8%')).length).toBeGreaterThan(0)
    expect(screen.getAllByText('10.0%').length).toBeGreaterThan(0)
    expect(screen.getByText('Train image A')).toBeInTheDocument()
    expect(screen.getByRole('status', { name: 'Local service status' })).toHaveTextContent('LOCAL API READY')
    expect(screen.getAllByText(/TEST SPLIT SEALED|TEST SEALED/).length).toBeGreaterThan(0)
    expect(screen.getByText(/Lines connect measured points/)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('82.0%')).toBeInTheDocument())
    expect(screen.getByText(/An outage prediction may still be returned/)).toBeInTheDocument()
    expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual(expect.arrayContaining(['/api/metadata', '/api/images', '/api/chart?ratio=r_1_6', '/api/infer']))
  })

  it('uses only the discrete grid, updates marker, and sends selected image, ratio and SNR to inference', async () => {
    const fetchMock = mockApi()
    render(<App />)
    await screen.findAllByText('72.8%')
    fireEvent.change(screen.getByLabelText('CHANNEL SNR'), { target: { value: '2' } })
    await waitFor(() => expect(screen.getByLabelText(/Accuracy versus measured SNR; selected 18 dB/)).toBeInTheDocument())
    fireEvent.change(screen.getByLabelText('Bandwidth budget'), { target: { value: 'r_1_24' } })
    fireEvent.click(screen.getByRole('button', { name: /Choose image/i }))
    fireEvent.click(within(screen.getByLabelText('Choose a training image')).getByRole('button', { name: 'Train image B' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([url, options]) => url === '/api/infer' && JSON.parse(String(options?.body)).image_id === 'image-b' && JSON.parse(String(options?.body)).snr_db === 18 && JSON.parse(String(options?.body)).ratio === 'r_1_24')).toBe(true))
    expect(screen.getByText('83.4%')).toBeInTheDocument()
  })

  it('fails visibly if the measured chart does not match the published SNR grid', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/metadata') return response(metadata)
      if (url === '/api/images') return response(images)
      if (url.startsWith('/api/chart')) return response({ ...chart(), series: [{ ...chart().series[0], points: [{ snr_db: -8, accuracy: .728 }] }] })
      return response(infer())
    }))
    render(<App />)
    expect(await screen.findByText('The API returned an incomplete or unmeasured accuracy curve.')).toBeInTheDocument()
    expect(screen.queryAllByText('72.8%')).toHaveLength(0)
  })

  it('shows a service error rather than inventing predictions when inference is unavailable', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/metadata') return response(metadata)
      if (url === '/api/images') return response(images)
      if (url.startsWith('/api/chart')) return response(chart())
      throw new Error('offline')
    }))
    render(<App />)
    expect(await screen.findByText('The local demo service is unavailable. Check that the API is running.')).toBeInTheDocument()
    expect(screen.getByLabelText('Semantic DJSCC')).toHaveTextContent('Not available')
    expect(screen.getByRole('status', { name: 'Local service status' })).toHaveTextContent('SERVICE ERROR')
  })

  it('labels unavailable inference honestly without describing it as an outage', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/metadata') return response({ ...metadata, inference_available: false, inference_reason: 'Checkpoint not provisioned.' })
      if (url === '/api/images') return response(images)
      if (url.startsWith('/api/chart')) return response(chart())
      return response({ ...infer(), classical: { status: 'unavailable', predicted_label: null, confidence: null, image_url: null, detail: 'Classical decode unavailable.' }, learned: { status: 'unavailable', predicted_label: null, confidence: null, image_url: null, detail: 'Checkpoint not provisioned.' } })
    }))
    render(<App />)
    await waitFor(() => expect(screen.getByLabelText('Classical digital')).toHaveTextContent('Classical decode unavailable.'))
    expect(screen.getByRole('status', { name: 'Local service status' })).toHaveTextContent('INFERENCE UNAVAILABLE')
    expect(screen.queryByText(/An outage prediction may still/)).not.toBeInTheDocument()
    expect(screen.getAllByText('Not available')).toHaveLength(2)
  })

  it('refuses a validation gallery rather than silently labelling it as train', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url === '/api/metadata') return response(metadata)
      if (url === '/api/images') return response({ images: images.images.map((image) => ({ ...image, split: 'val' })) })
      if (url.startsWith('/api/chart')) return response(chart())
      throw new Error('Inference should not run without a train image')
    }))
    render(<App />)
    expect(await screen.findByText('The API must provide four unique bundled training images with local URLs.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Choose image/i })).toBeDisabled()
  })

  it('never displays a late response for a superseded SNR', async () => {
    const late: Array<(value: Response) => void> = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, options?: RequestInit) => {
      if (url === '/api/metadata') return response(metadata)
      if (url === '/api/images') return response(images)
      if (url.startsWith('/api/chart')) return response(chart())
      const body = JSON.parse(String(options?.body)) as { image_id: string; snr_db: number; ratio: string }
      if (body.snr_db === -8) return new Promise<Response>((resolve) => { late.push(resolve) })
      return response({ ...infer(body.image_id, body.snr_db, body.ratio), learned: { ...infer().learned, predicted_label: 'parachute' } })
    }))
    render(<App />)
    await waitFor(() => expect(late.length).toBeGreaterThan(0))
    fireEvent.change(screen.getByLabelText('CHANNEL SNR'), { target: { value: '1' } })
    await waitFor(() => expect(screen.getByLabelText('Semantic DJSCC')).toHaveTextContent('parachute'))
    for (const resolve of late) resolve(response({ ...infer(), learned: { ...infer().learned, predicted_label: 'stale old result' } }))
    await new Promise((resolve) => setTimeout(resolve, 25))
    expect(screen.getByLabelText('Semantic DJSCC')).toHaveTextContent('parachute')
    expect(screen.queryByText('stale old result')).not.toBeInTheDocument()
  })
})
