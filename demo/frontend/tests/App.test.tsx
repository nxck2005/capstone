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
  classical: snr_db === -8
    ? { status: 'decode_failure', predicted_label: 'tench', confidence: null, image_url: null }
    : { status: 'delivered', predicted_label: 'English springer', confidence: .89, image_url: '/api/assets/decoded.png' },
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
    expect(screen.getByRole('status', { name: 'Local service status' })).toHaveTextContent(/local api ready/i)
    expect(await screen.findByLabelText(/Accuracy versus measured SNR; selected -8 dB/)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('82% sure')).toBeInTheDocument())
    const classical = screen.getByLabelText('Classical digital')
    expect(classical).toHaveTextContent('Fallback guess (no image received)')
    expect(classical).toHaveTextContent('tench')
    expect(classical).toHaveTextContent(/fixed guess used whenever nothing arrives\. It is not a prediction from this picture/)
    expect(classical).toHaveTextContent(/Confidence\s*N\/A/)
    expect(classical).toHaveTextContent('No image received')
    expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual(expect.arrayContaining(['/api/metadata', '/api/images', '/api/chart?ratio=r_1_6', '/api/infer']))
  })

  it('uses only the discrete grid, updates marker, and sends selected image, ratio and SNR to inference', async () => {
    const fetchMock = mockApi()
    window.history.pushState({}, '', '/?ratio=r_1_24')
    render(<App />)
    await screen.findAllByText('72.8%')
    fireEvent.change(screen.getByLabelText(/channel snr/i), { target: { value: '2' } })
    await waitFor(() => expect(screen.getByLabelText(/Accuracy versus measured SNR; selected 18 dB/)).toBeInTheDocument())
    fireEvent.click(within(screen.getByLabelText('Choose a training image')).getByRole('button', { name: 'Train image B' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([url, options]) => url === '/api/infer' && JSON.parse(String(options?.body)).image_id === 'image-b' && JSON.parse(String(options?.body)).snr_db === 18 && JSON.parse(String(options?.body)).ratio === 'r_1_24')).toBe(true))
    expect(screen.getByText('83.4%')).toBeInTheDocument()
    window.history.pushState({}, '', '/')
  })

  it('restores the image-based prediction label when classical delivery succeeds', async () => {
    render(<App />)
    await screen.findByText('Fallback guess (no image received)')
    fireEvent.change(screen.getByLabelText(/channel snr/i), { target: { value: '1' } })
    const classical = screen.getByLabelText('Classical digital')
    await waitFor(() => expect(classical).toHaveTextContent('English springer'))
    expect(classical).toHaveTextContent(/Prediction/)
    expect(classical).toHaveTextContent('89% sure')
    expect(classical).not.toHaveTextContent('Fallback guess')
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
    expect(screen.getByRole('status', { name: 'Local service status' })).toHaveTextContent(/service error/i)
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
    expect(screen.getByRole('status', { name: 'Local service status' })).toHaveTextContent(/inference unavailable/i)
    expect(screen.queryByText(/No image was recovered/)).not.toBeInTheDocument()
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
    expect(within(screen.getByLabelText('Choose a training image')).queryAllByRole('button')).toHaveLength(0)
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
    fireEvent.change(screen.getByLabelText(/channel snr/i), { target: { value: '1' } })
    await waitFor(() => expect(screen.getByLabelText('Semantic DJSCC')).toHaveTextContent('parachute'))
    for (const resolve of late) resolve(response({ ...infer(), learned: { ...infer().learned, predicted_label: 'stale old result' } }))
    await new Promise((resolve) => setTimeout(resolve, 25))
    expect(screen.getByLabelText('Semantic DJSCC')).toHaveTextContent('parachute')
    expect(screen.queryByText('stale old result')).not.toBeInTheDocument()
  })
})
