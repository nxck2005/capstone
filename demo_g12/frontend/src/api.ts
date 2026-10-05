export type Ratio = string

export interface Metadata {
  dataset: string
  snr_grid_db: number[]
  default_snr_db: number
  ratios: { id: Ratio; label: string; channel_uses: number }[]
  evidence_label: string
  source_commit?: string
  inference_available?: boolean
  inference_reason?: string
  n_images?: number
}

export interface ChartPoint {
  snr_db: number
  accuracy: number
  coverage?: number | null
  /** 95% image-bootstrap interval; present only for curves averaged over several seed cells. */
  ci_low?: number
  ci_high?: number
}

export interface ChartSeries {
  id: string
  label: string
  color?: string
  cells?: number
  points: ChartPoint[]
}

export interface ChartResponse {
  ratio: Ratio
  series: ChartSeries[]
}

export interface DemoImage {
  id: string
  label: string
  thumbnail_url: string
  truth_label?: string | null
  split?: 'train'
  example_split?: 'train'
}

export interface ImagesResponse {
  images: DemoImage[]
}

export interface InferenceArm {
  predicted_label: string | null
  confidence: number | null
  image_url: string | null
  status: 'delivered' | 'decode_failure' | 'codec_infeasibility' | 'structural_infeasibility' | 'unavailable'
  detail?: string | null
}

export interface InferResponse {
  image_id: string
  ratio: Ratio
  snr_db: number
  input_image_url: string
  classical: InferenceArm
  learned: InferenceArm
  reason?: string | null
}

async function json<T>(url: string, options?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(url, { ...options, headers: { Accept: 'application/json', ...options?.headers } })
  } catch (error) {
    if (options?.signal?.aborted) throw error
    throw new Error('The local demo service is unavailable. Check that the API is running.')
  }
  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json() as { detail?: string }
      detail = typeof body.detail === 'string' ? body.detail : ''
    } catch { /* preserve status if server did not send JSON */ }
    throw new Error(detail || `The demo service returned HTTP ${response.status}.`)
  }
  return response.json() as Promise<T>
}

export const api = {
  metadata: (signal?: AbortSignal) => json<Metadata>('/api/metadata', { signal }),
  chart: (ratio: Ratio, signal?: AbortSignal) => json<ChartResponse>(`/api/chart?ratio=${encodeURIComponent(ratio)}`, { signal }),
  images: (signal?: AbortSignal) => json<ImagesResponse>('/api/images', { signal }),
  infer: (image_id: string, snr_db: number, ratio: Ratio, signal?: AbortSignal) => json<InferResponse>('/api/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id, snr_db, ratio }),
    signal,
  }),
}

export function isValidMetadata(value: Metadata): boolean {
  return Array.isArray(value.snr_grid_db) && value.snr_grid_db.length > 0 &&
    value.snr_grid_db.every((n) => Number.isFinite(n)) && new Set(value.snr_grid_db).size === value.snr_grid_db.length &&
    value.snr_grid_db.includes(value.default_snr_db) && Array.isArray(value.ratios) && value.ratios.length > 0 &&
    value.ratios.every((ratio) => typeof ratio.id === 'string' && ratio.id.length > 0 && typeof ratio.label === 'string')
}

export function isValidImages(value: ImagesResponse): boolean {
  return Array.isArray(value.images) && value.images.length === 4 &&
    value.images.every((image) => typeof image.id === 'string' && image.id.length > 0 &&
      typeof image.label === 'string' &&
      (image.split === 'train' || image.example_split === 'train') &&
      typeof image.thumbnail_url === 'string' && image.thumbnail_url.startsWith('/api/') && !image.thumbnail_url.startsWith('//')) &&
    new Set(value.images.map((image) => image.id)).size === 4
}

export function isValidChart(value: ChartResponse, ratio: Ratio, grid: number[]): boolean {
  return value.ratio === ratio && Array.isArray(value.series) && value.series.length > 0 &&
    value.series.every((series) => typeof series.id === 'string' && typeof series.label === 'string' &&
      Array.isArray(series.points) && series.points.length === grid.length &&
      grid.every((snr, index) => series.points[index]?.snr_db === snr &&
        Number.isFinite(series.points[index].accuracy) && series.points[index].accuracy >= 0 && series.points[index].accuracy <= 1))
}

export function isValidInfer(value: InferResponse, imageId: string, snr: number, ratio: Ratio): boolean {
  return value.image_id === imageId && value.snr_db === snr && value.ratio === ratio &&
    ['classical', 'learned'].every((arm) => {
      const result = value[arm as 'classical' | 'learned']
      return !!result && typeof result.status === 'string' &&
        (result.confidence === null || (typeof result.confidence === 'number' && result.confidence >= 0 && result.confidence <= 1))
    })
}
