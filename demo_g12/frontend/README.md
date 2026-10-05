# Signal Lab · exhibition frontend (G-12 edition)

A copy of `demo/frontend`: React 18 + TypeScript + Vite + Tailwind, fonts bundled, no CDN requests. It differs in three ways. Curves draw a 95% interval band when the API supplies `ci_low`/`ci_high`. The evidence heading reads "Final test · N pictures". The no-delivery sentence quotes the measured AI accuracy.

```bash
cd demo_g12/frontend
npm ci
npm run dev     # http://localhost:5173 ; /api proxies to 127.0.0.1:8001 (override with DEMO_API_URL)
npm test
npm run build
```

The API contract is in `../backend/API.md`. The frontend still rejects curves that miss a grid point, results for another image or SNR, and non-training galleries.
