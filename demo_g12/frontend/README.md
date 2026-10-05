# Signal Lab · exhibition frontend (G-12 edition)

A copy of `demo/frontend`: React 18 + TypeScript + Vite + Tailwind, fonts bundled, no CDN requests. It differs in five ways. Curves draw a 95% interval band when the API supplies `ci_low`/`ci_high`. The evidence heading reads "Final test · N pictures". The no-delivery sentence quotes the measured AI accuracy. The gallery is the ten default test images, plus a "Browse all test images" panel (class filter, 40 per page, Esc to close). Each receiver card shows the recorded G-12 outcome and whether the live result matches it.

```bash
cd demo_g12/frontend
npm ci
npm run dev     # http://localhost:5173 ; /api proxies to 127.0.0.1:8001 (override with DEMO_API_URL)
npm test
npm run build
```

The API contract is in `../backend/API.md`. The frontend still rejects curves that miss a grid point and results for another image or SNR. It also rejects any gallery that isn't ten test images with distinct classes.
