# Conformal Recovery Forecaster — Frontend

Next.js 14 + Tailwind + Recharts. Reads from the FastAPI backend.

## Local development

```bash
# Make sure the backend is running on http://localhost:8000
# (see ../backend for setup)

cp .env.local.example .env.local
npm install
npm run dev
# Open http://localhost:3000
```

## Deploy to Vercel

1. Push this repo to GitHub.
2. In Vercel: import the project, set the **root directory** to `web/`.
3. Add environment variable: `NEXT_PUBLIC_API_BASE_URL` = your Railway backend URL.
4. Deploy.

## File map

```
web/
├── src/app/page.tsx       ← single-page demo
├── src/app/layout.tsx
├── src/app/globals.css
├── tailwind.config.ts
├── next.config.js
├── tsconfig.json
└── package.json
```
