# LogiMind UI (Angular)

The chat interface from the Design Document, Section 10: role selector,
quick actions, chat panel with agent-trace chips, and the disclaimer banner.
It calls the backend's `POST /chat` (question, role, thread_id).

## Run it locally

Needs Node.js 18.19 or newer.

```
cd frontend
npm install        # first time only: downloads Angular into node_modules/
npm start          # same as `ng serve`; then open http://localhost:4200
```

The backend must be reachable at http://localhost:8000. If it runs on EC2,
open an SSH tunnel first and keep that window open:

```
ssh -i /path/to/your-key.pem -L 8000:localhost:8000 admin@<EC2-public-IP>
```

The backend also needs CORS allowing http://localhost:4200 (in
`src/api/routes.py` of the backend).

## Where things are

| File | Job |
|---|---|
| `src/app/api.config.ts` | The backend address (the one place to change for Vercel) |
| `src/app/chat.service.ts` | Calls `POST /chat` |
| `src/app/app.component.ts` | Messages, role, thread ID, quick actions |
| `src/app/app.component.html` | Layout |
| `src/app/app.component.css` | Styles (phone layout included) |

## Build for deployment

```
npm run build      # output in dist/logimind-ui/browser
```
