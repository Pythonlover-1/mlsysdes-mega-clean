# mlsysdes-mega

Full-stack приложение для конвертации изображений в CSV с помощью LLM.

## Стек

- **Backend**: Python, FastAPI, OpenRouter API
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **Инфра**: Docker, Docker Compose, Nginx

## Запуск

```bash
export OPENROUTER_API_KEY=your_key
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
