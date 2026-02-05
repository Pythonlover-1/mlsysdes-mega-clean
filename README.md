# mlsysdes-mega

Full-stack приложение для конвертации изображений в CSV с помощью LLM.

## Демо (уже развернуто)

Сервис доступен по адресу: `http://158.160.99.112:3001/`

## Стек

- **Backend**: Python, FastAPI, OpenRouter API
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **Инфра**: Docker, Docker Compose, Nginx

## Запуск

```bash
export OPENROUTER_API_KEY=your_key
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
