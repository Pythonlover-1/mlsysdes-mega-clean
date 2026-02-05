from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Detection settings
    detection_threshold: float = 0.5
    connection_threshold: float = 200.0
    max_workers: int = 4

    # OCR settings
    ocr_lang: str = "eng+rus"


settings = Settings()
