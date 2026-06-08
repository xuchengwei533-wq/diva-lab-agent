import logging

from DivaApp.Configuration import LoadAppSettings
from DivaApp.Gateway import App as app


logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import uvicorn

    Settings = LoadAppSettings().Gateway
    Logger.info("ASR_BASE_URL=%s", Settings.AsrBaseUrl)
    Logger.info("TTS_BASE_URL=%s", Settings.TtsBaseUrl)
    Logger.info("Starting Oh-My-Live2D Gateway on http://localhost:%s", Settings.Port)
    Logger.info("API docs: http://localhost:%s/docs", Settings.Port)
    uvicorn.run(app, host=Settings.Host, port=Settings.Port, reload=False)
