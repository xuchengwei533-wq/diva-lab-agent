from DivaApp.Configuration import LoadAppSettings
from tts import *  # noqa: F401,F403
from tts import app


if __name__ == "__main__":
    import uvicorn

    Settings = LoadAppSettings().Tts
    uvicorn.run(app, host=Settings.Host, port=Settings.Port)
