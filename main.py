from __future__ import annotations

import uvicorn

from service.settings import get_settings


if __name__ == "__main__":
    settings = get_settings()
    settings.validate()
    uvicorn.run(
        "service.app:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        reload=False,
    )
