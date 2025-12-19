import asyncio

from app.database.db import engine, Base
from app.database import models


async def run():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # WAJIB untuk aiomysql + Windows
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())
