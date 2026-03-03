import asyncio
import random
import time

async def newsProducer(myQueue):
    while True:
        await myQueue.put(random.randint(1, 5))
        await asyncio.sleep(1)
        print('produced news')


async def newsConsumer(myQueue):
    while True:
        articleId = await myQueue.get()
        print("News Reader Consumed News Article:", articleId)


myQueue = asyncio.Queue()
loop = asyncio.get_event_loop()

loop.create_task(newsProducer(myQueue))
loop.create_task(newsConsumer(myQueue))

try:
    loop.run_forever()
finally:
    loop.close()