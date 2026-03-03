# import asyncio


# async def helllo_world():
#     await asyncio.sleep(1)
#     print('hello world')
#     asyncio.ensure_future(helllo_world())

 

# async def good_evening():
#     await asyncio.sleep(1)
#     print('Good Evening')
#     asyncio.ensure_future(good_evening()) 

# print (f'step: {asyncio.get_event_loop()}')
# loop = asyncio.get_event_loop()
# try:
#     asyncio.ensure_future(helllo_world())
#     asyncio.ensure_future(good_evening())
#     loop.run_forever()

# except KeyboardInterrupt:
#     pass
# finally:
#     print(f"step: {loop.close()}")
#     loop.close()



import asyncio

async def myCoroutine():
    print("My Coroutine")

async def main():
    await asyncio.sleep(1)
loop = asyncio.get_event_loop()

try:
    loop.create_task(myCoroutine())
    loop.create_task(myCoroutine())
    loop.create_task(myCoroutine())

    pending = asyncio.Task.all_tasks()
    print(pending)
    loop.run_until_complete(main())
finally:
    loop.close()