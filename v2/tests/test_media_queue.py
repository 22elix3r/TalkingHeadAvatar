import asyncio
from fractions import Fraction

import pytest

from talkinghead_v2.audio.queue import EpochCancelled, MediaQueue
from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Envelope, GenerationKey


KEY = GenerationKey("session", "turn", 0)
NEXT = GenerationKey("session", "next", 2)


def chunk(count, sequence=0, key=KEY, rate=48000):
    return AudioChunk(Envelope(key, sequence, "trace"), "segment", rate, 0, (sequence/10,)*count)


def test_duration_backpressure_retains_every_sample_in_fifo_order():
    async def scenario():
        queue = MediaQueue(48000, Fraction(1, 100), 99)  # 480 frames, ample item slots.
        queue.reset(KEY)
        first, second = chunk(300), chunk(200, 1)
        assert await queue.put(first)
        blocked = asyncio.create_task(queue.put(second))
        await asyncio.wait_for(queue.put_blocked.wait(), 1)
        assert not blocked.done() and queue.samples == 300 and queue.item_count == 1
        assert await queue.get(KEY) == first
        assert await asyncio.wait_for(blocked, 1)
        assert await queue.get(KEY) == second
        assert queue.samples == 0 and queue.peak_samples <= 480
    asyncio.run(scenario())


def test_zero_duration_items_still_obey_item_bound_and_cancel_wakes_producer():
    async def scenario():
        queue = MediaQueue(48000, Fraction(1), 2)
        queue.reset(KEY)
        assert await queue.put(chunk(0)) and await queue.put(chunk(0, 1))
        blocked = asyncio.create_task(queue.put(AudioEOF(Envelope(KEY, 2, "trace"), 48000, 0)))
        await asyncio.wait_for(queue.put_blocked.wait(), 1)
        assert not blocked.done() and queue.item_count == 2 and queue.samples == 0
        assert queue.reset(NEXT) == (0, 2)
        assert not await asyncio.wait_for(blocked, 1)
        assert await queue.put(chunk(1, key=NEXT))
        with pytest.raises(EpochCancelled):
            await queue.get(KEY)
        assert (await queue.get(NEXT)).envelope.key == NEXT
    asyncio.run(scenario())


def test_empty_queue_waits_for_explicit_eof_and_old_consumer_cannot_drain_new_epoch():
    async def scenario():
        queue = MediaQueue(48000, Fraction(1), 2)
        queue.reset(KEY)
        consumer = asyncio.create_task(queue.get(KEY))
        await asyncio.sleep(0)
        assert not consumer.done()
        queue.reset(NEXT)
        assert await queue.put(AudioEOF(Envelope(NEXT, 0, "trace"), 48000, 0))
        with pytest.raises(EpochCancelled):
            await asyncio.wait_for(consumer, 1)
        assert isinstance(await queue.get(NEXT), AudioEOF)
    asyncio.run(scenario())


def test_oversized_or_wrong_rate_chunk_fails_without_evicting_existing_audio():
    async def scenario():
        queue = MediaQueue(48000, Fraction(1, 100), 2)
        queue.reset(KEY)
        first = chunk(100)
        await queue.put(first)
        for event in (chunk(481), chunk(1, rate=24000)):
            with pytest.raises(ContractError):
                await queue.put(event)
        assert not await queue.put(chunk(1, key=NEXT))
        assert queue.samples == 100 and queue.get_nowait(KEY) == first
    asyncio.run(scenario())


def test_epoch_reset_cannot_silently_clear_audio_or_reactivate_a_cancelled_key():
    async def scenario():
        queue = MediaQueue(48000, Fraction(1), 2)
        queue.reset(KEY)
        await queue.put(chunk(100))
        for invalid in (KEY, GenerationKey("other-session", "turn", 10)):
            with pytest.raises(ContractError):
                queue.reset(invalid)
            assert queue.samples == 100
        queue.reset(None)
        with pytest.raises(ContractError):
            queue.reset(KEY)
        queue.reset(NEXT)
    asyncio.run(scenario())
