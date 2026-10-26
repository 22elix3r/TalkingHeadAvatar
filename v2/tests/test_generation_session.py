import asyncio
from dataclasses import replace
from fractions import Fraction

import pytest

from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Discontinuity, Envelope, MotionFrame, RigIdentity
from talkinghead_v2.session.epochs import GenerationSession
from talkinghead_v2.session.synthetic import SyntheticSink


def audio(key, count, sequence=0, start=0, rate=24000):
    return AudioChunk(Envelope(key, sequence, "trace"), "segment", rate, start, (0.25,)*count)


async def drain(session, key):
    while True:
        event = await session.queue.get(key)
        assert session.sink.submit(event)
        if isinstance(event, AudioEOF):
            return


@pytest.mark.parametrize("count", [0, 1, 17, 1009])
def test_eof_delivers_all_samples_once_including_short_final_chunk(count):
    async def scenario():
        sink = SyntheticSink()
        session = GenerationSession("session", sink)
        started = session.start("turn", 24000)
        consumer = asyncio.create_task(drain(session, started.key))
        assert await session.push(audio(started.key, count))
        assert not sink.finished
        eof = AudioEOF(Envelope(started.key, 1, "trace"), 24000, count)
        assert await session.finish(eof)
        await asyncio.wait_for(consumer, 2)
        assert sum(record.chunk.sample_count for record in sink.records) == count*2
        assert sink.eofs[0].end_sample == count*2
        with pytest.raises(ContractError, match="closed"):
            await session.finish(eof)
        assert len(sink.eofs) == 1
        if count:
            assert sink.records[-1].chunk.samples[-1] != 0
            assert sink.records[-1].chunk.segment_id == "segment"
    asyncio.run(scenario())


def test_cancel_before_audio_rejects_late_callback_and_restart_is_isolated():
    async def scenario():
        sink = SyntheticSink()
        session = GenerationSession("session", sink)
        old = session.start("old", 24000)
        ack = session.cancel("cancel before inference")
        assert ack.scope == "synthetic_in_process" and ack.fence_epoch > old.key.epoch
        assert ack.last_scheduled_sample == 0
        assert not await session.push(audio(old.key, 10))
        current = session.start("new", 16000)
        assert current.key.epoch > ack.fence_epoch
        assert not await session.finish(AudioEOF(Envelope(old.key, 1, "trace"), 24000, 0))
        assert not sink.submit(audio(old.key, 10, rate=48000))
        assert sink.records == [] and not sink.finished
        session.cancel("test done")
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["cancel", "discontinuity", "task_cancel"])
def test_cancel_at_backpressure_flushes_queued_and_dequeued_output_without_hanging(mode):
    async def scenario():
        sink = SyntheticSink()
        session = GenerationSession("session", sink, max_queue_items=2,
                                    max_queue_duration=Fraction(1, 50))
        old = session.start("old", 48000)
        producer = asyncio.create_task(session.push(audio(old.key, 2000, rate=48000)))
        await asyncio.wait_for(session.queue.put_blocked.wait(), 1)
        held = session.queue.get_nowait(old.key)
        assert not producer.done()
        if mode == "cancel":
            ack = session.cancel("interrupt")
            assert ack.discarded_queue_samples == 480
        elif mode == "discontinuity":
            ack = session.discontinuity(Discontinuity(Envelope(old.key, 1, "trace"), 48000, 2000, "reset"))
            assert "discontinuity" in ack.reason
        else:
            producer.cancel()
        if mode == "task_cancel":
            with pytest.raises(asyncio.CancelledError):
                await producer
        else:
            assert not await asyncio.wait_for(producer, 1)
        assert session.queue.item_count == 0 and session.queue.samples == 0
        assert not sink.submit(held)
        current = session.start("new", 24000)
        assert not sink.submit(held)
        consumer = asyncio.create_task(drain(session, current.key))
        assert await session.push(audio(current.key, 17))
        assert await session.finish(AudioEOF(Envelope(current.key, 1, "trace"), 24000, 17))
        await asyncio.wait_for(consumer, 1)
        assert len(sink.eofs) == 1 and sink.eofs[0].envelope.key == current.key
        assert sum(item.chunk.sample_count for item in sink.records) == 34
    asyncio.run(scenario())


def test_cancel_during_blocked_eof_discards_tail_instead_of_flushing_old_speech():
    async def scenario():
        sink = SyntheticSink()
        session = GenerationSession("session", sink, max_queue_items=1)
        started = session.start("old", 24000)
        assert await session.push(audio(started.key, 1))
        finishing = asyncio.create_task(session.finish(AudioEOF(Envelope(started.key, 1, "trace"), 24000, 1)))
        await asyncio.wait_for(session.queue.put_blocked.wait(), 1)
        ack = session.cancel("cancel while EOF waits behind tail")
        assert ack.discarded_queue_samples == 2
        assert not await asyncio.wait_for(finishing, 1)
        assert sink.records == [] and sink.eofs == []
    asyncio.run(scenario())


def test_faults_do_not_corrupt_ledger_and_scheduling_pts_continue_across_turns():
    async def scenario():
        sink = SyntheticSink()
        session = GenerationSession("session", sink)
        first = session.start("first", 48000)
        with pytest.raises(ContractError, match="before restart"):
            session.start("too soon", 48000)
        assert await session.push(audio(first.key, 10, rate=48000))
        for event in (audio(first.key, 2, sequence=1, start=11, rate=48000),
                      audio(first.key, 2, sequence=1, start=10, rate=24000),
                      audio(first.key, 2, sequence=0, start=10, rate=48000)):
            with pytest.raises(ContractError):
                await session.push(event)
        assert await session.push(audio(first.key, 7, sequence=1, start=10, rate=48000))
        assert await session.finish(AudioEOF(Envelope(first.key, 2, "trace"), 48000, 17))
        await drain(session, first.key)
        second = session.start("second", 48000)
        assert await session.push(audio(second.key, 5, rate=48000))
        assert await session.finish(AudioEOF(Envelope(second.key, 1, "trace"), 48000, 5))
        await drain(session, second.key)
        assert sink.transport_cursor == 22
        assert sink.records[-1].session_start_sample == 17
        assert all(a.end_ns == b.start_ns for a, b in zip(sink.records, sink.records[1:]))
    asyncio.run(scenario())


def test_motion_rejected_at_final_sink_for_stale_or_incompatible_basis():
    sink = SyntheticSink()
    session = GenerationSession("session", sink)
    rig = RigIdentity("synthetic", "sha256:"+"0"*64, "sha256:"+"1"*64, ("lip",))
    old = session.start("old", 24000, rig=rig)
    motion = MotionFrame(Envelope(old.key, 0, "trace"), rig, 0, 48000, 480, (0.5,))
    session.cancel("interrupt")
    current = session.start("new", 24000, rig=rig)
    assert not sink.submit_motion(motion)
    motion = replace(motion, envelope=Envelope(current.key, 0, "trace"))
    with pytest.raises(ContractError, match="basis"):
        sink.submit_motion(replace(motion, rig=replace(rig, basis_hash="sha256:"+"2"*64)))
    assert sink.submit_motion(motion)
    with pytest.raises(ContractError, match="reordered"):
        sink.submit_motion(motion)


def test_segment_labels_follow_source_sample_spans_despite_filter_lookahead():
    async def scenario():
        sink = SyntheticSink()
        session = GenerationSession("session", sink)
        started = session.start("turn", 24000)
        await session.push(replace(audio(started.key, 10), segment_id="first"))
        assert not sink.records and session.queue.item_count == 0
        await session.push(replace(audio(started.key, 100, sequence=1, start=10), segment_id="second"))
        await session.finish(AudioEOF(Envelope(started.key, 2, "trace"), 24000, 110))
        await drain(session, started.key)
        first = [item.chunk for item in sink.records if item.chunk.segment_id == "first"]
        second = [item.chunk for item in sink.records if item.chunk.segment_id == "second"]
        assert sum(item.sample_count for item in first) == 20
        assert sum(item.sample_count for item in second) == 200
        assert first[-1].end_sample == second[0].start_sample == 20
    asyncio.run(scenario())


def test_processing_failure_explicitly_cancels_instead_of_resuming_after_lost_output(monkeypatch):
    async def scenario():
        sink = SyntheticSink()
        session = GenerationSession("session", sink)
        started = session.start("turn", 24000)

        async def fail(*args):
            raise RuntimeError("injected output boundary failure")

        monkeypatch.setattr(session, "_enqueue", fail)
        with pytest.raises(RuntimeError, match="injected"):
            await session.push(audio(started.key, 100))
        assert session.queue.item_count == 0
        assert sink.key is None and session.epoch > started.key.epoch
        assert not await session.push(audio(started.key, 100, sequence=1, start=100))
    asyncio.run(scenario())


def test_fractional_native_padding_is_rejected_before_eof_flush():
    async def scenario():
        sink = SyntheticSink()
        session = GenerationSession("session", sink)
        started = session.start("turn", 24000)
        await session.push(audio(started.key, 3))
        padded = AudioEOF(Envelope(started.key, 1, "trace"), 24000, 3, Fraction(5, 2))
        with pytest.raises(ContractError, match="padding"):
            await session.finish(padded)
        assert session.queue.item_count == 0 and not sink.finished
        session.cancel("provider must trim padding before the adapter")
    asyncio.run(scenario())
