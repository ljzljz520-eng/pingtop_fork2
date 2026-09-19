from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pingtop.models import PingResult, SampleStatus, SessionConfig, SortKey
from pingtop.session import PingSession


def build_session(*targets: str) -> PingSession:
    return PingSession(SessionConfig(), targets)


def test_add_edit_delete_and_select_host() -> None:
    session = build_session("1.1.1.1")
    host_id = next(iter(session.hosts))

    added_id = session.add_host("8.8.8.8")
    assert added_id in session.hosts

    session.edit_host(added_id, "9.9.9.9")
    assert session.hosts[added_id].config.target == "9.9.9.9"

    session.select(added_id)
    assert session.current_host() is session.hosts[added_id]

    session.delete_host(host_id)
    assert host_id not in session.hosts


def test_pause_resume_reset_and_aggregate_updates() -> None:
    session = build_session("1.1.1.1")
    host_id = next(iter(session.hosts))

    session.apply_result(host_id, PingResult(success=True, rtt_ms=11.5, resolved_ip="1.1.1.1"))
    session.apply_result(host_id, PingResult(success=False, resolved_ip="1.1.1.1"))
    row = session.host_snapshot(host_id)

    assert row["seq"] == 2
    assert row["lost"] == 1
    assert row["trend"]

    session.pause_host(host_id)
    assert session.hosts[host_id].paused is True

    session.resume_host(host_id)
    assert session.hosts[host_id].paused is False

    session.reset_host(host_id)
    reset_row = session.host_snapshot(host_id)
    assert reset_row["seq"] == 0
    assert reset_row["lost"] == 0
    assert reset_row["trend"] == ""


def test_cycle_and_toggle_sort() -> None:
    session = build_session("b.example", "a.example")

    assert session.sort_key == SortKey.HOST
    session.cycle_sort()
    assert session.sort_key == SortKey.IP

    session.toggle_sort_order()
    assert session.sort_reverse is True


def test_dotted_host_sort_uses_numeric_segments() -> None:
    session = build_session("1.1.1.10", "1.1.1.6", "1.1.1.9", "1.1.1.7")

    rows = session.host_snapshots()

    assert [row["target"] for row in rows] == [
        "1.1.1.6",
        "1.1.1.7",
        "1.1.1.9",
        "1.1.1.10",
    ]

    session.set_sort(SortKey.HOST, reverse=True)
    rows = session.host_snapshots()

    assert [row["target"] for row in rows] == [
        "1.1.1.10",
        "1.1.1.9",
        "1.1.1.7",
        "1.1.1.6",
    ]


def success(seq: int, *, ip: str = "1.1.1.1") -> PingResult:
    return PingResult(success=True, rtt_ms=float(seq), resolved_ip=ip)


def test_generation_bumps_on_edit_and_reset() -> None:
    session = build_session("1.1.1.1")
    host_id = next(iter(session.hosts))
    session.apply_result(host_id, success(1))
    assert session.hosts[host_id].generation == 1

    session.reset_host(host_id)
    assert session.hosts[host_id].generation == 2
    assert session.hosts[host_id].stats.seq == 0
    # Reset only clears statistics: the last resolved IP is retained.
    assert session.hosts[host_id].stats.resolved_ip == "1.1.1.1"

    session.edit_host(host_id, "2.2.2.2")
    assert session.hosts[host_id].generation == 3
    # Edit starts a fully clean slate for the new target.
    assert session.hosts[host_id].stats.resolved_ip is None


def test_accept_result_gates_generation_and_seq() -> None:
    session = build_session("1.1.1.1")
    host_id = next(iter(session.hosts))
    generation = session.hosts[host_id].generation

    assert (
        session.accept_result(host_id, success(1), generation=generation, seq=1)
        is SampleStatus.ACCEPTED
    )
    # Duplicate delivery of the same probe sequence.
    assert (
        session.accept_result(host_id, success(1), generation=generation, seq=1)
        is SampleStatus.STALE_SEQ
    )
    # Gaps are fine, but the water mark moves forward ...
    assert (
        session.accept_result(host_id, success(3), generation=generation, seq=3)
        is SampleStatus.ACCEPTED
    )
    # ... so an older probe arriving afterwards is an out-of-order stale seq.
    assert (
        session.accept_result(host_id, success(2), generation=generation, seq=2)
        is SampleStatus.STALE_SEQ
    )
    stats = session.hosts[host_id].stats
    assert stats.seq == 2
    assert stats.accepted_seq == 3

    # Results from prior or future generations never touch the window.
    assert (
        session.accept_result(host_id, success(4), generation=generation - 1, seq=4)
        is SampleStatus.STALE_GENERATION
    )
    assert (
        session.accept_result(host_id, success(1), generation=generation + 1, seq=1)
        is SampleStatus.STALE_GENERATION
    )
    assert stats.seq == 2


def test_stale_generation_samples_never_touch_new_window() -> None:
    session = build_session("1.1.1.1")
    host_id = next(iter(session.hosts))
    session.accept_result(host_id, success(1), generation=1, seq=1)

    session.reset_host(host_id)
    current_generation = session.hosts[host_id].generation
    stats = session.hosts[host_id].stats
    assert stats.seq == 0
    assert stats.accepted_seq == 0

    # A late pre-reset result cannot enter the fresh window.
    assert (
        session.accept_result(
            host_id, success(99, ip="9.9.9.9"), generation=1, seq=2
        )
        is SampleStatus.STALE_GENERATION
    )
    assert stats.seq == 0
    assert stats.last_rtt_ms is None

    # Only a sample issued after the reset can become sequence number 1.
    assert (
        session.accept_result(
            host_id, success(1, ip="2.2.2.2"),
            generation=current_generation, seq=1,
        )
        is SampleStatus.ACCEPTED
    )
    assert stats.seq == 1
    assert stats.last_rtt_ms == 1.0


@pytest.mark.parametrize("order", list(itertools.permutations(range(1, 5))))
def test_all_permutations_accept_strictly_monotonic(order: tuple[int, ...]) -> None:
    session = build_session("1.1.1.1")
    host_id = next(iter(session.hosts))
    generation = session.hosts[host_id].generation

    accepted: list[int] = []
    high_water = 0
    for seq in order:
        status = session.accept_result(
            host_id, success(seq), generation=generation, seq=seq
        )
        if seq > high_water:
            high_water = seq
            assert status is SampleStatus.ACCEPTED
            accepted.append(seq)
        else:
            assert status is SampleStatus.STALE_SEQ

    stats = session.hosts[host_id].stats
    assert stats.seq == len(accepted)
    assert stats.accepted_seq == 4
    # Accepted probe sequences are strictly increasing regardless of delivery
    # order.
    assert all(
        left < right
        for left, right in zip(accepted, accepted[1:], strict=False)
    )


@given(stream=st.lists(st.integers(min_value=1, max_value=10), max_size=40))
def test_property_delivery_interleavings(stream: list[int]) -> None:
    """Every duplicate/out-of-order interleaving preserves causal ordering."""
    session = build_session("1.1.1.1")
    host_id = next(iter(session.hosts))
    generation = session.hosts[host_id].generation

    accepted: list[int] = []
    high_water = 0
    for seq in stream:
        # Every fourth high-water sample is a timeout to exercise loss.
        result = (
            PingResult(success=False, resolved_ip="1.1.1.1")
            if seq % 4 == 0
            else success(seq)
        )
        status = session.accept_result(
            host_id, result, generation=generation, seq=seq
        )
        if seq > high_water:
            high_water = seq
            assert status is SampleStatus.ACCEPTED
            accepted.append(seq)
        else:
            assert status is SampleStatus.STALE_SEQ

    stats = session.hosts[host_id].stats
    assert stats.accepted_seq == high_water
    assert stats.seq == len(accepted)
    assert len(stats.history_ms) == len(accepted)
    assert stats.lost == sum(1 for seq in accepted if seq % 4 == 0)
    assert all(
        left < right
        for left, right in zip(accepted, accepted[1:], strict=False)
    )
