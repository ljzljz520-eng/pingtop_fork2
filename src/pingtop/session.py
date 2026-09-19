from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime, timezone
from uuid import uuid4

from pingtop.models import (
    ExportFormat,
    Generation,
    HostConfig,
    HostId,
    HostRecord,
    HostState,
    PingResult,
    SampleSeq,
    SampleStatus,
    SessionConfig,
    SessionSnapshot,
    SortKey,
    normalize_target,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dotted_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts = value.strip().split(".")
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts
    )


class PingSession:
    def __init__(self, config: SessionConfig, targets: Iterable[str]) -> None:
        self.config = config
        self._hosts: OrderedDict[HostId, HostRecord] = OrderedDict()
        self.sort_key = SortKey.HOST
        self.sort_reverse = False
        self.selected_host_id: HostId | None = None
        for target in targets:
            self.add_host(target)

    @property
    def hosts(self) -> OrderedDict[HostId, HostRecord]:
        return self._hosts

    def add_host(self, target: str) -> HostId:
        clean_target = target.strip()
        if not clean_target:
            raise ValueError("Host cannot be empty.")
        normalized = normalize_target(clean_target)
        if any(
            normalize_target(record.config.target) == normalized
            for record in self._hosts.values()
        ):
            raise ValueError(f"Host '{clean_target}' already exists.")
        host_id = uuid4().hex[:8]
        record = HostRecord(config=HostConfig(id=host_id, target=clean_target))
        self._hosts[host_id] = record
        if self.selected_host_id is None:
            self.selected_host_id = host_id
        return host_id

    def edit_host(self, host_id: HostId, new_target: str) -> None:
        record = self.require_host(host_id)
        clean_target = new_target.strip()
        if not clean_target:
            raise ValueError("Host cannot be empty.")
        normalized = normalize_target(clean_target)
        for existing_id, existing in self._hosts.items():
            if existing_id == host_id:
                continue
            if normalize_target(existing.config.target) == normalized:
                raise ValueError(f"Host '{clean_target}' already exists.")
        record.config.target = clean_target
        # Editing the target opens a new measurement window: results issued
        # for the old target must never land in this window.
        self._bump_generation(record, keep_ip=False)

    def delete_host(self, host_id: HostId) -> None:
        record = self.require_host(host_id)
        record.stats.mark_deleted()
        del self._hosts[host_id]
        if self.selected_host_id == host_id:
            self.selected_host_id = next(iter(self._hosts), None)

    def pause_host(self, host_id: HostId) -> None:
        record = self.require_host(host_id)
        record.paused = True
        record.stats.mark_paused()

    def resume_host(self, host_id: HostId) -> None:
        record = self.require_host(host_id)
        record.paused = False
        record.stats.mark_pending()

    def toggle_host_pause(self, host_id: HostId) -> None:
        record = self.require_host(host_id)
        if record.paused:
            self.resume_host(host_id)
        else:
            self.pause_host(host_id)

    def pause_all(self) -> None:
        for host_id in list(self._hosts):
            self.pause_host(host_id)

    def resume_all(self) -> None:
        for host_id in list(self._hosts):
            self.resume_host(host_id)

    def toggle_all_pause(self) -> None:
        if any(not record.paused for record in self._hosts.values()):
            self.pause_all()
        else:
            self.resume_all()

    def reset_host(self, host_id: HostId) -> None:
        record = self.require_host(host_id)
        # A reset closes the measurement window as well: probes issued before
        # the reset must not be counted in the fresh window.
        self._bump_generation(record, keep_ip=True)
        if record.paused:
            record.stats.mark_paused()

    def reset_all(self) -> None:
        for host_id in list(self._hosts):
            self.reset_host(host_id)

    def accept_result(
        self,
        host_id: HostId,
        result: PingResult,
        *,
        generation: Generation,
        seq: SampleSeq,
        when: datetime | None = None,
    ) -> SampleStatus:
        """Gate a delivered probe sample before it touches statistics.

        A sample is accepted only when it was issued for the host's current
        generation and its probe sequence number is strictly greater than the
        last one accepted. This rejects late results from prior targets or
        pre-reset probes (stale generation) as well as duplicate and
        out-of-order delivery within a generation.
        """
        when = when or utcnow()
        record = self.require_host(host_id)
        if generation != record.generation:
            return SampleStatus.STALE_GENERATION
        if seq <= record.stats.accepted_seq:
            return SampleStatus.STALE_SEQ
        record.stats.accepted_seq = seq
        self.apply_result(host_id, result, when)
        return SampleStatus.ACCEPTED

    @staticmethod
    def _bump_generation(record: HostRecord, *, keep_ip: bool) -> None:
        record.generation += 1
        record.stats.reset(keep_ip=keep_ip)

    def apply_result(
        self, host_id: HostId, result: PingResult, when: datetime | None = None
    ) -> None:
        when = when or utcnow()
        record = self.require_host(host_id)
        if record.paused:
            record.stats.mark_paused()
            return
        if result.success and result.rtt_ms is not None:
            record.stats.register_success(result.rtt_ms, result.resolved_ip, when)
            return
        if result.error_message:
            record.stats.register_error(result.error_message, when)
            if result.resolved_ip:
                record.stats.resolved_ip = result.resolved_ip
            return
        record.stats.register_timeout(when)
        if result.resolved_ip:
            record.stats.resolved_ip = result.resolved_ip

    def select(self, host_id: HostId | None) -> None:
        self.selected_host_id = host_id if host_id in self._hosts else None

    def current_host(self) -> HostRecord | None:
        if self.selected_host_id is None:
            return None
        return self._hosts.get(self.selected_host_id)

    def require_host(self, host_id: HostId) -> HostRecord:
        try:
            return self._hosts[host_id]
        except KeyError as exc:
            raise KeyError(f"Unknown host id: {host_id}") from exc

    def host_snapshot(self, host_id: HostId) -> dict[str, object]:
        return self.require_host(host_id).snapshot()

    def host_snapshots(self) -> list[dict[str, object]]:
        rows = [record.snapshot() for record in self._hosts.values()]
        rows.sort(key=self._sort_value, reverse=self.sort_reverse)
        return rows

    def set_sort(self, sort_key: SortKey, reverse: bool | None = None) -> None:
        self.sort_key = sort_key
        if reverse is not None:
            self.sort_reverse = reverse

    def cycle_sort(self) -> None:
        keys = list(SortKey)
        next_index = (keys.index(self.sort_key) + 1) % len(keys)
        self.sort_key = keys[next_index]

    def toggle_sort_order(self) -> None:
        self.sort_reverse = not self.sort_reverse

    def aggregates(self) -> dict[str, object]:
        total_hosts = len(self._hosts)
        active_hosts = sum(1 for record in self._hosts.values() if not record.paused)
        paused_hosts = sum(1 for record in self._hosts.values() if record.paused)
        error_hosts = sum(
            1 for record in self._hosts.values() if record.stats.state == HostState.ERROR
        )
        total_sent = sum(record.stats.seq for record in self._hosts.values())
        total_lost = sum(record.stats.lost for record in self._hosts.values())
        loss_percent = (total_lost / total_sent) * 100 if total_sent else 0.0
        return {
            "total_hosts": total_hosts,
            "active_hosts": active_hosts,
            "paused_hosts": paused_hosts,
            "error_hosts": error_hosts,
            "total_sent": total_sent,
            "total_lost": total_lost,
            "loss_percent": loss_percent,
        }

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            generated_at=utcnow(),
            config=self.config,
            hosts=[record.snapshot() for record in self._hosts.values()],
            aggregates=self.aggregates(),
        )

    def _sort_value(self, row: dict[str, object]) -> tuple[bool, object]:
        value = row.get(self.sort_key.value)
        if self.sort_key in {SortKey.HOST, SortKey.IP} and value is not None:
            value = _dotted_sort_key(str(value))
        elif self.sort_key == SortKey.STATE and value is not None:
            value = str(value)
        return value is None, value


def infer_export_format(export_path: str, explicit: str | None) -> ExportFormat:
    if explicit:
        return ExportFormat(explicit)
    suffix = export_path.rsplit(".", 1)[-1].lower() if "." in export_path else ""
    if suffix == "json":
        return ExportFormat.JSON
    if suffix == "csv":
        return ExportFormat.CSV
    raise ValueError("Unable to infer export format. Use --export-format json|csv.")
