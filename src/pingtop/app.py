from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import cast

from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Footer, Header, Static

from pingtop.models import (
    Generation,
    HostId,
    PingEngine,
    PingResult,
    SampleSeq,
    SampleStatus,
    SortKey,
)
from pingtop.screens.host_form import ConfirmScreen, HelpScreen, HostFormScreen
from pingtop.session import PingSession
from pingtop.widgets.details_panel import DetailsPanel
from pingtop.widgets.host_table import HostTable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingUpdate:
    host_id: HostId
    generation: Generation
    seq: SampleSeq
    result: PingResult


class PingSample(Message):
    def __init__(
        self,
        host_id: HostId,
        generation: Generation,
        seq: SampleSeq,
        result: PingResult,
    ) -> None:
        self.host_id = host_id
        self.generation = generation
        self.seq = seq
        self.result = result
        super().__init__()


class EditConfirmed(Message):
    def __init__(self, host_id: HostId, target: str) -> None:
        self.host_id = host_id
        self.target = target
        super().__init__()


class DeleteConfirmed(Message):
    def __init__(self, host_id: HostId) -> None:
        self.host_id = host_id
        super().__init__()


class PingTopApp(App[None]):
    CSS_PATH = "app.tcss"
    SORT_HOTKEYS = {
        "H": "target",
        "G": "resolved_ip",
        "S": "seq",
        "R": "last_rtt_ms",
        "I": "min_rtt_ms",
        "A": "avg_rtt_ms",
        "M": "max_rtt_ms",
        "T": "stddev_ms",
        "L": "lost",
        "P": "loss_percent",
        "U": "state",
        "W": "trend",
    }
    BINDINGS = [
        Binding("q", "quit_session", "Quit"),
        Binding("a", "add_host", "Add"),
        Binding("e", "edit_selected", "Edit"),
        Binding("d", "delete_selected", "Delete"),
        Binding("i", "toggle_details", "Details"),
        Binding("space", "toggle_selected_pause", "Pause"),
        Binding("p", "toggle_all_pause", "Pause All"),
        Binding("r", "reset_selected", "Reset"),
        Binding("ctrl+r", "reset_all", "Reset All"),
        Binding("tab", "focus_next"),
        Binding("h", "show_help", "Help"),
        Binding("?", "show_help", "Help", show=False),
    ]

    def __init__(self, session: PingSession, engine: PingEngine) -> None:
        super().__init__()
        self.session = session
        self.engine = engine
        self._ping_tasks: dict[HostId, asyncio.Task[None]] = {}
        self._pending_updates: deque[PendingUpdate] = deque()
        self.stale_host_count = 0
        self.stale_generation_count = 0
        self.stale_seq_count = 0
        self._last_sort_refresh = 0.0
        self._last_fd_log = 0.0
        self._details_visible = False
        self._details_manually_toggled = False
        self._current_column_profile = "wide"
        self._pending_viewport_restore: tuple[float, float] | None = None
        self._viewport_restore_scheduled = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield HostTable(id="host-table")
        yield DetailsPanel(id="details-panel", classes="hidden-panel")
        yield Static("", id="status-strip")
        yield Footer()

    def on_mount(self) -> None:
        self.table = self.query_one(HostTable)
        self.details = self.query_one(DetailsPanel)
        self.status_strip = self.query_one("#status-strip", Static)
        self._apply_responsive_layout(force=True)
        self._sync_all_rows()
        self._refresh_selected_details()
        self._refresh_status_strip()
        self.set_interval(0.25, self.flush_updates)
        for host_id in list(self.session.hosts):
            self._start_ping_task(host_id)
        if self.session.selected_host_id:
            self.table.select_host(self.session.selected_host_id, scroll=False)

    async def on_unmount(self) -> None:
        await self._stop_all_ping_tasks()

    def on_resize(self, event: events.Resize) -> None:
        if not hasattr(self, "table"):
            return
        changed = self._apply_responsive_layout()
        if changed:
            self._sync_all_rows()
            self._refresh_status_strip()
            return
        self._sync_all_rows()

    def on_key(self, event: events.Key) -> None:
        character = event.character
        if character is None:
            return
        if character in self.SORT_HOTKEYS:
            self.action_sort_by(self.SORT_HOTKEYS[character])
            event.stop()

    @on(HostTable.RowHighlighted)
    def on_row_highlighted(self, event: HostTable.RowHighlighted) -> None:
        row_key = event.row_key.value if event.row_key else None
        self.session.select(str(row_key) if row_key is not None else None)
        self._refresh_selected_details()

    def action_focus_next(self) -> None:
        if self._details_visible and self.focused is self.table:
            self.details.focus()
        else:
            self.table.focus()

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_toggle_details(self) -> None:
        self._details_manually_toggled = True
        self._set_details_visible(not self._details_visible)
        self._sync_all_rows()
        self._refresh_status_strip()

    async def action_quit_session(self) -> None:
        await self._stop_all_ping_tasks()
        self.exit()

    def action_add_host(self) -> None:
        self.push_screen(HostFormScreen("Add host"), self._handle_add_host)

    def action_edit_selected(self) -> None:
        record = self.session.current_host()
        if record is None:
            self.notify("No host selected.", severity="warning")
            return
        self.push_screen(
            HostFormScreen("Edit host", value=record.config.target),
            lambda target: self._handle_edit_host(record.config.id, target),
        )

    def action_delete_selected(self) -> None:
        record = self.session.current_host()
        if record is None:
            self.notify("No host selected.", severity="warning")
            return
        self.push_screen(
            ConfirmScreen(f"Delete host '{record.config.target}'?"),
            lambda confirmed: self._handle_delete_host(record.config.id, confirmed),
        )

    def action_toggle_selected_pause(self) -> None:
        record = self.session.current_host()
        if record is None:
            self.notify("No host selected.", severity="warning")
            return
        self.session.toggle_host_pause(record.config.id)
        self._refresh_host(record.config.id)

    def action_toggle_all_pause(self) -> None:
        self.session.toggle_all_pause()
        for host_id in list(self.session.hosts):
            self._refresh_host(host_id)
        self._refresh_status_strip()
        self._refresh_selected_details()

    async def action_reset_selected(self) -> None:
        record = self.session.current_host()
        if record is None:
            self.notify("No host selected.", severity="warning")
            return
        host_id = record.config.id
        self.session.reset_host(host_id)
        await self._restart_ping_task(host_id)
        self._refresh_host(host_id)

    async def action_reset_all(self) -> None:
        self.session.reset_all()
        for host_id in list(self.session.hosts):
            await self._restart_ping_task(host_id)
        self._sync_all_rows()
        self._refresh_status_strip()

    def action_sort_by(self, column_key: str) -> None:
        sort_key = SortKey(column_key)
        if self.session.sort_key == sort_key:
            self.session.toggle_sort_order()
        else:
            self.session.set_sort(sort_key, reverse=False)
        self._sync_all_rows()
        self._refresh_status_strip()

    @on(PingSample)
    def on_ping_sample(self, message: PingSample) -> None:
        self._pending_updates.append(
            PendingUpdate(
                message.host_id, message.generation, message.seq, message.result
            )
        )

    def flush_updates(self) -> None:
        now = asyncio.get_running_loop().time()
        self._log_fd_usage(now)
        if not self._pending_updates:
            return
        touched: set[HostId] = set()
        while self._pending_updates:
            pending = self._pending_updates.popleft()
            if pending.host_id not in self.session.hosts:
                self.stale_host_count += 1
                continue
            status = self.session.accept_result(
                pending.host_id,
                pending.result,
                generation=pending.generation,
                seq=pending.seq,
            )
            if status is SampleStatus.ACCEPTED:
                touched.add(pending.host_id)
            elif status is SampleStatus.STALE_GENERATION:
                # Result was issued for an earlier target or pre-reset window.
                self.stale_generation_count += 1
            elif status is SampleStatus.STALE_SEQ:
                # Duplicate delivery or out-of-order within the generation.
                self.stale_seq_count += 1
        for host_id in touched:
            self._refresh_host(host_id)
        if touched and (now - self._last_sort_refresh) >= 1.0:
            self._last_sort_refresh = now
            self._sync_all_rows()
        self._refresh_status_strip()
        self._refresh_selected_details()

    def _refresh_host(self, host_id: HostId) -> None:
        if host_id not in self.session.hosts:
            return
        viewport = (self.table.scroll_x, self.table.scroll_y)
        self.table.upsert_host(self.session.host_snapshot(host_id))
        if self.session.selected_host_id == host_id:
            self._refresh_selected_details()
        self.table.select_host(self.session.selected_host_id, scroll=False)
        self._restore_table_viewport(*viewport)

    def _sync_all_rows(self) -> None:
        viewport = (self.table.scroll_x, self.table.scroll_y)
        self.table.set_column_profile(self._column_profile_for_width(self.size.width))
        self.table.sync_rows(self.session.host_snapshots())
        self.table.set_sort_indicator(self.session.sort_key, self.session.sort_reverse)
        self.table.select_host(self.session.selected_host_id, scroll=False)
        self._restore_table_viewport(*viewport)
        self._refresh_selected_details()

    def _refresh_selected_details(self) -> None:
        record = self.session.current_host()
        if self._details_visible:
            self.details.show_host(record.snapshot() if record else None)

    def _refresh_status_strip(self) -> None:
        aggregates = self.session.aggregates()
        self.status_strip.update(
            " | ".join(
                [
                    f"Hosts {aggregates['total_hosts']}",
                    f"Active {aggregates['active_hosts']}",
                    f"Paused {aggregates['paused_hosts']}",
                    f"Errors {aggregates['error_hosts']}",
                    f"Sent {aggregates['total_sent']}",
                    f"Lost {aggregates['total_lost']}",
                    f"Loss {float(cast(float, aggregates['loss_percent'])):.1f}%",
                    f"Sort {self.session.sort_key.value}",
                    "DESC" if self.session.sort_reverse else "ASC",
                    f"Details {'ON' if self._details_visible else 'OFF'}",
                ]
            )
        )

    def _handle_add_host(self, target: str | None) -> None:
        if target is None:
            return
        try:
            host_id = self.session.add_host(target)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        self._start_ping_task(host_id)
        self._sync_all_rows()

    @on(EditConfirmed)
    async def on_edit_confirmed(self, message: EditConfirmed) -> None:
        await self._commit_edit_host(message.host_id, message.target)

    @on(DeleteConfirmed)
    async def on_delete_confirmed(self, message: DeleteConfirmed) -> None:
        await self._commit_delete_host(message.host_id)

    def _handle_edit_host(self, host_id: HostId, target: str | None) -> None:
        if target is None:
            return
        self.post_message(EditConfirmed(host_id, target))

    async def _commit_edit_host(self, host_id: HostId, target: str) -> None:
        try:
            self.session.edit_host(host_id, target)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        # Bump already happened, but wait for the old probe's cancellation to be
        # confirmed before arming the new generation's loop.
        await self._restart_ping_task(host_id)
        self._refresh_host(host_id)
        self._refresh_status_strip()

    def _handle_delete_host(self, host_id: HostId, confirmed: bool | None) -> None:
        if not confirmed:
            return
        self.post_message(DeleteConfirmed(host_id))

    async def _commit_delete_host(self, host_id: HostId) -> None:
        # Wait for cancellation confirmation before the host disappears.
        await self._stop_ping_task(host_id)
        if host_id not in self.session.hosts:
            return
        self._discard_pending_updates(host_id)
        self.session.delete_host(host_id)
        self.table.remove_host(host_id)
        self.table.select_host(self.session.selected_host_id, scroll=False)
        self._refresh_selected_details()
        self._refresh_status_strip()

    def _discard_pending_updates(self, host_id: HostId) -> None:
        self._pending_updates = deque(
            update for update in self._pending_updates if update.host_id != host_id
        )

    def _start_ping_task(self, host_id: HostId) -> None:
        if host_id in self._ping_tasks:
            return
        self._ping_tasks[host_id] = asyncio.create_task(self._run_host_loop(host_id))

    async def _restart_ping_task(self, host_id: HostId) -> None:
        await self._stop_ping_task(host_id)
        self._start_ping_task(host_id)

    async def _stop_ping_task(self, host_id: HostId) -> None:
        task = self._ping_tasks.pop(host_id, None)
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _stop_all_ping_tasks(self) -> None:
        for host_id in list(self._ping_tasks):
            await self._stop_ping_task(host_id)

    def _apply_responsive_layout(self, force: bool = False) -> bool:
        changed = False
        profile = self._column_profile_for_width(self.size.width)
        if force or profile != self._current_column_profile:
            self._current_column_profile = profile
            changed = True
        return changed

    def _column_profile_for_width(self, width: int) -> str:
        if width >= 150:
            return "wide"
        if width >= 105:
            return "medium"
        return "narrow"

    def _set_details_visible(self, visible: bool) -> bool:
        if self._details_visible == visible:
            return False
        self._details_visible = visible
        self.details.set_class(not visible, "hidden-panel")
        return True

    def _restore_table_viewport(self, scroll_x: float, scroll_y: float) -> None:
        self._pending_viewport_restore = (scroll_x, scroll_y)
        self.table.scroll_to(
            x=scroll_x,
            y=scroll_y,
            immediate=True,
            force=True,
        )
        if self._viewport_restore_scheduled:
            return
        self._viewport_restore_scheduled = True
        self.call_after_refresh(self._flush_table_viewport_restore)

    def _flush_table_viewport_restore(self) -> None:
        self._viewport_restore_scheduled = False
        viewport = self._pending_viewport_restore
        self._pending_viewport_restore = None
        if viewport is None:
            return
        scroll_x, scroll_y = viewport
        self.table.scroll_to(
            x=scroll_x,
            y=scroll_y,
            immediate=True,
            force=True,
        )

    def _log_fd_usage(self, now: float) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if (now - self._last_fd_log) < 5.0:
            return
        self._last_fd_log = now
        fd_count = self._open_fd_count()
        if fd_count is None:
            logger.debug("fd_count unavailable")
            return
        logger.debug(
            "fd_count=%s hosts=%s pending_updates=%s",
            fd_count, len(self.session.hosts), len(self._pending_updates),
        )

    @staticmethod
    def _open_fd_count() -> int | None:
        for path in ("/dev/fd", "/proc/self/fd"):
            try:
                return len(os.listdir(path))
            except OSError:
                continue
        return None

    async def _run_host_loop(self, host_id: HostId) -> None:
        flag = int(host_id[:2], 16)
        record = self.session.hosts.get(host_id)
        if record is None:
            return
        # Pin the generation this loop is serving and number every probe it
        # issues strictly within that generation.
        generation = record.generation
        probe_seq: SampleSeq = 0
        try:
            while True:
                record = self.session.hosts.get(host_id)
                if record is None or record.generation != generation:
                    return
                if record.paused:
                    await asyncio.sleep(0.1)
                    continue
                probe_seq += 1
                target = record.config.target
                result = await self.engine.ping_once(
                    target=target,
                    timeout=self.session.config.timeout,
                    packet_size=self.session.config.packet_size,
                    flag=flag,
                )
                self.post_message(
                    PingSample(host_id, generation, probe_seq, result)
                )
                await asyncio.sleep(self.session.config.interval)
        except asyncio.CancelledError:
            raise
