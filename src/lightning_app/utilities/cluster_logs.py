import dataclasses
import json
import queue
from dataclasses import dataclass
from threading import Thread
from typing import Callable, Iterator, List, Optional

import dateutil.parser
from websocket import WebSocketApp

from lightning_app.utilities.log_helpers import _error_callback, _OrderedLogEntry
from lightning_app.utilities.logs_socket_api import _ClusterLogsSocketAPI
from lightning_app.utilities.network import LightningClient


class _ClusterLogEventLabels:
    cluster_id: str
    grid_url: str
    hostname: str
    level: str
    logger: str

    def __init__(self, **kwargs):
        self._fields = set()
        for k, v in kwargs.items():
            self.__dict__[k] = v
            self._fields.add(k)

    def __eq__(self, other: "_ClusterLogEventLabels"):
        for field in self._fields | other._fields:
            if not hasattr(self, field) or not hasattr(other, field):
                return False
            if getattr(self, field) != getattr(other, field):
                return False
        return True


@dataclass
class _ClusterLogEvent(_OrderedLogEntry):
    labels: _ClusterLogEventLabels


def _push_log_events_to_read_queue_callback(read_queue: queue.PriorityQueue):
    """Pushes _LogEvents from websocket to read_queue.

    Returns callback function used with `on_message_callback` of websocket.WebSocketApp.
    """

    def callback(_: WebSocketApp, msg: str):
        for ev in _parse_log_event(msg):
            read_queue.put(ev)

    return callback


def _parse_log_event(msg: str) -> List[_ClusterLogEvent]:
    # We strongly trust that the contract on API will hold atm :D
    event_dict = json.loads(msg)
    labels = _ClusterLogEventLabels(**event_dict["labels"])
    log_events = []

    if "message" in event_dict:
        message = event_dict["message"]
        timestamp = dateutil.parser.isoparse(event_dict["timestamp"])
        event = _ClusterLogEvent(
            message=message,
            timestamp=timestamp,
            labels=labels,
        )
        log_events.append(event)
    return log_events


def _cluster_logs_reader(
    logs_api_client: _ClusterLogsSocketAPI,
    cluster_id: str,
    start: int,
    end: int,
    limit: int,
    follow: bool,
    batch_size: int = 5000,
) -> Iterator[_ClusterLogEvent]:
    read_queue = queue.PriorityQueue()
    items_read = 0

    # We will use a socket inside a thread to read logs,
    # to follow our typical reading pattern

    # helper function which will start logs streams to the read_queue from the start onwards, till the end
    def start_logs(start: int) -> Callable:
        log_socket = logs_api_client.create_cluster_logs_socket(
            cluster_id=cluster_id,
            start=start,
            end=end,
            limit=min(limit - items_read, batch_size),
            on_message_callback=_push_log_events_to_read_queue_callback(read_queue),
            on_error_callback=lambda _, ex: read_queue.put(ex),
        )

        log_thread = Thread(target=log_socket.run_forever)

        # Establish connection and begin pushing logs to the queue
        log_thread.start()

        def stop():
            # Close connection - it will cause run_forever() to finish -> thread as finishes as well
            log_socket.close()

            # The socket was closed, we can just wait for thread to finish.
            log_thread.join()

        return stop

    stop_fn = start_logs(start)
    # Print logs from queue when log event is available
    try:
        items_remaining_in_batch = batch_size
        while True:
            log_event: _ClusterLogEvent = read_queue.get(timeout=None if follow else 1.0)

            # Exception happened during queue processing
            if isinstance(log_event, Exception):
                raise log_event

            yield log_event

            items_read += 1
            if items_read == limit:
                # We've read enough entries, just terminate and close the connection
                break

            items_remaining_in_batch -= 1
            if items_remaining_in_batch == 0:
                stop_fn()
                start_logs(int(log_event.timestamp.timestamp()))
                items_remaining_in_batch = batch_size

    except queue.Empty:
        # Empty is raised by queue.get if timeout is reached. Follow = False case.
        pass

    except KeyboardInterrupt:
        # User pressed CTRL+C to exit, we should respect that
        pass

    finally:
        stop_fn()
