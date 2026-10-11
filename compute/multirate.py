from dataclasses import dataclass
from typing import Callable


@dataclass
class ScheduledTask:
    name: str
    requested_interval_s: float
    interval_ticks: int
    actual_interval_s: float
    callback: Callable
    calls: int = 0


class MultiRateScheduler:
    """
    Stateful multirate simulation scheduler.

    Every subsystem runs at an integer multiple of one
    global base timestep.

    The scheduler keeps track of its current tick so repeated
    calls to run() continue the same simulation clock.
    """

    def __init__(
        self,
        base_dt_s: float,
    ):

        if base_dt_s <= 0:
            raise ValueError(
                "base_dt_s must be > 0"
            )

        self.base_dt_s = float(
            base_dt_s
        )

        self.tasks = []

        self.current_tick = 0


    @property
    def current_time_s(self):

        return (
            self.current_tick
            * self.base_dt_s
        )


    def add_task(
        self,
        name: str,
        interval_s: float,
        callback: Callable,
    ):

        if interval_s <= 0:
            raise ValueError(
                "interval_s must be > 0"
            )

        ticks = max(
            1,
            round(
                interval_s
                / self.base_dt_s
            )
        )

        actual_interval_s = (
            ticks
            * self.base_dt_s
        )

        task = ScheduledTask(
            name=name,
            requested_interval_s=interval_s,
            interval_ticks=ticks,
            actual_interval_s=actual_interval_s,
            callback=callback,
        )

        self.tasks.append(
            task
        )

        return task


    def reset_counts(self):

        for task in self.tasks:
            task.calls = 0


    def reset(self):

        self.current_tick = 0

        self.reset_counts()


    def run(
        self,
        biological_seconds: float,
    ):

        if biological_seconds < 0:
            raise ValueError(
                "biological_seconds must be >= 0"
            )

        ticks_to_run = round(
            biological_seconds
            / self.base_dt_s
        )


        for _ in range(
            ticks_to_run
        ):

            self.current_tick += 1

            tick = (
                self.current_tick
            )


            for task in self.tasks:

                if (
                    tick
                    % task.interval_ticks
                    == 0
                ):

                    task.callback(
                        task.actual_interval_s,
                        tick,
                    )

                    task.calls += 1


        return ticks_to_run