import functools
import time
import types
from typing import Dict

import jax


class MeasureTime:
    decorators: Dict[str, "MeasureTime"] = {}

    def __init__(self, func):
        functools.wraps(func)(self)
        self.total_time = 0
        self.count = 0
        self.name = func.__qualname__
        self.enabled = True
        MeasureTime.decorators[self.name] = self

    def __call__(self, *args, **kwargs):
        if self.enabled:
            start_time = time.time()
            result = self.__wrapped__(*args, **kwargs)
            jax.block_until_ready(result)
            end_time = time.time()
            self.total_time += end_time - start_time
            self.count += 1
            return result
        else:
            return self.__wrapped__(*args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return types.MethodType(self, instance)

    def average_time(self):
        if self.count == 0:
            return 0
        return self.total_time / self.count

    def _reset(self):
        self.total_time = 0
        self.count = 0

    @classmethod
    def reset(cls):
        for name, decorator in cls.decorators.items():
            decorator._reset()

    def _disable(self):
        self.enabled = False

    @classmethod
    def disable(cls):
        for name, decorator in cls.decorators.items():
            decorator._disable()

    def _enable(self):
        self.enabled = True

    @classmethod
    def enable(cls):
        for name, decorator in cls.decorators.items():
            decorator._enable()

    @classmethod
    def print_stats(cls):
        """Print timing statistics for all decorated functions"""
        longest_name_len = max(
            [len(name) for name in cls.decorators.keys()], default=len("Name")
        )

        print(f"{'Name':<{longest_name_len}}\tAvg Time\tCalls")

        sorted_decorators = sorted(
            cls.decorators.items(), key=lambda x: x[1].average_time(), reverse=True
        )

        for name, decorator in sorted_decorators:
            if decorator.count > 0:
                avg_time = decorator.average_time()
                format_str = "{name:<{width}}\t{avg_time:.6f}\t{count}"
                print(
                    format_str.format(
                        name=name,
                        avg_time=avg_time,
                        count=decorator.count,
                        width=longest_name_len,
                    )
                )

        return longest_name_len
