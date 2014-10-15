#!/usr/bin/python

"""
A value that can be watched.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0
from action_queue import ActionQueue


class ObservableValue(object):
    """
    A value that can be watched.
    """

    def __init__(self, initialValue):
        self._old = initialValue
        self._value = initialValue
        self._subscribers = []
        self._queue = ActionQueue()

    def _watch_helper(self, callback):
        self._subscribers.append(callback)
        callback(self._value)

    def _update_helper(self, new_value):
        if self._value == new_value:
            return
        self._value = new_value
        for callback in self._subscribers:
            callback(new_value)
        self._old = new_value

    def watch(self, callback):
        """
        Adds a method to be invoked with the current value and new values whenever it changes.
        :param callback: The single argument method to invoke.
        """
        self._queue.schedule(lambda: self._watch_helper(callback))

    def update(self, new_value):
        """
        Schedules a new value to be assigned to the observable and reported to the outside.
        :param new_value: The value to assign.
        """
        self._queue.schedule(lambda: self._update_helper(new_value))

    def get(self):
        """
        Returns the current value of this observable.
        Scheduled values are not visible until previously scheduled actions finish.
        """
        return self._value

    def old(self):
        """
        Hackily returns the previous value of the observable, but only during a callback.
        """
        return self._old

