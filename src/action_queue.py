#!/usr/bin/python

"""
A synchronization mechanism used to avoid re-entrancy and deadlock issues.
"""

from __future__ import division  # so 1/2 returns 0.5 instead of 0

class ActionQueue(object):
    """
    A synchronization mechanism used to avoid re-entrancy and deadlock issues.
    """

    def __init__(self):
        self._queue = []
        self._work_counter = 0

    def schedule(self, callback):
        """
        Adds work to be performed in order.

        :param callback: The thing to invoke to do the work.
        """
        self._queue.append(callback)
        self._work_counter += 1
        if self._work_counter > 1:
            return
        while self._work_counter > 0:
            c = self._queue.pop(0)
            c()
            self._work_counter -= 1
