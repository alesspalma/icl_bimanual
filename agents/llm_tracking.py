import time
import threading
import numpy as np


class LLMTrackingMixin:
    """Mixin for tracking LLM call statistics (tokens, latency) per episode."""

    def _init_tracking(self):
        self._tracking_enabled = True
        self._tracking_lock = threading.Lock()
        self._parallel_factor = getattr(self, '_parallel_factor', 1)
        self._current_ep_stats = {
            'calls': 0, 'prompt_tokens': 0,
            'completion_tokens': 0, 'total_tokens': 0,
            'wall_time': 0.0
        }
        self._all_ep_stats = []

    def _record_call(self, usage, elapsed):
        with self._tracking_lock:
            self._current_ep_stats['calls'] += 1
            self._current_ep_stats['wall_time'] += elapsed
            if usage:
                self._current_ep_stats['prompt_tokens'] += usage['prompt_tokens']
                self._current_ep_stats['completion_tokens'] += usage['completion_tokens']
                self._current_ep_stats['total_tokens'] += usage['total_tokens']

    def _finalize_episode_stats(self):
        if hasattr(self, '_tracking_enabled') and self._tracking_enabled:
            if self._current_ep_stats['calls'] > 0:
                if self._parallel_factor > 1:
                    self._current_ep_stats['wall_time'] /= self._parallel_factor
                self._all_ep_stats.append(dict(self._current_ep_stats))
            self._current_ep_stats = {
                'calls': 0, 'prompt_tokens': 0,
                'completion_tokens': 0, 'total_tokens': 0,
                'wall_time': 0.0
            }

    def _wrap_llm_call_with_tracking(self, raw_call):
        """Wrap a raw_call that returns (content, usage) to add tracking."""
        def tracked_call(messages):
            t0 = time.monotonic()
            content, usage = raw_call(messages)
            elapsed = time.monotonic() - t0
            self._record_call(usage, elapsed)
            return content
        return tracked_call

    def _setup_tracking_if_enabled(self, raw_call):
        """Call at end of load_weights to wrap self.llm_call with optional tracking."""
        if getattr(self.model_config, 'track_llm_stats', False):
            self._init_tracking()
            self.llm_call = self._wrap_llm_call_with_tracking(raw_call)
        else:
            self._tracking_enabled = False
            self.llm_call = lambda messages: raw_call(messages)[0]

    def get_raw_episode_stats(self):
        """Return list of per-episode stat dicts. Auto-finalizes current episode."""
        if not hasattr(self, '_tracking_enabled') or not self._tracking_enabled:
            return []
        self._finalize_episode_stats()
        return list(self._all_ep_stats)
