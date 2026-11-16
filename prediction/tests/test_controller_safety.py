"""Day 9 — Predictive Scaling Controller Safety Tests.

Focused tests for replica limits, invalid forecast handling,
scale-up / scale-down / hold decisions, cooldown boundary precision,
ScalingConfig field validation, and reconcile edge-cases.

Rules:
- All tests use synthetic inputs only.
- No dataset files, saved models, prediction outputs, metrics, or graphs
  are read or written.
- dry_run=True for all ScalingConfig instances unless the specific test
  under examination requires dry_run=False to verify live-patch behaviour.
- No Kubernetes cluster is contacted; a FakeAppsApi stub is used throughout.
"""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from prediction.src.config import ScalingConfig
from prediction.src.controller import ControllerResult, PredictiveScalingController
from prediction.src.safety_rules import SafetyDecision, apply_safety_rules
from prediction.src.scaling_logic import calculate_desired_replicas


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cfg(
    capacity=100.0,
    min_replicas=1,
    max_replicas=10,
    cooldown_seconds=300,
    max_scale_up_step=3,
    max_scale_down_step=2,
    max_forecast_age_seconds=120,
    dry_run=True,
):
    """Return a ScalingConfig with explicit capacity and sensible defaults."""
    return ScalingConfig(
        capacity_rps_per_replica=capacity,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        cooldown_seconds=cooldown_seconds,
        max_scale_up_step=max_scale_up_step,
        max_scale_down_step=max_scale_down_step,
        max_forecast_age_seconds=max_forecast_age_seconds,
        dry_run=dry_run,
    )


def _now():
    return datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _recent_forecast():
    """A forecast timestamp 10 s before _now()."""
    return _now() - timedelta(seconds=10)


class FakeAppsApi:
    """Minimal Kubernetes AppsV1Api stub used throughout Day 9 tests."""

    def __init__(self, current_replicas=2, fail_read=False, fail_patch=False):
        self.current_replicas = current_replicas
        self.fail_read = fail_read
        self.fail_patch = fail_patch
        self.patches = []
        self.reads = []

    def read_namespaced_deployment_scale(self, name, namespace):
        self.reads.append((name, namespace))
        if self.fail_read:
            raise RuntimeError("simulated read failure")
        return SimpleNamespace(spec=SimpleNamespace(replicas=self.current_replicas))

    def patch_namespaced_deployment_scale(self, name, namespace, body):
        if self.fail_patch:
            raise RuntimeError("simulated patch failure")
        self.patches.append((name, namespace, body))


# ---------------------------------------------------------------------------
# 1. Replica minimum and maximum floors
# ---------------------------------------------------------------------------

class ReplicaBoundTests(unittest.TestCase):
    """Task 2 -- replica minimum and maximum limits."""

    def test_zero_forecast_is_clamped_to_min_replicas(self):
        cfg = _cfg(capacity=100, min_replicas=2)
        self.assertEqual(calculate_desired_replicas(0.0, cfg), 2)

    def test_tiny_positive_forecast_is_clamped_to_min_replicas(self):
        cfg = _cfg(capacity=100, min_replicas=3)
        self.assertEqual(calculate_desired_replicas(0.001, cfg), 3)

    def test_exact_capacity_gives_min_one_replica(self):
        cfg = _cfg(capacity=100, min_replicas=1)
        self.assertEqual(calculate_desired_replicas(100.0, cfg), 1)

    def test_one_above_capacity_ceiling_rounds_up(self):
        cfg = _cfg(capacity=100, min_replicas=1)
        self.assertEqual(calculate_desired_replicas(101.0, cfg), 2)

    def test_enormous_forecast_is_clamped_to_max_replicas(self):
        cfg = _cfg(capacity=100, max_replicas=5)
        self.assertEqual(calculate_desired_replicas(1_000_000.0, cfg), 5)

    def test_forecast_exactly_at_max_capacity_hits_max_replicas(self):
        cfg = _cfg(capacity=100, max_replicas=10)
        self.assertEqual(calculate_desired_replicas(1000.0, cfg), 10)

    def test_forecast_one_above_max_capacity_is_clamped(self):
        cfg = _cfg(capacity=100, max_replicas=10)
        self.assertEqual(calculate_desired_replicas(1001.0, cfg), 10)

    def test_min_equals_max_always_returns_that_replica_count(self):
        cfg = _cfg(capacity=100, min_replicas=4, max_replicas=4)
        for rate in (0.0, 50.0, 500.0, 50_000.0):
            with self.subTest(rate=rate):
                self.assertEqual(calculate_desired_replicas(rate, cfg), 4)

    def test_fractional_capacity_rounds_up_correctly(self):
        cfg = _cfg(capacity=50.0, min_replicas=1)
        self.assertEqual(calculate_desired_replicas(75.0, cfg), 2)
        self.assertEqual(calculate_desired_replicas(50.0, cfg), 1)


# ---------------------------------------------------------------------------
# 2. Invalid, negative, NaN, and infinite forecast values
# ---------------------------------------------------------------------------

class InvalidForecastTests(unittest.TestCase):
    """Task 3 -- invalid inputs to calculate_desired_replicas."""

    def setUp(self):
        self.cfg = _cfg()

    def test_nan_forecast_raises_value_error(self):
        with self.assertRaises(ValueError):
            calculate_desired_replicas(float("nan"), self.cfg)

    def test_positive_inf_forecast_raises_value_error(self):
        with self.assertRaises(ValueError):
            calculate_desired_replicas(float("inf"), self.cfg)

    def test_negative_inf_forecast_raises_value_error(self):
        with self.assertRaises(ValueError):
            calculate_desired_replicas(float("-inf"), self.cfg)

    def test_negative_float_raises_value_error(self):
        with self.assertRaises(ValueError):
            calculate_desired_replicas(-0.001, self.cfg)

    def test_negative_integer_raises_value_error(self):
        with self.assertRaises(ValueError):
            calculate_desired_replicas(-1, self.cfg)

    def test_boolean_true_raises_value_error(self):
        with self.assertRaises(ValueError):
            calculate_desired_replicas(True, self.cfg)

    def test_boolean_false_raises_value_error(self):
        with self.assertRaises(ValueError):
            calculate_desired_replicas(False, self.cfg)

    def test_numeric_string_is_accepted_and_converted(self):
        # float("200") succeeds, so the source code does not reject numeric
        # strings; it converts them.  ceil(200/100)=2, clamped to [1,10].
        result = calculate_desired_replicas("200", self.cfg)
        self.assertEqual(result, 2)

    def test_non_numeric_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            calculate_desired_replicas("two-hundred", self.cfg)

    def test_none_raises_exception(self):
        with self.assertRaises((ValueError, TypeError)):
            calculate_desired_replicas(None, self.cfg)

    def test_list_raises_exception(self):
        with self.assertRaises((ValueError, TypeError)):
            calculate_desired_replicas([100.0], self.cfg)


# ---------------------------------------------------------------------------
# 3. Controller.decide() -- None / invalid types hold
# ---------------------------------------------------------------------------

class DecideHoldTests(unittest.TestCase):
    """Task 3/4 -- controller.decide() returns hold for missing/bad forecasts."""

    def setUp(self):
        self.cfg = _cfg(min_replicas=1, max_replicas=10)
        self.controller = PredictiveScalingController(self.cfg)
        self.now = _now()
        self.ft = _recent_forecast()

    def _decide(self, forecast):
        return self.controller.decide(forecast, 3, self.ft, now=self.now)

    def test_none_forecast_returns_hold_at_current_replicas(self):
        d = self._decide(None)
        self.assertIsInstance(d, SafetyDecision)
        self.assertEqual(d.target_replicas, 3)
        self.assertFalse(d.allowed)
        self.assertIn("Missing prediction", d.reason)

    def test_nan_forecast_returns_hold_at_current_replicas(self):
        d = self._decide(float("nan"))
        self.assertEqual(d.target_replicas, 3)
        self.assertFalse(d.allowed)
        self.assertIn("Invalid prediction", d.reason)

    def test_positive_inf_returns_hold(self):
        d = self._decide(float("inf"))
        self.assertEqual(d.target_replicas, 3)
        self.assertFalse(d.allowed)

    def test_negative_inf_returns_hold(self):
        d = self._decide(float("-inf"))
        self.assertEqual(d.target_replicas, 3)
        self.assertFalse(d.allowed)

    def test_negative_float_returns_hold(self):
        d = self._decide(-0.5)
        self.assertEqual(d.target_replicas, 3)
        self.assertFalse(d.allowed)

    def test_boolean_true_returns_hold(self):
        d = self._decide(True)
        self.assertEqual(d.target_replicas, 3)
        self.assertFalse(d.allowed)

    def test_boolean_false_returns_hold(self):
        d = self._decide(False)
        self.assertEqual(d.target_replicas, 3)
        self.assertFalse(d.allowed)

    def test_numeric_string_forecast_is_accepted(self):
        # "100" converts via float() to 100.0; ceil(100/100)=1, step bounds
        # bring target from desired=1 up to max(1, 3-2)=1; allowed=True.
        # This documents actual code behaviour, not a hold.
        d = self._decide("100")
        self.assertTrue(d.allowed)
        self.assertGreaterEqual(d.target_replicas, 1)

    def test_non_numeric_string_forecast_returns_hold(self):
        d = self._decide("not-a-number")
        self.assertEqual(d.target_replicas, 3)
        self.assertFalse(d.allowed)

    def test_object_forecast_returns_hold(self):
        d = self._decide(object())
        self.assertEqual(d.target_replicas, 3)
        self.assertFalse(d.allowed)


# ---------------------------------------------------------------------------
# 4. Controller.decide() -- scale-up, scale-down, hold
# ---------------------------------------------------------------------------

class DecideScaleDecisionTests(unittest.TestCase):
    """Task 5 -- scale-up, scale-down, hold with controlled synthetic inputs."""

    def setUp(self):
        self.cfg = _cfg(
            capacity=100,
            min_replicas=1,
            max_replicas=10,
            max_scale_up_step=2,
            max_scale_down_step=1,
            cooldown_seconds=0,
            max_forecast_age_seconds=300,
        )
        self.controller = PredictiveScalingController(self.cfg)
        self.now = _now()
        self.ft = _recent_forecast()

    def _decide(self, forecast_rps, current=4):
        return self.controller.decide(forecast_rps, current, self.ft, now=self.now)

    def test_scale_up_within_step_limit(self):
        """500 RPS -> desired=5; current=4 -> target=5 (within +2 step)."""
        d = self._decide(500.0, current=4)
        self.assertTrue(d.allowed)
        self.assertEqual(d.requested_replicas, 5)
        self.assertEqual(d.target_replicas, 5)

    def test_scale_up_exceeds_step_limit_is_bounded(self):
        """1000 RPS -> desired=10; current=4 -> bounded target=min(10,4+2)=6."""
        d = self._decide(1000.0, current=4)
        self.assertTrue(d.allowed)
        self.assertEqual(d.requested_replicas, 10)
        self.assertEqual(d.target_replicas, 6)

    def test_scale_up_bounded_by_max_replicas(self):
        """1000 RPS -> desired=10; current=9 -> bounded target=min(10,9+2)=10."""
        d = self._decide(1000.0, current=9)
        self.assertTrue(d.allowed)
        self.assertEqual(d.target_replicas, 10)

    def test_scale_down_within_step_limit(self):
        """100 RPS -> desired=1; current=4 -> bounded target=max(1,4-1)=3."""
        d = self._decide(100.0, current=4)
        self.assertTrue(d.allowed)
        self.assertEqual(d.requested_replicas, 1)
        self.assertEqual(d.target_replicas, 3)

    def test_scale_down_bounded_by_min_replicas(self):
        """0 RPS -> desired=1 (clamped); current=2 -> target=max(1,2-1)=1."""
        d = self._decide(0.0, current=2)
        self.assertTrue(d.allowed)
        self.assertEqual(d.target_replicas, 1)

    def test_scale_down_cannot_go_below_min(self):
        cfg = _cfg(
            capacity=100,
            min_replicas=3,
            max_replicas=10,
            max_scale_down_step=10,
            cooldown_seconds=0,
        )
        controller = PredictiveScalingController(cfg)
        d = controller.decide(0.0, 5, self.ft, now=self.now)
        self.assertTrue(d.allowed)
        self.assertGreaterEqual(d.target_replicas, 3)

    def test_hold_when_desired_equals_current(self):
        """400 RPS -> desired=4 == current=4 -> target=4, allowed=True."""
        d = self._decide(400.0, current=4)
        self.assertTrue(d.allowed)
        self.assertEqual(d.target_replicas, 4)
        self.assertEqual(d.requested_replicas, 4)


# ---------------------------------------------------------------------------
# 5. Cooldown boundary precision
# ---------------------------------------------------------------------------

class CooldownBoundaryTests(unittest.TestCase):
    """Task 4 -- exact cooldown boundary: at boundary=allowed, inside=blocked."""

    def setUp(self):
        self.cfg = _cfg(cooldown_seconds=300)
        self.now = _now()
        self.ft = _recent_forecast()

    def _apply(self, elapsed_seconds, desired=5, current=3):
        last_scale = self.now - timedelta(seconds=elapsed_seconds)
        return apply_safety_rules(
            desired, current, self.cfg,
            forecast_timestamp=self.ft,
            last_scale_time=last_scale,
            now=self.now,
        )

    def test_exactly_at_cooldown_boundary_is_allowed(self):
        d = self._apply(300)
        self.assertTrue(d.allowed)

    def test_one_second_inside_cooldown_is_blocked(self):
        d = self._apply(299)
        self.assertFalse(d.allowed)
        self.assertIn("Cooldown active", d.reason)

    def test_one_second_past_cooldown_is_allowed(self):
        d = self._apply(301)
        self.assertTrue(d.allowed)

    def test_no_change_during_cooldown_is_allowed(self):
        d = self._apply(1, desired=3, current=3)
        self.assertTrue(d.allowed)
        self.assertEqual(d.target_replicas, 3)

    def test_cooldown_reason_includes_elapsed_seconds(self):
        d = self._apply(100, desired=5, current=3)
        self.assertIn("100", d.reason)


# ---------------------------------------------------------------------------
# 6. Forecast-age boundary precision
# ---------------------------------------------------------------------------

class ForecastAgeBoundaryTests(unittest.TestCase):
    """Task 4 -- forecast age boundary: at boundary=allowed, exceeded=blocked."""

    def setUp(self):
        self.cfg = _cfg(max_forecast_age_seconds=120)
        self.now = _now()

    def _apply(self, age_seconds, desired=5, current=3):
        ft = self.now - timedelta(seconds=age_seconds)
        return apply_safety_rules(
            desired, current, self.cfg,
            forecast_timestamp=ft, now=self.now,
        )

    def test_exactly_at_age_limit_is_allowed(self):
        self.assertTrue(self._apply(120).allowed)

    def test_one_second_past_age_limit_is_blocked(self):
        d = self._apply(121)
        self.assertFalse(d.allowed)
        self.assertIn("stale", d.reason)

    def test_very_fresh_forecast_is_allowed(self):
        self.assertTrue(self._apply(1).allowed)

    def test_exactly_current_timestamp_is_allowed(self):
        self.assertTrue(self._apply(0).allowed)


# ---------------------------------------------------------------------------
# 7. ScalingConfig field validation
# ---------------------------------------------------------------------------

class ScalingConfigValidationTests(unittest.TestCase):
    """Task 3/4 -- ScalingConfig rejects all invalid field values."""

    def test_boolean_capacity_is_rejected(self):
        for b in (True, False):
            with self.subTest(b=b):
                with self.assertRaises(ValueError):
                    ScalingConfig(capacity_rps_per_replica=b)

    def test_zero_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=0)

    def test_negative_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=-10.0)

    def test_nan_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=float("nan"))

    def test_inf_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=float("inf"))

    def test_min_replicas_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=100, min_replicas=0)

    def test_max_replicas_below_min_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(
                capacity_rps_per_replica=100, min_replicas=5, max_replicas=4
            )

    def test_boolean_min_replicas_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=100, min_replicas=True)

    def test_zero_scale_up_step_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=100, max_scale_up_step=0)

    def test_zero_scale_down_step_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=100, max_scale_down_step=0)

    def test_negative_cooldown_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=100, cooldown_seconds=-1)

    def test_negative_forecast_age_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=100, max_forecast_age_seconds=-1)

    def test_non_boolean_dry_run_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=100, dry_run=1)

    def test_empty_capacity_basis_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalingConfig(capacity_rps_per_replica=100, capacity_basis="   ")

    def test_valid_config_has_dry_run_true_by_default(self):
        cfg = ScalingConfig(capacity_rps_per_replica=50.0)
        self.assertTrue(cfg.dry_run)

    def test_valid_config_with_all_explicit_fields(self):
        cfg = ScalingConfig(
            capacity_rps_per_replica=200.0,
            capacity_basis="measured load test 2026-10-28",
            min_replicas=2,
            max_replicas=20,
            cooldown_seconds=60,
            max_scale_up_step=4,
            max_scale_down_step=2,
            max_forecast_age_seconds=90,
            dry_run=True,
        )
        self.assertEqual(cfg.capacity_rps_per_replica, 200.0)
        self.assertEqual(cfg.min_replicas, 2)
        self.assertTrue(cfg.dry_run)


# ---------------------------------------------------------------------------
# 8. ControllerResult field content and dry-run invariant
# ---------------------------------------------------------------------------

class ControllerResultContentTests(unittest.TestCase):
    """Task 5 -- verify ControllerResult fields from reconcile()."""

    def setUp(self):
        self.now = _now()
        self.ft = _recent_forecast()

    def _reconcile(self, forecast_rps, current_replicas=2, dry_run=True):
        cfg = _cfg(
            capacity=100,
            max_scale_up_step=3,
            max_scale_down_step=2,
            cooldown_seconds=0,
            max_forecast_age_seconds=300,
            dry_run=dry_run,
        )
        api = FakeAppsApi(current_replicas=current_replicas)
        controller = PredictiveScalingController(cfg, apps_api=api)
        result = controller.reconcile(
            "svc", "default", forecast_rps, self.ft, now=self.now
        )
        return result, api

    def test_dry_run_result_has_correct_action_and_current_replicas(self):
        result, api = self._reconcile(500.0, current_replicas=2)
        self.assertEqual(result.action, "dry-run")
        self.assertEqual(result.current_replicas, 2)
        self.assertIsNotNone(result.decision)
        # ceil(500/100)=5; current=2; bounded: min(10, 2+3)=5 -> target=5
        self.assertEqual(result.decision.target_replicas, 5)
        self.assertEqual(api.patches, [])

    def test_no_change_needed_returns_no_op(self):
        """200 RPS, capacity=100, current=2 -> desired=2 == current -> no-op."""
        result, api = self._reconcile(200.0, current_replicas=2)
        self.assertEqual(result.action, "no-op")
        self.assertEqual(result.current_replicas, 2)
        self.assertEqual(api.patches, [])

    def test_missing_forecast_reconcile_returns_no_op(self):
        result, api = self._reconcile(None, current_replicas=3)
        self.assertEqual(result.action, "no-op")
        self.assertIsNotNone(result.decision)
        self.assertFalse(result.decision.allowed)
        self.assertEqual(api.patches, [])

    def test_reconcile_message_is_non_empty_string(self):
        for forecast in (None, 500.0, float("nan")):
            with self.subTest(forecast=forecast):
                result, _ = self._reconcile(forecast, current_replicas=3)
                self.assertIsInstance(result.message, str)
                self.assertTrue(result.message.strip())

    def test_empty_deployment_name_returns_error(self):
        api = FakeAppsApi(current_replicas=2)
        cfg = _cfg()
        controller = PredictiveScalingController(cfg, apps_api=api)
        result = controller.reconcile("", "default", 500.0, self.ft, now=self.now)
        self.assertEqual(result.action, "error")
        self.assertIsNone(result.decision)
        self.assertEqual(api.patches, [])

    def test_empty_namespace_returns_error(self):
        api = FakeAppsApi(current_replicas=2)
        cfg = _cfg()
        controller = PredictiveScalingController(cfg, apps_api=api)
        result = controller.reconcile("svc", "", 500.0, self.ft, now=self.now)
        self.assertEqual(result.action, "error")
        self.assertIsNone(result.decision)
        self.assertEqual(api.patches, [])

    def test_dry_run_true_never_patches_for_any_rate(self):
        for rps in (100.0, 500.0, 1000.0):
            with self.subTest(rps=rps):
                result, api = self._reconcile(rps, current_replicas=2, dry_run=True)
                self.assertEqual(api.patches, [], msg="Unexpected patch at %s RPS" % rps)


# ---------------------------------------------------------------------------
# 9. apply_safety_rules -- desired == current hold
# ---------------------------------------------------------------------------

class SafetyRulesHoldTests(unittest.TestCase):
    """Task 5 -- hold when desired equals current."""

    def setUp(self):
        self.cfg = _cfg(cooldown_seconds=0, max_forecast_age_seconds=300)
        self.now = _now()
        self.ft = _recent_forecast()

    def test_desired_equals_current_returns_allowed(self):
        d = apply_safety_rules(
            3, 3, self.cfg, forecast_timestamp=self.ft, now=self.now
        )
        self.assertTrue(d.allowed)
        self.assertEqual(d.target_replicas, 3)
        self.assertEqual(d.requested_replicas, 3)

    def test_desired_equals_current_at_min_replicas(self):
        cfg = _cfg(min_replicas=2, max_replicas=10, cooldown_seconds=0)
        d = apply_safety_rules(
            2, 2, cfg, forecast_timestamp=self.ft, now=self.now
        )
        self.assertTrue(d.allowed)
        self.assertEqual(d.target_replicas, 2)

    def test_desired_equals_current_at_max_replicas(self):
        cfg = _cfg(min_replicas=1, max_replicas=10, cooldown_seconds=0)
        d = apply_safety_rules(
            10, 10, cfg, forecast_timestamp=self.ft, now=self.now
        )
        self.assertTrue(d.allowed)
        self.assertEqual(d.target_replicas, 10)


# ---------------------------------------------------------------------------
# 10. Missing and stale workload inputs
# ---------------------------------------------------------------------------

class StaleAndMissingInputTests(unittest.TestCase):
    """Task 4 -- missing or stale forecast inputs always hold."""

    def setUp(self):
        self.cfg = _cfg(max_forecast_age_seconds=60)
        self.now = _now()

    def test_none_forecast_timestamp_holds_current(self):
        d = apply_safety_rules(
            6, 3, self.cfg, forecast_timestamp=None, now=self.now
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.target_replicas, 3)

    def test_stale_forecast_holds_current(self):
        stale = self.now - timedelta(seconds=61)
        d = apply_safety_rules(
            6, 3, self.cfg, forecast_timestamp=stale, now=self.now
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.target_replicas, 3)

    def test_future_forecast_timestamp_holds_current(self):
        future = self.now + timedelta(seconds=1)
        d = apply_safety_rules(
            6, 3, self.cfg, forecast_timestamp=future, now=self.now
        )
        self.assertFalse(d.allowed)
        self.assertEqual(d.target_replicas, 3)

    def test_stale_reason_mentions_age(self):
        stale = self.now - timedelta(seconds=90)
        d = apply_safety_rules(
            6, 3, self.cfg, forecast_timestamp=stale, now=self.now
        )
        self.assertIn("stale", d.reason)
        self.assertIn("90", d.reason)


# ---------------------------------------------------------------------------
# 11. controller.decide() -- invalid current_replicas
# ---------------------------------------------------------------------------

class DecideInvalidCurrentReplicaTests(unittest.TestCase):
    """Task 4 -- decide() raises ValueError for invalid current_replicas."""

    def setUp(self):
        self.cfg = _cfg(min_replicas=1, max_replicas=10)
        self.controller = PredictiveScalingController(self.cfg)
        self.ft = _recent_forecast()
        self.now = _now()

    def test_boolean_current_replicas_raises(self):
        with self.assertRaises(ValueError):
            self.controller.decide(None, True, self.ft, now=self.now)

    def test_float_current_replicas_raises(self):
        with self.assertRaises(ValueError):
            self.controller.decide(None, 2.5, self.ft, now=self.now)

    def test_string_current_replicas_raises(self):
        with self.assertRaises(ValueError):
            self.controller.decide(None, "3", self.ft, now=self.now)

    def test_below_min_current_replicas_raises(self):
        with self.assertRaises(ValueError):
            self.controller.decide(None, 0, self.ft, now=self.now)

    def test_above_max_current_replicas_raises(self):
        with self.assertRaises(ValueError):
            self.controller.decide(None, 11, self.ft, now=self.now)

    def test_negative_current_replicas_raises(self):
        with self.assertRaises(ValueError):
            self.controller.decide(None, -1, self.ft, now=self.now)


if __name__ == "__main__":
    unittest.main()
