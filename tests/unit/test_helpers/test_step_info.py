"""Tests for step info module."""

import time


from plllm_mlx.helpers.step_info import PlStepHelper, PlStepUsage


class TestPlStepUsage:
    """Tests for PlStepUsage model."""

    def test_default_values(self):
        """Test default values."""
        usage = PlStepUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.prompt_tps == 0
        assert usage.generation_tps == 0
        assert usage.prompt_process == 0
        assert usage.first_token == 0

    def test_custom_values(self):
        """Test custom values."""
        usage = PlStepUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tps=100.0,
            generation_tps=50.0,
            prompt_process=10.5,
            first_token=0.5,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.prompt_tps == 100.0
        assert usage.generation_tps == 50.0
        assert usage.prompt_process == 10.5
        assert usage.first_token == 0.5

    def test_model_dump(self):
        """Test model serialization."""
        usage = PlStepUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        data = usage.model_dump()
        assert data["prompt_tokens"] == 10
        assert data["completion_tokens"] == 5
        assert data["total_tokens"] == 15

    def test_model_dump_json(self):
        """Test JSON serialization."""
        usage = PlStepUsage(prompt_tokens=10, completion_tokens=5)
        json_str = usage.model_dump_json()
        assert "prompt_tokens" in json_str
        assert "10" in json_str

    def test_float_precision(self):
        """Test float precision."""
        usage = PlStepUsage(
            prompt_tps=123.456789,
            generation_tps=98.765432,
        )
        assert abs(usage.prompt_tps - 123.456789) < 1e-6
        assert abs(usage.generation_tps - 98.765432) < 1e-6


class TestPlStepHelper:
    """Tests for PlStepHelper class."""

    def test_init(self):
        """Test initialization."""
        helper = PlStepHelper()
        assert helper._begin_prompt_time is None
        assert helper._prompt_processed is False
        assert helper._first_token_generated is False
        assert helper._last_step_usage.prompt_tokens == 0

    def test_begin_process_prompt(self):
        """Test begin_process_prompt."""
        helper = PlStepHelper()
        helper.begin_process_prompt()
        assert helper._begin_prompt_time is not None

    def test_begin_process_prompt_not_idempotent(self):
        """Test begin_process_prompt updates time each call before end_process_prompt."""
        helper = PlStepHelper()
        helper.begin_process_prompt()
        first_time = helper._begin_prompt_time
        time.sleep(0.01)
        helper.begin_process_prompt()
        # Time should be updated since _prompt_processed is still False
        assert helper._begin_prompt_time >= first_time

    def test_end_process_prompt(self):
        """Test end_process_prompt."""
        helper = PlStepHelper()
        helper.begin_process_prompt()
        time.sleep(0.01)
        helper.end_process_prompt()
        assert helper._prompt_processed is True
        assert helper._last_step_usage.prompt_process > 0

    def test_end_process_prompt_idempotent(self):
        """Test end_process_prompt is idempotent."""
        helper = PlStepHelper()
        helper.begin_process_prompt()
        helper.end_process_prompt()
        first_process_time = helper._last_step_usage.prompt_process
        helper.end_process_prompt()
        assert helper._last_step_usage.prompt_process == first_process_time

    def test_update_step(self):
        """Test update_step."""
        helper = PlStepHelper()
        step = PlStepUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        helper.update_step(step)
        assert helper._last_step_usage.prompt_tokens == 100
        assert helper._last_step_usage.completion_tokens == 50
        assert helper._last_step_usage.total_tokens == 150

    def test_update_step_sets_first_token_time(self):
        """Test update_step sets first token time."""
        helper = PlStepHelper()
        helper.begin_process_prompt()
        helper.end_process_prompt()
        step = PlStepUsage(prompt_tokens=10, completion_tokens=5)
        helper.update_step(step)
        assert helper._first_token_generated is True
        assert helper._last_step_usage.first_token >= 0

    def test_update_step_only_sets_first_token_once(self):
        """Test update_step only sets first token time once."""
        helper = PlStepHelper()
        helper.begin_process_prompt()
        helper.end_process_prompt()
        step1 = PlStepUsage(prompt_tokens=10, completion_tokens=5)
        helper.update_step(step1)
        first_token = helper._last_step_usage.first_token
        step2 = PlStepUsage(prompt_tokens=20, completion_tokens=10)
        helper.update_step(step2)
        assert helper._last_step_usage.first_token == first_token

    def test_build_usage(self):
        """Test build_usage."""
        helper = PlStepHelper()
        helper.begin_process_prompt()
        time.sleep(0.01)
        helper.end_process_prompt()
        step = PlStepUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tps=100.0,
            generation_tps=50.0,
        )
        helper.update_step(step)
        usage = helper.build_usage()
        assert isinstance(usage, dict)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["prompt_process"] >= 0

    def test_full_flow(self):
        """Test full flow."""
        helper = PlStepHelper()
        helper.begin_process_prompt()
        time.sleep(0.01)
        helper.end_process_prompt()
        step = PlStepUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tps=100.0,
            generation_tps=50.0,
        )
        helper.update_step(step)
        usage = helper.build_usage()
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["prompt_process"] > 0
        assert usage["prompt_tps"] == 100.0
        assert usage["generation_tps"] == 50.0

    def test_build_usage_before_process(self):
        """Test build_usage before any processing."""
        helper = PlStepHelper()
        usage = helper.build_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["prompt_process"] == 0
