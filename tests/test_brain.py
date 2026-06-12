"""
Tests for brain/tasks.py and brain/agent.self_correct — llm_call monkeypatched, isolated DB.
"""

import json
import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect every test to a fresh SQLite file."""
    import config
    import core.memory as memory

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "brain_test.db"))
    memory._conn = None
    memory.init_db()

    yield

    if memory._conn:
        memory._conn.close()
    memory._conn = None


def _llm_stub(tasks: list[dict]):
    """Return a monkeypatch target that yields *tasks* as JSON."""
    return lambda *a, **kw: json.dumps(tasks)


# ---------------------------------------------------------------------------
# generate_tasks
# ---------------------------------------------------------------------------

class TestGenerateTasks:
    def test_adds_tasks_to_db(self, monkeypatch):
        import brain.tasks as bt
        stub_tasks = [
            {"title": "T1", "description": "D1", "done_check": "assert 1 == 1"},
            {"title": "T2", "description": "D2", "done_check": "assert 2 == 2"},
        ]
        monkeypatch.setattr(bt, "llm_call", _llm_stub(stub_tasks))

        ids = bt.generate_tasks("test context")

        assert len(ids) == 2
        import core.memory as memory
        all_tasks = memory.get_tasks()
        assert len(all_tasks) == 2
        assert all_tasks[0]["title"] == "T1"

    def test_returns_ids(self, monkeypatch):
        import brain.tasks as bt
        monkeypatch.setattr(bt, "llm_call", _llm_stub([
            {"title": "X", "description": "", "done_check": "assert True"},
        ]))
        ids = bt.generate_tasks("ctx")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_caps_at_three_tasks(self, monkeypatch):
        import brain.tasks as bt
        monkeypatch.setattr(bt, "llm_call", _llm_stub([
            {"title": f"T{i}", "description": "", "done_check": "assert True"}
            for i in range(5)
        ]))
        ids = bt.generate_tasks("ctx")
        assert len(ids) == 3

    def test_invalid_json_returns_empty(self, monkeypatch):
        import brain.tasks as bt
        monkeypatch.setattr(bt, "llm_call", lambda *a, **kw: "not json at all")
        ids = bt.generate_tasks("ctx")
        assert ids == []

    def test_empty_llm_returns_empty(self, monkeypatch):
        import brain.tasks as bt
        monkeypatch.setattr(bt, "llm_call", lambda *a, **kw: "")
        ids = bt.generate_tasks("ctx")
        assert ids == []

    def test_strips_markdown_fences(self, monkeypatch):
        import brain.tasks as bt
        raw = '```json\n[{"title":"T","description":"D","done_check":"assert True"}]\n```'
        monkeypatch.setattr(bt, "llm_call", lambda *a, **kw: raw)
        ids = bt.generate_tasks("ctx")
        assert len(ids) == 1

    def test_missing_done_check_defaults_empty(self, monkeypatch):
        import brain.tasks as bt
        monkeypatch.setattr(bt, "llm_call", _llm_stub([
            {"title": "No check", "description": "desc"},
        ]))
        ids = bt.generate_tasks("ctx")
        import core.memory as memory
        task = memory.get_tasks()[0]
        assert task["done_check"] == ""

    def test_single_object_not_wrapped_in_list(self, monkeypatch):
        import brain.tasks as bt
        single = {"title": "Solo", "description": "desc", "done_check": "assert True"}
        monkeypatch.setattr(bt, "llm_call", lambda *a, **kw: json.dumps(single))
        ids = bt.generate_tasks("ctx")
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# verify_done_check
# ---------------------------------------------------------------------------

class TestVerifyDoneCheck:
    def test_passing_assert_returns_done(self):
        from brain.tasks import verify_done_check
        task = {"id": 1, "done_check": "assert 1 == 1"}
        assert verify_done_check(task) == "done"

    def test_failing_assert_returns_in_progress(self):
        from brain.tasks import verify_done_check
        task = {"id": 2, "done_check": "assert 1 == 2"}
        assert verify_done_check(task) == "in_progress"

    def test_empty_done_check_returns_needs_review(self):
        from brain.tasks import verify_done_check
        assert verify_done_check({"id": 3, "done_check": ""}) == "needs_review"
        assert verify_done_check({"id": 4, "done_check": None}) == "needs_review"
        assert verify_done_check({"id": 5}) == "needs_review"

    def test_sandbox_blocked_returns_needs_review(self):
        from brain.tasks import verify_done_check
        task = {"id": 6, "done_check": "import os; os.system('ls')"}
        assert verify_done_check(task) == "needs_review"

    def test_runtime_error_returns_in_progress(self):
        from brain.tasks import verify_done_check
        task = {"id": 7, "done_check": "assert int('not-a-number') == 1"}
        assert verify_done_check(task) == "in_progress"

    def test_multiline_assert_passes(self):
        from brain.tasks import verify_done_check
        task = {"id": 8, "done_check": "x = 2 + 2\nassert x == 4"}
        assert verify_done_check(task) == "done"


# ---------------------------------------------------------------------------
# run_pending_tasks
# ---------------------------------------------------------------------------

class TestRunPendingTasks:
    def _log(self):
        logs = []
        return logs, logs.append

    def test_marks_done_when_check_passes(self):
        import core.memory as memory
        from brain.tasks import run_pending_tasks

        tid = memory.add_task("passing", "desc", "assert 1 == 1")
        logs, log_fn = self._log()
        run_pending_tasks(log_fn)

        tasks = memory.get_tasks("done")
        assert any(t["id"] == tid for t in tasks)

    def test_keeps_in_progress_when_check_fails(self):
        import core.memory as memory
        from brain.tasks import run_pending_tasks

        tid = memory.add_task("failing", "desc", "assert 1 == 2")
        _, log_fn = self._log()
        run_pending_tasks(log_fn)

        tasks = memory.get_tasks("in_progress")
        assert any(t["id"] == tid for t in tasks)

    def test_needs_review_when_no_check(self):
        import core.memory as memory
        from brain.tasks import run_pending_tasks

        tid = memory.add_task("no check", "desc", "")
        _, log_fn = self._log()
        run_pending_tasks(log_fn)

        tasks = memory.get_tasks("needs_review")
        assert any(t["id"] == tid for t in tasks)

    def test_no_active_tasks_logs_message(self):
        from brain.tasks import run_pending_tasks
        logs, log_fn = self._log()
        run_pending_tasks(log_fn)
        assert any("no active tasks" in msg for msg in logs)

    def test_processes_in_progress_tasks_too(self):
        import core.memory as memory
        from brain.tasks import run_pending_tasks

        tid = memory.add_task("retried", "desc", "assert 2 + 2 == 4")
        memory.update_task_status(tid, "in_progress")

        _, log_fn = self._log()
        run_pending_tasks(log_fn)

        tasks = memory.get_tasks("done")
        assert any(t["id"] == tid for t in tasks)


# ---------------------------------------------------------------------------
# archive_done
# ---------------------------------------------------------------------------

class TestArchiveDone:
    def test_removes_done_tasks(self):
        import core.memory as memory
        from brain.tasks import archive_done

        tid = memory.add_task("finished", "desc", "assert True")
        memory.update_task_status(tid, "done")

        result = archive_done()
        assert result["done"] >= 1
        assert memory.get_tasks("done") == []

    def test_removes_failed_tasks(self):
        import core.memory as memory
        from brain.tasks import archive_done

        tid = memory.add_task("failed task", "desc", "assert False")
        memory.update_task_status(tid, "failed")

        archive_done()
        assert memory.get_tasks("failed") == []

    def test_removes_needs_review_tasks(self):
        import core.memory as memory
        from brain.tasks import archive_done

        tid = memory.add_task("review task", "desc", "")
        memory.update_task_status(tid, "needs_review")

        archive_done()
        assert memory.get_tasks("needs_review") == []

    def test_returns_counts(self):
        import core.memory as memory
        from brain.tasks import archive_done

        t1 = memory.add_task("d1", "", ""); memory.update_task_status(t1, "done")
        t2 = memory.add_task("f1", "", ""); memory.update_task_status(t2, "failed")

        result = archive_done()
        assert isinstance(result["done"], int)
        assert isinstance(result["failed"], int)
        assert isinstance(result["needs_review"], int)

    def test_pending_tasks_untouched(self):
        import core.memory as memory
        from brain.tasks import archive_done

        memory.add_task("still pending", "desc", "assert True")
        archive_done()
        assert len(memory.get_tasks("pending")) == 1


# ---------------------------------------------------------------------------
# brain/agent — self_correct
# ---------------------------------------------------------------------------

class TestSelfCorrect:
    """Tests for brain.agent.self_correct and _action_self_correct."""

    _logs: list

    def _log(self, msg: str) -> None:
        self._logs.append(msg)

    def setup_method(self):
        self._logs = []

    # -- helpers --

    def _patch(self, monkeypatch, llm_reply: str, tests_pass: bool = True):
        """Patch llm_call and _run_tests in brain.agent namespace."""
        import brain.agent as ba
        monkeypatch.setattr(ba, "llm_call", lambda *a, **kw: llm_reply)
        monkeypatch.setattr(
            "brain.agent._action_self_correct.__module__",  # touch nothing
            "brain.agent",
            raising=False,
        )
        # Patch _run_tests inside ops.self_update (imported lazily in self_correct)
        import ops.self_update as su
        monkeypatch.setattr(su, "_run_tests", lambda log: tests_pass)

    # -- sandbox fix --

    def test_sandbox_fix_applied(self, monkeypatch):
        import brain.agent as ba
        fix = json.dumps({"type": "sandbox", "code": "x = 1 + 1"})
        self._patch(monkeypatch, fix, tests_pass=True)
        result = ba.self_correct("SomeError: boom", self._log)
        assert "sandbox" in result

    def test_sandbox_fix_blocked_by_sandbox(self, monkeypatch):
        import brain.agent as ba
        fix = json.dumps({"type": "sandbox", "code": "import os; os.system('ls')"})
        self._patch(monkeypatch, fix, tests_pass=True)
        result = ba.self_correct("err", self._log)
        assert "bloqué" in result

    # -- file fix --

    def test_file_fix_rejected_for_protected_path(self, monkeypatch, tmp_path):
        import brain.agent as ba
        import ops.self_update as su
        fix = json.dumps({"type": "file", "path": "config.py", "content": "x=1"})
        self._patch(monkeypatch, fix, tests_pass=True)
        result = ba.self_correct("err", self._log)
        assert "write_and_test échoué" in result or "Refused" in result.lower() or "protégé" in result.lower() or "échoué" in result

    def test_file_fix_applied_to_brain(self, monkeypatch, tmp_path):
        import brain.agent as ba
        import ops.self_update as su
        # write_and_test itself checks is_modifiable and does real file IO — patch it
        monkeypatch.setattr(su, "write_and_test", lambda path, code, log: (True, ""))
        fix = json.dumps({"type": "file", "path": "brain/agent.py", "content": "# fix"})
        self._patch(monkeypatch, fix, tests_pass=True)
        result = ba.self_correct("err", self._log)
        assert "mis à jour" in result

    # -- pre-flight gate --

    def test_fix_rejected_when_tests_broken(self, monkeypatch):
        import brain.agent as ba
        fix = json.dumps({"type": "sandbox", "code": "x = 1"})
        self._patch(monkeypatch, fix, tests_pass=False)
        result = ba.self_correct("err", self._log)
        assert "rejeté" in result
        assert any("rejeté" in m for m in self._logs)

    # -- retry counter --

    def test_try_counter_resets_on_success(self, monkeypatch):
        import brain.agent as ba
        import core.memory as memory
        # "x = 1" contains "=" so _looks_like_shell passes
        fix = json.dumps({"type": "sandbox", "code": "x = 1"})
        self._patch(monkeypatch, fix, tests_pass=True)
        error = "RetryError: keep failing"
        key = "self_correct_tries:" + ba._error_key(error)
        assert memory.get_state(key, 0) == 0
        ba.self_correct(error, self._log)
        # Counter resets to 0 on success
        assert memory.get_state(key, 0) == 0

    def test_counter_increments_on_preflight_failure(self, monkeypatch):
        import brain.agent as ba
        import core.memory as memory
        fix = json.dumps({"type": "sandbox", "code": "x = 1"})
        self._patch(monkeypatch, fix, tests_pass=False)
        error = "PrefailError: x"
        key = "self_correct_tries:" + ba._error_key(error)
        ba.self_correct(error, self._log)
        assert memory.get_state(key, 0) == 1

    def test_abandons_after_max_tries(self, monkeypatch):
        import brain.agent as ba
        import core.memory as memory
        import config
        fix = json.dumps({"type": "sandbox", "code": "x = 1"})
        self._patch(monkeypatch, fix, tests_pass=False)
        error = "MaxTriesError: exhausted"
        key = "self_correct_tries:" + ba._error_key(error)
        # Pre-seed counter at the limit
        memory.set_state(key, config.ERROR_MAX_TRIES)
        result = ba.self_correct(error, self._log)
        assert "abandonné" in result
        assert "unsolved" in result

    def test_abandon_saves_knowledge(self, monkeypatch):
        import brain.agent as ba
        import core.memory as memory
        import config
        fix = json.dumps({"type": "sandbox", "code": "x = 1"})
        self._patch(monkeypatch, fix, tests_pass=False)
        error = "KnowledgeError: store me"
        key = "self_correct_tries:" + ba._error_key(error)
        memory.set_state(key, config.ERROR_MAX_TRIES)
        ba.self_correct(error, self._log)
        results = memory.search_knowledge(error[:20])
        assert len(results) >= 1

    # -- action wrapper --

    def test_action_self_correct_no_error_in_memory(self, monkeypatch):
        import brain.agent as ba
        import core.memory as memory
        memory.set_state("last_error", None)
        result = ba._action_self_correct(self._log)
        assert "aucune erreur" in result

    def test_action_self_correct_reads_last_error(self, monkeypatch):
        import brain.agent as ba
        import core.memory as memory
        import ops.self_update as su
        memory.set_state("last_error", "ReadError: from memory")
        fix = json.dumps({"type": "sandbox", "code": "pass"})
        monkeypatch.setattr(ba, "llm_call", lambda *a, **kw: fix)
        monkeypatch.setattr(su, "_run_tests", lambda log: True)
        result = ba._action_self_correct(self._log)
        assert result  # action ran something

    # -- registry presence --

    def test_self_correct_in_registry(self):
        from brain.agent import ACTION_REGISTRY
        assert "self_correct" in ACTION_REGISTRY

    def test_invalid_json_from_llm(self, monkeypatch):
        import brain.agent as ba
        self._patch(monkeypatch, "not json", tests_pass=True)
        result = ba.self_correct("err", self._log)
        assert "JSON" in result

    def test_unknown_fix_type(self, monkeypatch):
        import brain.agent as ba
        fix = json.dumps({"type": "magic", "spell": "expelliarmus"})
        self._patch(monkeypatch, fix, tests_pass=True)
        result = ba.self_correct("err", self._log)
        assert "inconnu" in result
