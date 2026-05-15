"""Tests for models/session.py and services/session_store.py."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from models.cleaning_plan import CleaningOperation, CleaningOperationType, CleaningPlan
from models.insight import Insight
from models.kpi import KPI, ChartConfig
from models.profile import (
    CategoricalStats,
    ColumnSchema,
    DataProfile,
    NullReport,
    SampleRows,
    SemanticType,
)
from models.session import (
    FilePaths,
    Session,
    SessionState,
    VALID_TRANSITIONS,
    ValidationReport,
)
from services.session_store import SessionNotFoundError, SessionStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    """SessionStore backed by a temporary directory — isolated per test."""
    return SessionStore(sessions_dir=tmp_path / "sessions")


@pytest.fixture
def simple_profile() -> DataProfile:
    return DataProfile(
        row_count=100,
        column_count=2,
        schema_report=[
            ColumnSchema(name="x", dtype="float64", semantic_type=SemanticType.NUMERIC_CONTINUOUS),
            ColumnSchema(name="y", dtype="object", semantic_type=SemanticType.CATEGORICAL),
        ],
        null_analysis=[
            NullReport(column="x", null_count=0, null_pct=0.0),
            NullReport(column="y", null_count=0, null_pct=0.0),
        ],
        duplicate_count=0,
        statistics={
            "y": CategoricalStats(unique_count=3, cardinality_ratio=0.03, top_values={"A": 50}),
        },
        quality_flags=[],
        sample_rows=SampleRows(first=[], last=[], random=[]),
    )


# ---------------------------------------------------------------------------
# Session model — state machine
# ---------------------------------------------------------------------------

class TestSessionStateMachine:
    def test_initial_state_is_uploaded(self):
        s = Session()
        assert s.state == SessionState.UPLOADED

    def test_valid_transition_uploaded_to_profiled(self):
        s = Session()
        s.transition_to(SessionState.PROFILED)
        assert s.state == SessionState.PROFILED

    def test_full_happy_path(self):
        s = Session()
        path = [
            SessionState.PROFILED,
            SessionState.PLAN_PROPOSED,
            SessionState.PLAN_APPROVED,
            SessionState.CLEANING,
            SessionState.VALIDATED,
            SessionState.CONTEXT_SET,
            SessionState.KPI_COMPUTING,
            SessionState.COMPLETE,
        ]
        for state in path:
            s.transition_to(state)
        assert s.state == SessionState.COMPLETE

    def test_plan_proposed_can_re_propose(self):
        s = Session()
        s.transition_to(SessionState.PROFILED)
        s.transition_to(SessionState.PLAN_PROPOSED)
        s.transition_to(SessionState.PLAN_PROPOSED)  # re-generate plan
        assert s.state == SessionState.PLAN_PROPOSED

    def test_invalid_transition_raises(self):
        s = Session()
        with pytest.raises(ValueError, match="Invalid transition"):
            s.transition_to(SessionState.COMPLETE)

    def test_invalid_transition_skipping_states_raises(self):
        s = Session()
        with pytest.raises(ValueError):
            s.transition_to(SessionState.CLEANING)

    def test_transition_from_complete_raises(self):
        s = Session()
        for state in [
            SessionState.PROFILED, SessionState.PLAN_PROPOSED,
            SessionState.PLAN_APPROVED, SessionState.CLEANING,
            SessionState.VALIDATED, SessionState.CONTEXT_SET,
            SessionState.KPI_COMPUTING, SessionState.COMPLETE,
        ]:
            s.transition_to(state)
        with pytest.raises(ValueError):
            s.transition_to(SessionState.UPLOADED)

    def test_error_message_lists_allowed_states(self):
        s = Session()
        with pytest.raises(ValueError, match="PROFILED"):
            s.transition_to(SessionState.COMPLETE)

    def test_transition_records_timestamp(self):
        s = Session()
        s.transition_to(SessionState.PROFILED)
        assert "PROFILED" in s.timestamps
        assert s.timestamps["PROFILED"] is not None

    def test_multiple_transitions_record_timestamps(self):
        s = Session()
        s.transition_to(SessionState.PROFILED)
        s.transition_to(SessionState.PLAN_PROPOSED)
        assert "PROFILED" in s.timestamps
        assert "PLAN_PROPOSED" in s.timestamps

    def test_valid_transitions_map_covers_all_states(self):
        """Every SessionState must appear as a key in VALID_TRANSITIONS."""
        for state in SessionState:
            assert state in VALID_TRANSITIONS


# ---------------------------------------------------------------------------
# Session model — data fields
# ---------------------------------------------------------------------------

class TestSessionFields:
    def test_session_id_is_uuid(self):
        s = Session()
        assert isinstance(s.session_id, uuid.UUID)

    def test_two_sessions_have_different_ids(self):
        assert Session().session_id != Session().session_id

    def test_file_paths_default_empty(self):
        s = Session()
        assert s.file_paths.raw is None
        assert s.file_paths.cleaned is None

    def test_total_llm_cost_default_zero(self):
        s = Session()
        assert s.total_llm_cost == 0.0

    def test_timestamps_default_empty(self):
        s = Session()
        assert s.timestamps == {}

    def test_set_data_profile(self, simple_profile):
        s = Session()
        s.data_profile = simple_profile
        assert s.data_profile.row_count == 100

    def test_set_domain_context(self):
        s = Session()
        s.domain_context = "Financial Services"
        assert s.domain_context == "Financial Services"

    def test_set_llm_cost(self):
        s = Session()
        s.total_llm_cost = 0.05
        assert s.total_llm_cost == 0.05

    def test_json_round_trip(self, simple_profile):
        s = Session()
        s.data_profile = simple_profile
        s.transition_to(SessionState.PROFILED)
        raw = s.model_dump_json()
        restored = Session.model_validate_json(raw)
        assert restored.session_id == s.session_id
        assert restored.state == s.state
        assert restored.data_profile.row_count == 100

    def test_validation_report_stored(self):
        s = Session()
        s.validation_report = ValidationReport(
            status="PASS",
            checks=[{"name": "row_count", "status": "PASS", "message": "ok"}],
            summary="All checks passed.",
        )
        assert s.validation_report.status == "PASS"

    def test_kpi_results_stored(self):
        s = Session()
        s.kpi_results = [
            KPI(
                name="Avg", category="Rev", formula_description="mean",
                code="df.mean()", business_value="b", priority="high",
                chart_config=ChartConfig(chart_type="bar", x_axis=None, y_axis=None, title="t"),
            )
        ]
        assert len(s.kpi_results) == 1

    def test_insights_stored(self):
        s = Session()
        s.insights = [Insight(text="Revenue is up.", priority="high", category="Revenue")]
        assert len(s.insights) == 1


# ---------------------------------------------------------------------------
# SessionStore — create / get
# ---------------------------------------------------------------------------

class TestSessionStoreCreateGet:
    def test_create_returns_session(self, store):
        s = store.create_session()
        assert isinstance(s, Session)

    def test_create_persists_to_disk(self, store):
        s = store.create_session()
        path = store._dir / f"{s.session_id}.json"
        assert path.exists()

    def test_get_returns_same_session(self, store):
        s = store.create_session()
        loaded = store.get_session(s.session_id)
        assert loaded.session_id == s.session_id

    def test_get_by_string_id(self, store):
        s = store.create_session()
        loaded = store.get_session(str(s.session_id))
        assert loaded.session_id == s.session_id

    def test_get_nonexistent_raises(self, store):
        fake_id = uuid.uuid4()
        with pytest.raises(SessionNotFoundError):
            store.get_session(fake_id)

    def test_error_message_contains_session_id(self, store):
        fake_id = uuid.uuid4()
        with pytest.raises(SessionNotFoundError, match=str(fake_id)):
            store.get_session(fake_id)

    def test_create_state_is_uploaded(self, store):
        s = store.create_session()
        assert s.state == SessionState.UPLOADED

    def test_sessions_dir_created_automatically(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested" / "sessions"
        store = SessionStore(sessions_dir=new_dir)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# SessionStore — update_session
# ---------------------------------------------------------------------------

class TestSessionStoreUpdate:
    def test_update_persists_changes(self, store, simple_profile):
        s = store.create_session()
        s.data_profile = simple_profile
        store.update_session(s)

        reloaded = store.get_session(s.session_id)
        assert reloaded.data_profile is not None
        assert reloaded.data_profile.row_count == 100

    def test_update_persists_llm_cost(self, store):
        s = store.create_session()
        s.total_llm_cost = 0.042
        store.update_session(s)

        reloaded = store.get_session(s.session_id)
        assert reloaded.total_llm_cost == pytest.approx(0.042)

    def test_update_persists_cleaning_plan(self, store):
        s = store.create_session()
        s.cleaning_plan = CleaningPlan(operations=[
            CleaningOperation(
                id="op1", type=CleaningOperationType.DROP_DUPLICATES,
                description="d", impact="i", confidence="high",
                reversible=True, code="df = df.drop_duplicates()",
                column_names=[], alternatives=[],
            )
        ])
        store.update_session(s)

        reloaded = store.get_session(s.session_id)
        assert reloaded.cleaning_plan is not None
        assert len(reloaded.cleaning_plan.operations) == 1

    def test_update_overwrites_previous(self, store):
        s = store.create_session()
        s.domain_context = "first"
        store.update_session(s)

        s.domain_context = "second"
        store.update_session(s)

        reloaded = store.get_session(s.session_id)
        assert reloaded.domain_context == "second"


# ---------------------------------------------------------------------------
# SessionStore — update_state
# ---------------------------------------------------------------------------

class TestSessionStoreUpdateState:
    def test_update_state_changes_state(self, store):
        s = store.create_session()
        updated = store.update_state(s.session_id, SessionState.PROFILED)
        assert updated.state == SessionState.PROFILED

    def test_update_state_persisted(self, store):
        s = store.create_session()
        store.update_state(s.session_id, SessionState.PROFILED)
        reloaded = store.get_session(s.session_id)
        assert reloaded.state == SessionState.PROFILED

    def test_update_state_records_timestamp(self, store):
        s = store.create_session()
        store.update_state(s.session_id, SessionState.PROFILED)
        reloaded = store.get_session(s.session_id)
        assert "PROFILED" in reloaded.timestamps

    def test_invalid_state_transition_raises(self, store):
        s = store.create_session()
        with pytest.raises(ValueError, match="Invalid transition"):
            store.update_state(s.session_id, SessionState.COMPLETE)

    def test_invalid_transition_does_not_persist(self, store):
        s = store.create_session()
        with pytest.raises(ValueError):
            store.update_state(s.session_id, SessionState.COMPLETE)
        reloaded = store.get_session(s.session_id)
        assert reloaded.state == SessionState.UPLOADED

    def test_update_state_nonexistent_session_raises(self, store):
        with pytest.raises(SessionNotFoundError):
            store.update_state(uuid.uuid4(), SessionState.PROFILED)


# ---------------------------------------------------------------------------
# SessionStore — get_all_sessions
# ---------------------------------------------------------------------------

class TestSessionStoreGetAll:
    def test_empty_store_returns_empty_list(self, store):
        assert store.get_all_sessions() == []

    def test_returns_all_created_sessions(self, store):
        store.create_session()
        store.create_session()
        store.create_session()
        sessions = store.get_all_sessions()
        assert len(sessions) == 3

    def test_all_are_session_instances(self, store):
        store.create_session()
        for s in store.get_all_sessions():
            assert isinstance(s, Session)

    def test_malformed_file_skipped(self, store, tmp_path):
        # Write a corrupt JSON file into the sessions dir
        (store._dir / "bad-file.json").write_text("not valid json", encoding="utf-8")
        store.create_session()
        sessions = store.get_all_sessions()
        # Only the valid session is returned
        assert len(sessions) == 1

    def test_returns_sessions_with_correct_ids(self, store):
        s1 = store.create_session()
        s2 = store.create_session()
        ids = {s.session_id for s in store.get_all_sessions()}
        assert s1.session_id in ids
        assert s2.session_id in ids


# ---------------------------------------------------------------------------
# SessionStore — delete
# ---------------------------------------------------------------------------

class TestSessionStoreDelete:
    def test_delete_removes_file(self, store):
        s = store.create_session()
        path = store._dir / f"{s.session_id}.json"
        assert path.exists()
        store.delete_session(s.session_id)
        assert not path.exists()

    def test_delete_makes_get_raise(self, store):
        s = store.create_session()
        store.delete_session(s.session_id)
        with pytest.raises(SessionNotFoundError):
            store.get_session(s.session_id)

    def test_delete_nonexistent_raises(self, store):
        with pytest.raises(SessionNotFoundError):
            store.delete_session(uuid.uuid4())

    def test_delete_only_removes_target(self, store):
        s1 = store.create_session()
        s2 = store.create_session()
        store.delete_session(s1.session_id)
        remaining = store.get_all_sessions()
        assert len(remaining) == 1
        assert remaining[0].session_id == s2.session_id


# ---------------------------------------------------------------------------
# Persistence across "server restarts" (new store instance, same dir)
# ---------------------------------------------------------------------------

class TestPersistenceAcrossRestarts:
    def test_session_survives_new_store_instance(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        store1 = SessionStore(sessions_dir=sessions_dir)
        s = store1.create_session()
        s.domain_context = "Financial Services"
        store1.update_session(s)

        # Simulate restart: new store instance pointing at same dir
        store2 = SessionStore(sessions_dir=sessions_dir)
        reloaded = store2.get_session(s.session_id)
        assert reloaded.domain_context == "Financial Services"

    def test_state_machine_position_survives_restart(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        store1 = SessionStore(sessions_dir=sessions_dir)
        s = store1.create_session()
        store1.update_state(s.session_id, SessionState.PROFILED)
        store1.update_state(s.session_id, SessionState.PLAN_PROPOSED)

        store2 = SessionStore(sessions_dir=sessions_dir)
        reloaded = store2.get_session(s.session_id)
        assert reloaded.state == SessionState.PLAN_PROPOSED

    def test_full_profile_survives_restart(self, tmp_path, simple_profile):
        sessions_dir = tmp_path / "sessions"
        store1 = SessionStore(sessions_dir=sessions_dir)
        s = store1.create_session()
        s.data_profile = simple_profile
        store1.update_session(s)

        store2 = SessionStore(sessions_dir=sessions_dir)
        reloaded = store2.get_session(s.session_id)
        assert reloaded.data_profile.row_count == 100
        assert reloaded.data_profile.column_count == 2

    def test_timestamps_survive_restart(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        store1 = SessionStore(sessions_dir=sessions_dir)
        s = store1.create_session()
        store1.update_state(s.session_id, SessionState.PROFILED)

        store2 = SessionStore(sessions_dir=sessions_dir)
        reloaded = store2.get_session(s.session_id)
        assert "PROFILED" in reloaded.timestamps

    def test_all_sessions_visible_after_restart(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        store1 = SessionStore(sessions_dir=sessions_dir)
        ids = {store1.create_session().session_id for _ in range(3)}

        store2 = SessionStore(sessions_dir=sessions_dir)
        reloaded_ids = {s.session_id for s in store2.get_all_sessions()}
        assert ids == reloaded_ids

    def test_json_file_is_valid_json(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path / "sessions")
        s = store.create_session()
        path = store._dir / f"{s.session_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "session_id" in data
        assert "state" in data
