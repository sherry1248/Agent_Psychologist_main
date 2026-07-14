"""
Pytest configuration and fixtures.

This module provides shared fixtures for all tests,
ensuring MOCK mode is used throughout.
"""

import os
import inspect
import sys
import types
import pytest
import asyncio

# CRITICAL: Set MOCK mode before any imports
os.environ["LLM_TYPE"] = "MOCK"


class _FakeArray:
    def __init__(self, data):
        self.data = self._unwrap(data)

    @classmethod
    def _unwrap(cls, data):
        if isinstance(data, _FakeArray):
            return cls._unwrap(data.data)
        if isinstance(data, list):
            return [cls._unwrap(value) for value in data]
        if isinstance(data, tuple):
            return [cls._unwrap(value) for value in data]
        return data

    @property
    def ndim(self):
        return 2 if self.data and isinstance(self.data[0], list) else 1

    @property
    def shape(self):
        if self.ndim == 1:
            return (len(self.data),)
        return (len(self.data), len(self.data[0]) if self.data else 0)

    @property
    def T(self):
        if self.ndim == 1:
            return _FakeArray([[value] for value in self.data])
        return _FakeArray([list(col) for col in zip(*self.data)])

    def astype(self, _dtype):
        return self

    def flatten(self):
        if self.ndim == 1:
            return _FakeArray(list(self.data))
        return _FakeArray([value for row in self.data for value in row])

    def reshape(self, rows, cols):
        flat = self.flatten().data
        if rows == 1 and cols == -1:
            return _FakeArray([flat])
        if cols == -1:
            cols = len(flat) // rows
        return _FakeArray([flat[idx * cols:(idx + 1) * cols] for idx in range(rows)])

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, item):
        if isinstance(item, list):
            return _FakeArray([self.data[idx] for idx in item])
        if isinstance(item, slice):
            return _FakeArray(self.data[item])
        return self.data[item]

    def __eq__(self, other):
        other_data = other.data if isinstance(other, _FakeArray) else other
        if self.ndim == 1:
            return _FakeArray([value == other_data[idx] for idx, value in enumerate(self.data)])
        return _FakeArray([
            [
                value == other_data[row_idx][col_idx]
                for col_idx, value in enumerate(row)
            ]
            for row_idx, row in enumerate(self.data)
        ])

    def all(self):
        return all(self.flatten().data)

    def __truediv__(self, other):
        other_data = other.data if isinstance(other, _FakeArray) else other
        if self.ndim == 1:
            if isinstance(other_data, list):
                return _FakeArray([value / other_data[idx] for idx, value in enumerate(self.data)])
            return _FakeArray([value / other_data for value in self.data])

        divided = []
        for row_idx, row in enumerate(self.data):
            if isinstance(other_data, list):
                denominator = other_data[row_idx]
                if isinstance(denominator, list):
                    denominator = denominator[0]
            else:
                denominator = other_data
            divided.append([value / denominator for value in row])
        return _FakeArray(divided)

    def __mul__(self, other):
        other_data = other.data if isinstance(other, _FakeArray) else other
        if self.ndim == 1:
            if isinstance(other_data, list):
                return _FakeArray([value * other_data[idx] for idx, value in enumerate(self.data)])
            return _FakeArray([value * other_data for value in self.data])

        multiplied = []
        for row_idx, row in enumerate(self.data):
            if isinstance(other_data, list):
                multiplier = other_data[row_idx]
                if isinstance(multiplier, list):
                    multiplied.append([
                        value * multiplier[col_idx]
                        for col_idx, value in enumerate(row)
                    ])
                    continue
            else:
                multiplier = other_data
            multiplied.append([value * multiplier for value in row])
        return _FakeArray(multiplied)

    __rmul__ = __mul__

    def __add__(self, other):
        if self.ndim == 1:
            return _FakeArray([value + other for value in self.data])
        return _FakeArray([[value + other for value in row] for row in self.data])

    __radd__ = __add__


class _FakeRandom:
    def __init__(self):
        self._seed = 0

    def seed(self, seed):
        self._seed = int(seed or 0)

    def randn(self, *shape):
        if len(shape) == 1:
            rows, cols = 1, shape[0]
            flat = True
        else:
            rows, cols = shape
            flat = False

        values = []
        state = self._seed or 1
        for row in range(rows):
            row_values = []
            for col in range(cols):
                state = (1103515245 * state + 12345) % (2**31)
                row_values.append((state / (2**30)) - 1.0)
            values.append(row_values)
        self._seed = state

        return _FakeArray(values[0] if flat else values)


class _FakeLinalg:
    @staticmethod
    def norm(values, axis=None, keepdims=False):
        array = values if isinstance(values, _FakeArray) else _FakeArray(values)
        if axis == 1:
            norms = [(sum(value * value for value in row) ** 0.5) for row in array.data]
            if keepdims:
                return _FakeArray([[value] for value in norms])
            return _FakeArray(norms)

        flat = array.flatten().data
        return sum(value * value for value in flat) ** 0.5


def _fake_array(values):
    return values if isinstance(values, _FakeArray) else _FakeArray(values)


def _fake_vstack(arrays):
    rows = []
    for array in arrays:
        fake = _fake_array(array)
        if fake.ndim == 1:
            rows.append(fake.data)
        else:
            rows.extend(fake.data)
    return _FakeArray(rows)


def _fake_dot(left, right):
    left_array = _fake_array(left)
    right_array = _fake_array(right)
    left_rows = [left_array.data] if left_array.ndim == 1 else left_array.data
    right_rows = [right_array.data] if right_array.ndim == 1 else right_array.data
    right_columns = [list(col) for col in zip(*right_rows)]
    result = []
    for left_row in left_rows:
        result.append([
            sum(a * b for a, b in zip(left_row, right_col))
            for right_col in right_columns
        ])
    return _FakeArray(result)


def _fake_argsort(values):
    data = values.data if isinstance(values, _FakeArray) else values
    return sorted(range(len(data)), key=lambda idx: data[idx])


def _install_fake_numpy():
    if "numpy" in sys.modules:
        return
    try:
        import numpy  # noqa: F401
        return
    except ImportError:
        pass

    sys.modules["numpy"] = types.SimpleNamespace(
        ndarray=_FakeArray,
        array=_fake_array,
        vstack=_fake_vstack,
        dot=_fake_dot,
        argsort=_fake_argsort,
        argmax=lambda values: max(range(len(values)), key=lambda idx: values[idx]),
        max=max,
        linalg=types.SimpleNamespace(norm=_FakeLinalg.norm),
        random=_FakeRandom(),
        float32="float32",
    )


_install_fake_numpy()


def _install_fake_pydantic():
    if "pydantic" in sys.modules:
        return
    try:
        import pydantic  # noqa: F401
        return
    except ImportError:
        pass

    class BaseModel:
        def __init__(self, **kwargs):
            annotations = getattr(self.__class__, "__annotations__", {})
            for key in annotations:
                if key in kwargs:
                    value = kwargs[key]
                else:
                    default = getattr(self.__class__, key, None)
                    value = None if default is Ellipsis else default
                setattr(self, key, value)
            for key, value in kwargs.items():
                if key not in annotations:
                    setattr(self, key, value)

        def dict(self):
            return dict(self.__dict__)

        def model_dump(self):
            return self.dict()

    def Field(default=None, **kwargs):
        if "default_factory" in kwargs:
            return kwargs["default_factory"]()
        return default

    sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=BaseModel, Field=Field)


_install_fake_pydantic()


def pytest_pyfunc_call(pyfuncitem):
    """Run @pytest.mark.asyncio tests when pytest-asyncio is unavailable."""
    testfunction = pyfuncitem.obj
    if pyfuncitem.get_closest_marker("asyncio") and inspect.iscoroutinefunction(testfunction):
        kwargs = {
            name: pyfuncitem.funcargs[name]
            for name in pyfuncitem._fixtureinfo.argnames
        }
        asyncio.run(testfunction(**kwargs))
        return True
    return None


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_mode():
    """Ensure mock mode is set."""
    original = os.environ.get("LLM_TYPE")
    os.environ["LLM_TYPE"] = "MOCK"
    yield True
    if original:
        os.environ["LLM_TYPE"] = original


@pytest.fixture
async def mock_agent():
    """Create a mock-mode PsychologistAgent."""
    from src.main import PsychologistAgent

    agent = PsychologistAgent(mock_mode=True)
    await agent.initialize()
    yield agent
    await agent.shutdown()


@pytest.fixture
async def mock_session(mock_agent):
    """Create a mock session."""
    session = await mock_agent.session_manager.create_session()
    yield session
    await mock_agent.session_manager.end_session(session.session_id)


@pytest.fixture
def safety_gateway():
    """Create a mock-mode SafetyGateway."""
    from src.safety.gateway import SafetyGateway
    from src.safety.embeddings import EmbeddingManager
    from src.safety.patterns import PatternLoader

    EmbeddingManager.reset_instance()
    PatternLoader.clear_cache()

    gateway = SafetyGateway(mock_mode=True)
    yield gateway

    EmbeddingManager.reset_instance()
    PatternLoader.clear_cache()


@pytest.fixture
def pii_redactor():
    """Create a mock-mode PIIRedactor."""
    from src.privacy.pii_redactor import PIIRedactor
    return PIIRedactor(mock_mode=True, use_presidio=False)


@pytest.fixture
async def rag_retriever():
    """Create a mock-mode RAGRetriever."""
    from src.rag.retriever import RAGRetriever
    from src.safety.embeddings import EmbeddingManager

    EmbeddingManager.reset_instance()

    retriever = RAGRetriever(mock_mode=True)
    await retriever.initialize()
    yield retriever

    EmbeddingManager.reset_instance()


@pytest.fixture
def prompt_generator():
    """Create a PromptGenerator."""
    from src.prompt.generator import PromptGenerator
    return PromptGenerator()


@pytest.fixture
def deepseek_client():
    """Create a mock-mode DeepseekClient."""
    from src.api.deepseek_client import DeepseekClient
    return DeepseekClient(mock_mode=True)


@pytest.fixture
def risk_checker():
    """Create a RiskChecker."""
    from src.audit.risk_checker import RiskChecker
    return RiskChecker()


@pytest.fixture
def crisis_handler():
    """Create a CrisisHandler."""
    from src.audit.crisis_handler import CrisisHandler
    return CrisisHandler()


@pytest.fixture
async def local_generator():
    """Create a mock-mode LocalGenerator."""
    from src.inference.generator import LocalGenerator

    generator = LocalGenerator(mock_mode=True)
    await generator.initialize()
    yield generator
    await generator.unload()


@pytest.fixture
def memory_store():
    """Create a MemoryStore."""
    from src.memory.store import MemoryStore
    return MemoryStore()


@pytest.fixture
def session_manager(memory_store):
    """Create a SessionManager."""
    from src.session.manager import SessionManager
    return SessionManager(memory_store=memory_store)


@pytest.fixture
def audit_logger(tmp_path):
    """Create an AuditLogger with temp directory."""
    from src.audit.logger import AuditLogger, AuditLoggerConfig

    config = AuditLoggerConfig(
        log_dir=str(tmp_path / "audit"),
        log_to_console=False
    )
    return AuditLogger(config)


# Helper functions for tests
def assert_safe_response(result):
    """Assert that a response is safe (no crisis indicators)."""
    assert result.get("requires_crisis_response", False) is False
    assert result.get("risk_level", "none") in ["none", "low"]


def assert_crisis_response(result):
    """Assert that a response indicates crisis."""
    assert result.get("requires_crisis_response", False) is True
    assert "988" in result.get("response", "") or result.get("risk_level") in ["high", "critical"]
