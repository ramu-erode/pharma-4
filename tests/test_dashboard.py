"""Every dashboard page renders against the running stack (`pytest -m compose`)."""

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.compose


@pytest.mark.parametrize("page", ["live", "alerts", "yield", "graph", "uns", "ask"])
def test_page_renders(page):
    app = AppTest.from_string(
        f"from dashboard import views\nviews.sidebar()\nviews.page_{page}()\n",
        default_timeout=60,
    )
    app.run()
    assert not app.exception, [e.value for e in app.exception]


def test_navigation_renders():
    from pathlib import Path

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"))
    app.run(timeout=60)
    assert not app.exception, [e.value for e in app.exception]
