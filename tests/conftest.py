import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
from app.models import BlogDoc
from datetime import datetime

@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_blog_doc():
    """Sample BlogDoc for testing."""
    return BlogDoc(
        id="1",
        title="Understanding Data Science",
        url="https://example.com/data-science",
        short_desc="A comprehensive guide to data science",
        body="Data science is an interdisciplinary field. It combines statistics, programming, and domain expertise. Python and R are popular languages.",
        author="John Doe",
        tags=["data science", "python", "statistics"],
        published_at=datetime(2023, 1, 15)
    )

@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing HTML cleaning."""
    return '''
    <div class="content">
        <h2>Introduction to Machine Learning</h2>
        <p>Machine learning is a subset of <a href="https://en.wikipedia.org/wiki/Artificial_intelligence">artificial intelligence</a>.</p>
        <style>
            .content { color: blue; }
        </style>
        <script>
            console.log("test");
        </script>
        <p>Check out our <a href="https://example.com/course">machine learning course</a> for more details.</p>
    </div>
    '''

@pytest.fixture
def mock_llm_response():
    """Mock LLM API response for SQL processing."""
    return [
        {
            "id": 1,
            "title": "Box Plot Analysis",
            "url": "what-is-box-plot",
            "short_desc": "Understanding box plots in data visualization",
            "body": "<p>Box plots are statistical visualizations. Visit <a href='https://example.com'>our site</a> for more.</p>",
            "author": "Data Scientist",
            "meta_title": "Box Plot Tutorial - Data Science",
            "meta_description": "Learn about box plots, quartiles, and data visualization",
            "blog_date": "Aug 26, 2025"
        }
    ]