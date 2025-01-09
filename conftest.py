import pytest

def pytest_collection_modifyitems(config, items):
    for item in items:
        if not {"interfaces", "visualization"}.intersection(marker.name for marker in item.iter_markers()):
            item.add_marker(pytest.mark.basic)