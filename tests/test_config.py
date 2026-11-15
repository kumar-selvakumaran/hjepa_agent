from hjepa_agent.config import ASSETS_DIR, DATA_DIR, OUTPUTS_DIR, PROJECT_DIR


def test_paths_inside_project():
    for p in (DATA_DIR, ASSETS_DIR, OUTPUTS_DIR):
        assert PROJECT_DIR in p.parents
