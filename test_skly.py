
import pytest
import yaml
import skly
from pathlib import Path

class YAMLFileFixtures:

    def __init__(self, tmpdir, data=None):
        self.tmpdir = Path(tmpdir)
        self.data = data or Path(__file__).parent / "tests" / "data"

    def example(self, name):
        filepath = self.tmpdir / name
        filepath.write_bytes((self.data / name).read_bytes())
        return filepath

@pytest.fixture
def yamlexample(tmpdir):
    return YAMLFileFixtures(tmpdir)


def test_include(yamlexample):
    
    filename = yamlexample.example("include-src.yml")
    yamlexample.example("include-target.yml")
    
    with filename.open("r") as f:
        data = skly.load(f)
    assert data == {'document': {'subtree': {'other': 'hello!'}}}
    
class MockLoaded:

    def __init__(self, setting=True):
        super().__init__()
        self.setting = setting

def test_object(yamlexample):
    o = yaml.load(yamlexample.example("mock-obj.yml").open("r"), Loader=skly.Loader)

    assert o['item'].setting is False
