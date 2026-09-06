import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app


class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home_endpoint(self):
        """Test API root GET route."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertEqual(json_data['status'], 'running')
        self.assertIn('Multimodal', json_data['system'])

    def test_health_endpoint(self):
        """Test health check route."""
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertEqual(json_data['status'], 'healthy')

    def test_sample_videos_endpoint(self):
        """Test list sample videos route."""
        res = self.client.get('/sample_videos')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertIn('samples', json_data)
        self.assertIsInstance(json_data['samples'], list)

    def test_dashboard_endpoint(self):
        """Test dashboard index HTML serving route."""
        res = self.client.get('/dashboard')
        self.assertIn(res.status_code, [200, 404])


if __name__ == "__main__":
    unittest.main()
