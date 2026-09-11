import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import release


class ReleaseTest(unittest.TestCase):
    def test_committed_artifacts_match_generated_statistics(self):
        data = release.artifacts()
        self.assertEqual(sum(data['q02_n333']['proof_counts'].values()), 496)
        self.assertEqual(sum(data['q03_n233']['proof_counts'].values()), 31)
        for name, expected in release.generated(data, include_paper=(release.ROOT / 'paper').is_dir()).items():
            self.assertEqual((release.ROOT / name).read_text(), expected)

    def test_checkout_without_paper(self):
        data = release.artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in release.generated(data, include_paper=False).items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            with mock.patch.object(release, 'ROOT', root), \
                 mock.patch.object(release, 'artifacts', return_value=data), \
                 mock.patch('sys.argv', ['release.py']):
                release.main()
            self.assertFalse((root / 'paper').exists())

    def test_manifest_does_not_infer_success_from_reference_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'unfinished.log'
            path.write_text(json.dumps({'commit': 'old', 'inputs_sha256': {}}) + '\npartial output\n')
            with mock.patch.object(release.subprocess, 'check_output', side_effect=['current\n', '']):
                result = release.manifest({'expected': 'success'}, [path])
            self.assertEqual(result['code_commit'], 'current')
            self.assertEqual(result['runs'][0]['runner_status'], 'INCOMPLETE')
            self.assertEqual(result['runs'][0]['metadata']['commit'], 'old')
