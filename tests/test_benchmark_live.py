import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.benchmark import benchmark


class LiveBenchmarkTests(unittest.TestCase):
    def exercise(self, live):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source'; task=source/'m';(task/'workspace').mkdir(parents=True)
            (task/'task.json').write_text(json.dumps(dict(task_id='m',task_type='math',instruction='Compute 17*23',reward=2,expected='391',check='exact_int')))
            def run(command, **kwargs):
                target=Path(command[command.index('--tasks-dir')+1])
                metadata=json.loads((target/'m'/'task.json').read_text())
                self.assertNotIn('expected',metadata)
                self.assertNotIn('check',metadata)
                self.assertFalse((target/'.oracle.json').exists())
                self.assertNotIn('sample-secret',str(command))
                self.assertEqual('--deterministic-only' not in command,live)
                self.assertEqual('--without-framework' not in command,live)
                self.assertEqual(kwargs['env'].get('LLM_API_KEY'), 'sample-secret' if live else None)
                (target/'m'/'answer.txt').write_text('391')
                checkpoint=Path(command[command.index('--output-file')+1])
                checkpoint.write_text(json.dumps({'budget':{'spent':10,'calls':1},'events':[]}))
                return SimpleNamespace(returncode=0,stderr='sample-secret' if live else '')
            with patch('tools.benchmark.subprocess.run',side_effect=run):
                report=benchmark(source,budget=3000,live=live,base_url='https://example.invalid/v1',api_key='sample-secret')
            self.assertEqual(report['summary']['correct'],1)
            self.assertNotIn('sample-secret',json.dumps(report))
            return report

    def test_live_uses_key_only_in_environment_and_forces_real_framework(self):
        self.assertIn('live proxy',self.exercise(True)['mode'])

    def test_offline_mode_remains_free_even_with_a_key_available(self):
        with patch.dict('os.environ',{'LLM_API_KEY':'sample-secret','LLM_BASE_URL':'https://example.invalid/v1'}):
            self.exercise(False)

    def test_live_requires_credentials_and_rejects_offline_docker_mode(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(ValueError): benchmark(Path('.'),live=True)
            with self.assertRaises(ValueError): benchmark(Path('.'),live=True,container_image='hack:latest')
