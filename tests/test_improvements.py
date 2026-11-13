import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error

from harness.budget import Budget
from harness.llm import Proxy, ProxyError, retry_delay
from harness.patches import merge
from harness.planner import Planner
from harness.runtime import Harness, Task
from harness.specs import arithmetic, evaluate, format_value
from harness.tools import LocalTools
from tools.hard_cases import generate


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        (self.root/'sales.csv').write_text('order_id,region,units,price\n1,north,2,0.1\n1,north,3,0.2\n2,south,4,0.1\n')

    def query(self, **changes):
        plan=dict(file='sales.csv',filters=[],aggregate='sum',column='units*price')
        plan.update(changes)
        return evaluate('data_analysis',plan,self.root,'Use sales.csv with region north or south and months 5 through 8.')

    def test_exact_math_and_each_floor_step(self):
        self.assertEqual(evaluate('math',{'expression':'((1752 // 2)*2//3)*2//3'},self.root,''),'389')
        self.assertEqual(format_value(arithmetic('0.1 + 0.2'),''),'0.3')
        self.assertEqual(format_value(arithmetic('1/7'),'rounded to 2 decimal places'),'0.14')

    def test_disallowed_math_and_unspecified_rounding(self):
        for expression in ('__import__("os")','2**1000','True','[1,2]'):
            with self.subTest(expression=expression), self.assertRaises(ValueError): arithmetic(expression)
        with self.assertRaises(ValueError): format_value(arithmetic('1/3'),'')
        with self.assertRaises(ValueError): format_value(arithmetic('1/8'),'rounded to 2 decimal places')

    def test_csv_exact_revenue_group_and_distinct(self):
        self.assertEqual(self.query(),'1.2')
        self.assertEqual(self.query(group_by='region',pick='max',column='units'),'north')
        self.assertEqual(self.query(aggregate='count_distinct',column='order_id'),'2')
        self.assertEqual(self.query(filters=[dict(column='region',op='eq',value='north')],column='units'),'5')

    def test_csv_missing_cells_and_ties_abstain(self):
        (self.root/'sales.csv').write_text('region,units\nnorth,\nsouth,4\n')
        with self.assertRaises(ValueError): self.query(column='units')
        self.assertEqual(self.query(column='units',filters=[dict(column='region',op='eq',value='south')]),'4')
        (self.root/'sales.csv').write_text('region,units\nnorth,4\nsouth,4\n')
        with self.assertRaises(ValueError): self.query(column='units',group_by='region',pick='max')

    def test_csv_unknown_filter_relevant_and_irrelevant(self):
        (self.root/'sales.csv').write_text('region,month,units\nnorth,,3\nsouth,5,7\n')
        filters=[dict(column='month',op='between',value=5,high=8),dict(column='region',op='eq',value='south')]
        self.assertEqual(self.query(column='units',filters=filters),'7')
        filters[-1]['value']='north'
        with self.assertRaises(ValueError): self.query(column='units',filters=filters)

    def test_csv_paths_fabricated_values_and_incomplete_rows(self):
        for changes in (dict(file='../outside.csv'),dict(filters=[dict(column='region',op='eq',value='fabricated')])):
            with self.assertRaises((ValueError,FileNotFoundError)): self.query(**changes)
        with self.assertRaises(ValueError):
            evaluate('data_analysis',dict(file='sales.csv',aggregate='count'),self.root,'The CSV has 99 rows.')

    def test_kb_plan_follows_actual_links(self):
        kb=self.root/'kb';kb.mkdir()
        (kb/'a.md').write_text('# Ada Vale\nEmployer: Cedar\n')
        (kb/'c.md').write_text('# Cedar\nHeadquarters: Lima\n')
        plan=dict(entity='Ada Vale',path=['Employer','Headquarters'])
        self.assertEqual(evaluate('search',plan,self.root,'Where does Ada Vale work?'),'Lima')
        plan['path']=['mentor']
        with self.assertRaises(ValueError): evaluate('search',plan,self.root,'Ada Vale')

    def test_one_model_plan_computes_and_persists_answer(self):
        folder=self.root/'tasks'/'m';(folder/'workspace').mkdir(parents=True)
        (folder/'task.json').write_text(json.dumps(dict(task_type='math',instruction='Compute 17 * 23',reward=1,expected='SECRET')))
        harness=Harness(self.root/'tasks')
        class Model:
            def complete(self,messages,*args):
                assert 'SECRET' not in json.dumps(messages)
                return '{"tool":"spec","expression":"17 * 23"}',12
        harness.proxy=Model()
        solved,cost,_=harness.solve_llm(harness.tasks[0],1000)
        self.assertTrue(solved)
        self.assertEqual(cost,12)
        self.assertEqual((folder/'answer.txt').read_text().strip(),'391')


class PatchTests(unittest.TestCase):
    def test_function_preserves_other_source_and_decorators(self):
        source='# keep\nVALUE=9\ndef f(x):\n    return 0\n\ndef helper():\n    return VALUE\n'
        result=merge(source,'def f(x):\n    return x+1\n')
        self.assertIn('# keep\nVALUE=9',result)
        self.assertIn('def helper():\n    return VALUE',result)
        namespace={};exec(result,namespace)
        self.assertEqual(namespace['f'](2),3)

    def test_partial_class_retains_attributes_methods_and_indentation(self):
        source='class C:\n  marker=42\n  def f(self,x):\n    return 0\n  def keep(self):\n    return self.marker\n'
        result=merge(source,'class C:\n    def f(self,x):\n        return x+1\n')
        namespace={};exec(result,namespace)
        self.assertEqual(namespace['C']().f(2),3)
        self.assertEqual(namespace['C']().keep(),42)

    def test_missing_staticmethod_decorator_is_preserved(self):
        source='class C:\n    @staticmethod\n    def f(x):\n        return 0\n'
        result=merge(source,'class C:\n    def f(x):\n        return x+1\n')
        namespace={};exec(result,namespace)
        self.assertEqual(namespace['C'].f(2),3)
        self.assertIn('@staticmethod',result)

    def test_patch_rejects_statements_unknown_definitions_and_new_class_members(self):
        for text in ('print(42)','def missing(): pass','class C:\n    marker=0\n'):
            with self.assertRaises(ValueError): merge('class C:\n    marker=42\n',text)

    def test_tool_patch_rolls_back_and_never_edits_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source='def f():\n    return 0\n'
            (root/'m.py').write_text(source);(root/'test_m.py').write_text(source)
            tools=LocalTools(root)
            self.assertIn('Saved',tools.execute(dict(tool='patch',path='m.py',content='def f():\n    return 1\n')))
            self.assertIn('Tool error',tools.execute(dict(tool='patch',path='test_m.py',content='def f():\n    return 1\n')))
            tools.rollback()
            self.assertEqual((root/'m.py').read_text(),source)
            self.assertEqual((root/'test_m.py').read_text(),source)

    def test_agent_patches_function_then_checks_unchanged_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); folder=root/'job'; workspace=folder/'workspace';workspace.mkdir(parents=True)
            (folder/'task.json').write_text(json.dumps(dict(task_type='coding',instruction='Fix custom.',reward=2)))
            (workspace/'mylib.py').write_text('def custom(x):\n    return 0\n\ndef keep():\n    return 42\n')
            tests='import unittest\nimport mylib\nclass T(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(mylib.custom(2),4)\n        self.assertEqual(mylib.keep(),42)\n'
            (workspace/'test_mylib.py').write_text(tests)
            actions=iter([dict(tool='patch',path='mylib.py',content='def custom(x):\n    return x*2\n'),dict(tool='test')])
            class Model:
                def complete(self,*args): return json.dumps(next(actions)),10
            harness=Harness(root);harness.proxy=Model()
            self.assertTrue(harness.solve_llm(harness.tasks[0],2000)[0])
            self.assertEqual((workspace/'test_mylib.py').read_text(),tests)
            self.assertEqual((folder/'answer.txt').read_text().strip(),'DONE')


class RetryTests(unittest.TestCase):
    def call(self, errors, budget=5000, **options):
        ledger=Budget(budget)
        client=Proxy('https://example.invalid/v1','fake-key',ledger,**options)
        success=io.BytesIO(json.dumps(dict(choices=[dict(message=dict(content='OK'))],usage=dict(prompt_tokens=10,completion_tokens=2))).encode())
        with patch('harness.llm.urllib.request.build_opener') as opener, patch('harness.llm.time.sleep') as sleep:
            opener.return_value.open.side_effect=errors+[success]
            result=client.complete([], 'qwen3.8-27b-noreason',200,budget)
            return result,ledger,opener.return_value.open.call_count,sleep

    def error(self, code, headers=None):
        return urllib.error.HTTPError('https://example.invalid',code,'',headers or {},io.BytesIO(b'{}'))

    def test_retry_after_then_success_charges_both_requests(self):
        (answer,cost),ledger,calls,sleep=self.call([self.error(429,{'Retry-After':'3'})])
        self.assertEqual(answer,'OK');self.assertEqual(calls,2)
        self.assertEqual(cost,ledger.spent);self.assertEqual(ledger.calls,2)
        sleep.assert_called_once_with(3)

    def test_transient_503_and_date_header(self):
        result,ledger,calls,_=self.call([self.error(503)])
        self.assertEqual(calls,2);self.assertEqual(ledger.reserved,0)
        with patch('harness.llm.time.time',return_value=0):
            self.assertEqual(retry_delay({'Retry-After':'Thu, 01 Jan 1970 00:00:05 GMT'}),5)

    def test_auth_and_long_quota_wait_are_not_retried(self):
        for error in (self.error(401),self.error(429,{'Retry-After':'60'})):
            with patch('harness.llm.urllib.request.build_opener') as opener, patch('harness.llm.time.sleep') as sleep:
                opener.return_value.open.side_effect=error
                client=Proxy('https://example.invalid','x',Budget(5000))
                with self.assertRaises(ProxyError): client.complete([], 'qwen3.8-27b-noreason',200,5000)
                self.assertEqual(opener.return_value.open.call_count,1);sleep.assert_not_called()

    def test_deadline_prevents_retry(self):
        with self.assertRaises(ProxyError): self.call([self.error(503)],deadline=time.monotonic()+.5)

    def test_retries_are_bounded_and_no_reservation_leaks(self):
        ledger=Budget(5000);client=Proxy('https://example.invalid','x',ledger)
        with patch('harness.llm.urllib.request.build_opener') as opener, patch('harness.llm.time.sleep'):
            opener.return_value.open.side_effect=[self.error(503) for _ in range(4)]
            with self.assertRaises(ProxyError) as error:
                client.complete([], 'qwen3.8-27b-noreason',200,1000)
            self.assertEqual(opener.return_value.open.call_count,3)
            self.assertEqual(ledger.reserved,0)
            self.assertEqual(error.exception.cost,ledger.spent)
            self.assertLessEqual(ledger.spent,1000)


class EstimatorAndFixturesTests(unittest.TestCase):
    def test_actual_prompt_size_affects_cost(self):
        small=Task('s','coding','x',1,Path('.'),Path('.'),prompt_bytes=100,expected_output=100,expected_calls=2)
        large=Task('l','coding','x',1,Path('.'),Path('.'),prompt_bytes=20000,expected_output=100,expected_calls=2)
        self.assertLess(Planner().estimate(small)[0],Planner().estimate(large)[0])

    def test_independent_fixtures_have_hidden_contracts_and_no_labels_in_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'hard';manifest=generate(root,100,89)
            self.assertEqual(sum(not t['solvable'] for t in manifest['tasks'].values()),20)
            self.assertTrue(any(t.get('hidden_test') for t in manifest['tasks'].values()))
            for path in root.glob('*/task.json'):
                data=json.loads(path.read_text())
                self.assertEqual(set(data),{'task_id','task_type','instruction','reward'})
            with self.assertRaises(ValueError): generate(root,100,89)
