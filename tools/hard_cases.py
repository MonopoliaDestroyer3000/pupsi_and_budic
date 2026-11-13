"""Independent challenge fixtures beyond the baseline warehouse templates."""
import argparse
import csv
import json
from pathlib import Path
import random


def generate(output: Path, count: int = 100, seed: int = 89) -> dict:
    if count < 1 or count > 10000: raise ValueError('Count must be 1..10000')
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    rng, labels = random.Random(seed), {}
    for i in range(count):
        variant, kind = (i // 4) % 5, ('math','data_analysis','search','coding')[i % 4]
        name = f'{kind}_{i:04d}'
        workspace = output / name / 'workspace'
        workspace.mkdir(parents=True)
        n, reward = rng.randint(10,99), rng.randint(1,5)
        solvable, hidden = variant != 4, None
        if kind == 'math':
            stories = [
                (f'A tank contains {n} litres. Add 13 litres, then use half the total. What volume remains?', str((n+13)/2).rstrip('0').rstrip('.') if (n+13)%2 else str((n+13)//2)),
                (f'An account starts with {n} coins. Each of two days adds 10 percent interest, rounded down separately each day. What is the final balance?', str(((n*110)//100*110)//100)),
                (f'The expression is ({n} + 7) * (19 - 3). What is its value?', str((n+7)*16)),
                (f'Divide {n} by 7, rounded to 2 decimal places. What is the result?', f'{n/7:.2f}'),
                (f'A tank starts with {n} litres and leaks an unspecified amount. What volume remains?', None),
            ]
            instruction, answer = stories[variant]
        elif kind == 'data_analysis':
            rows = [[1,'north',n,2],[1,'north',3,5],[2,'south',2,7],[3,'south',4,3]]
            if variant == 4: rows[0][2] = ''
            with (workspace/'sales.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f);writer.writerow(['order_id','region','units','price']);writer.writerows(rows)
            options = [
                ('In sales.csv which region has the largest total units?', 'north'),
                ('In sales.csv how many distinct order_id values occur in region north?', '1'),
                ('In sales.csv calculate total revenue as units * price for all rows.', str(n*2+41)),
                ('In sales.csv what is the maximum price for region south?', '7'),
                ('In sales.csv what is the total units for region north?', None),
            ]
            instruction, answer = options[variant]
        elif kind == 'search':
            kb=workspace/'kb';kb.mkdir()
            (kb/'a.md').write_text(f'# Ada Vale\nBorn: {1900+n}\nEmployer: Cedar\nMentor: Ben Vale\n',encoding='utf-8')
            (kb/'b.md').write_text('# Ben Vale\nBorn: 1971\nEmployer: Cedar\n',encoding='utf-8')
            (kb/'c.md').write_text('# Cedar\nHeadquarters: Lima\nFounded: 1993\n',encoding='utf-8')
            if variant == 4:
                (kb/'a.md').write_text('# Ada Vale\nBorn: unknown\n',encoding='utf-8')
            options = [
                ('Find the headquarters city of the company employing Ada Vale.', 'Lima'),
                ("In which year was Ada Vale's mentor born?", '1971'),
                ("What year was Ada Vale's employer founded?", '1993'),
                ('Return the employer of Ben Vale.', 'Cedar'),
                ('In which year was Ada Vale born?', None),
            ]
            instruction, answer = options[variant]
        else:
            instruction = 'Fix mylib.py until the unchanged tests pass. Save DONE as the final answer.'
            answer = 'DONE'
            if variant == 0:
                source='def adjust(value):\n    return value - 7\n\ndef untouched():\n    return "sentinel"\n'
                checks='self.assertEqual(m.adjust(8),15)\n        self.assertEqual(m.untouched(),"sentinel")'
                hidden='self.assertEqual(m.adjust(-2),5)\n        self.assertEqual(m.adjust(0),7)'
            elif variant == 1:
                source='class Counter:\n    label = "keep"\n    def apply(self, value):\n        return value - 1\n    def untouched(self):\n        return 42\n'
                checks='self.assertEqual(m.Counter().apply(3),4)\n        self.assertEqual(m.Counter.label,"keep")'
                hidden='self.assertEqual(m.Counter().apply(-1),0)\n        self.assertEqual(m.Counter().untouched(),42)'
            elif variant == 2:
                source='def clipped(value):\n    return max(10,min(0,value))\n'
                checks='self.assertEqual(m.clipped(5),5)'
                hidden='self.assertEqual(m.clipped(-20),0)\n        self.assertEqual(m.clipped(99),10)'
            elif variant == 3:
                source='def distinct(xs):\n    return sorted(set(xs))\n'
                checks='self.assertEqual(m.distinct([3,1,3]),[3,1])'
                hidden='self.assertEqual(m.distinct([2,0,2,1]),[2,0,1])\n        self.assertEqual(m.distinct([]),[])'
            else:
                instruction += ' The repaired function must be pure and deterministic, depending only on its arguments.'
                source='def custom(value):\n    return value\n'
                checks='self.assertEqual(m.custom(3),4)\n        self.assertEqual(m.custom(3),5)'
            (workspace/'mylib.py').write_text(source,encoding='utf-8')
            template='import unittest\nimport mylib as m\nclass T(unittest.TestCase):\n    def test_contract(self):\n        '
            (workspace/'test_mylib.py').write_text(template+checks+'\n',encoding='utf-8')
            if hidden: hidden=template+hidden+'\n'
        metadata=dict(task_id=name,task_type=kind,instruction=instruction,reward=reward)
        (workspace.parent/'task.json').write_text(json.dumps(metadata),encoding='utf-8')
        labels[name]=dict(task_type=kind,reward=reward,solvable=solvable,expected=answer,
                          check='unittest' if kind=='coding' else 'exact_str',hidden_test=hidden)
    manifest=dict(schema=1,seed=seed,count=count,tasks=labels,
                  warning='Independent harder fixtures: phrasing, grouping, distinct orders, rounding, missing values and hidden coding contracts. Not organizer tasks.')
    (output/'.oracle.json').write_text(json.dumps(manifest),encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--seed',type=int,default=89)
    args=parser.parse_args()
    generate(args.output,args.count,args.seed)
