from argparse import Namespace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from medrag.training.graph_input_data import atomic_json, atomic_jsonl, digest
from scripts import label_rag2_kev_roles_dual as dual
from scripts import label_rag2_kev_roles_full as full
from tests.test_kev_role_labeling import SilentProgress


class DualRoleTests(unittest.TestCase):
    def test_real_shards_exactly_cover_union(self):
        a = Namespace(output_root=Path('/tmp/unused'),preflight=True)
        sets = [set(range(g,full.N,2)) for g in (0,1)]
        self.assertEqual([len(x) for x in sets],[141731,141731])
        self.assertFalse(sets[0]&sets[1])
        self.assertEqual(len(sets[0]|sets[1]),full.N)
        self.assertEqual([full.shard(dual.worker_args(a,g))[2] for g in (0,1)],[141731,141731])

    def test_single_launcher_defaults_unchanged(self):
        a=full.arguments([])
        self.assertEqual((a.gpu,a.shard_count,a.shard_index),(1,1,0))
        self.assertEqual(full.shard(a)[2],full.N)

    def fixture(self,root):
        db=sqlite3.connect(':memory:')
        db.row_factory=sqlite3.Row
        db.execute('CREATE TABLE proposition(idx INTEGER,pid TEXT,did TEXT,claim TEXT)')
        db.executemany('INSERT INTO proposition VALUES (?,?,?,?)',
                       [(i,f'p{i}',f'd{i}',f'text {i}') for i in range(5)])
        a=Namespace(output_root=root,preflight=False)
        contracts=[{'gpu':g,'shard_count':2} for g in (0,1)]
        for g in (0,1):
            rows=[{'index':i,'proposition_id':f'p{i}','document_id':f'd{i}',
                   'text':f'text {i}','contract':digest(contracts[g]),
                   'primary':{'choice':'DESCRIBE'},'secondary_labels':[]} for i in range(g,5,2)]
            atomic_jsonl(root/f'gpu{g}/labels.jsonl',rows)
            atomic_json(root/f'gpu{g}/summary.json',{'status':'complete','propositions':len(rows),
                        'missing_labels':0,'contract':digest(contracts[g])})
        return db,a,contracts

    def test_shard_query_and_checked_merge(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(full,'N',5):
            db,a,contracts=self.fixture(Path(tmp))
            self.assertEqual([r['idx'] for r in full.source_rows(db,dual.worker_args(a,0))],[0,2,4])
            self.assertEqual([r['idx'] for r in full.source_rows(db,dual.worker_args(a,1))],[1,3])
            p=SilentProgress(())
            dual.merge(a,db,contracts,'run',p)
            path=a.output_root/'labels.jsonl'
            original=path.read_text()
            self.assertEqual([json.loads(r)['index'] for r in original.splitlines()],list(range(5)))
            dual.merge(a,db,contracts,'run',p)
            self.assertEqual(path.read_text(),original)
            self.assertEqual(json.loads((a.output_root/'summary.json').read_text())['duplicate_ids'],0)
            db.close()

    def test_merge_rejects_wrong_identity_and_incomplete_shard(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(full,'N',5):
            db,a,contracts=self.fixture(Path(tmp))
            path=a.output_root/'gpu1/labels.jsonl'
            rows=[json.loads(r) for r in path.read_text().splitlines()]
            rows[0]['proposition_id']='wrong'
            atomic_jsonl(path,rows)
            with self.assertRaisesRegex(ValueError,'altered proposition'):
                dual.merge(a,db,contracts,'run',SilentProgress(()))
            self.assertFalse((a.output_root/'labels.jsonl').exists())
            summary=a.output_root/'gpu0/summary.json'
            value=json.loads(summary.read_text()); value['status']='incomplete'
            atomic_json(summary,value)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                dual.merge(a,db,contracts,'run',SilentProgress(()))
            db.close()


if __name__=='__main__': unittest.main()
