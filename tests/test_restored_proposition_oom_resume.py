import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import extract_rag2_restored_gemma_propositions as job


class OOMResumeTest(unittest.TestCase):
    def test_failed_group_only_and_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            job.save(root/'manifest.json',dict(contract_sha256='same',contract=dict(effective_batch_size=32)))
            job.save(root/'generation_cursor.json',dict(contract_sha256='same',phase='generate',start=32,stop=64))
            self.assertTrue(job.reduce_batch_after_oom(root))
            self.assertEqual(job.runtime_batch(root,'same',32),16)
            state=json.loads((root/'oom_batch_limit.json').read_text())
            missing=[(s,s+8,root/f'{s}.jsonl') for s in [32,40,48,56,64,72]]
            groups=list(job.pending_groups(missing,16,state['retry_stop']))
            self.assertEqual([[s for s,_,_ in g] for g in groups],[[32,40],[48,56]])
            job.save(root/'generation_cursor.json',dict(contract_sha256='same',phase='generate',start=32,stop=48))
            self.assertTrue(job.reduce_batch_after_oom(root))
            self.assertEqual(job.runtime_batch(root,'same',32),8)
            self.assertEqual(json.loads((root/'oom_batch_limit.json').read_text())['retry_stop'],64)
            self.assertFalse(job.reduce_batch_after_oom(root))
            job.restore_batch(root,'same',32)
            self.assertEqual(job.runtime_batch(root,'same',32),32)
            self.assertIsNone(json.loads((root/'oom_batch_limit.json').read_text())['retry_stop'])

    def test_loading_oom_and_stale_contract_do_not_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            job.save(root/'manifest.json',dict(contract_sha256='same',contract=dict(effective_batch_size=32)))
            job.save(root/'generation_cursor.json',dict(contract_sha256='same',phase='load'))
            self.assertFalse(job.reduce_batch_after_oom(root))
            self.assertFalse((root/'oom_batch_limit.json').exists())
            job.save(root/'oom_batch_limit.json',dict(contract_sha256='other',batch_size=16))
            with self.assertRaises(RuntimeError):job.runtime_batch(root,'same',32)

    def test_cached_holes_and_tail_keep_order(self):
        spans=[(0,8,'a'),(16,24,'b'),(24,32,'c'),(40,43,'d'),(48,49,'e')]
        groups=list(job.pending_groups(spans,32))
        self.assertEqual(groups,[spans[:4],spans[4:]])


if __name__=='__main__':unittest.main()
