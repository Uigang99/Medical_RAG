import importlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from medrag import train_kg_data as d
import run_rag2_train_kg_stage as adapter


class TrainDataTests(unittest.TestCase):
    def upstream(self, root, stage, summary, records_name, records):
        ch=d.freeze(root/stage,{'fixture':stage})
        d.write_json(root/stage/'summary.json',summary)
        d.write_json(root/stage/'COMPLETE.json',dict(contract_sha256=ch,
            propositions=summary.get('structural_diagnostics',{}).get('propositions',0),
            assertions=summary.get('structural_diagnostics',{}).get('assertions',0),
            kept=summary.get('result',{}).get('kept',0)))
        d.write_rows(root/stage/records_name,records)

    def test_step3_through_step7_cpu_preflight(self):
        kg=ROOT/'datasets/kg'
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            discovery=next(d.rows(kg/'rag2_test6545_union88294_stage2_relation_discovery_v2/discovery_outputs.jsonl'))
            n=len(discovery['assertions']);src=discovery['source'];seg=discovery['segment']
            self.upstream(root,'step2',{'structural_diagnostics':{'propositions':1,'assertions':n},
                'scope':{'documents_with_usable_propositions':1},'by_source':{src:{'assertions':n}},
                'by_segment':{seg:{'assertions':n}}},'discovery_outputs.jsonl',[discovery])
            m=adapter.step3(root);sys.argv.append('--preflight-only');m.main()
            baseline=d.load(kg/'rag2_test6545_union88294_stage3_endpoint_binding_v1/manifest.json')['contract']
            self.assertEqual(d.load(root/'step3/manifest.json')['contract']['method'],baseline['method'])
            binding=next(d.rows(kg/'rag2_test6545_union88294_stage3_endpoint_binding_v1/binding_outputs.jsonl'))
            d.write_json(root/'step3/summary.json',{'scope':{'documents':1,'propositions':1},'structural_diagnostics':{'assertions':1}})
            d.write_json(root/'step3/COMPLETE.json',{'contract_sha256':d.load(root/'step3/manifest.json')['contract_sha256'],'assertions':1})
            d.write_rows(root/'step3/binding_outputs.jsonl',[binding])
            m=adapter.step4(root);sys.argv.append('--preflight-only');m.main()
            baseline=d.load(kg/'rag2_test6545_union88294_stage4_risk_routed_entailment_v1/manifest.json')['contract']
            self.assertEqual(d.load(root/'step4/manifest.json')['contract']['method'],baseline['method'])
            normalized=next(d.rows(kg/'rag2_test6545_union88294_stage5_contextual_entity_generalization_v4/normalized_frames.jsonl'))
            normalizer=importlib.import_module('normalize_rag2_stage5_contextual_union_full')
            raw=normalizer.strip_annotations(normalized)
            d.write_rows(root/'step4/graph_ready_surface_frames.jsonl',[raw])
            d.write_json(root/'step4/summary.json',{'result':{'kept':1},'scope':{'documents':1}})
            d.write_json(root/'step4/COMPLETE.json',{'contract_sha256':d.load(root/'step4/manifest.json')['contract_sha256'],'kept':1})
            m=adapter.step5(root,'inventory');sys.argv.append('--preflight-only');m.main()
            baseline=d.load(kg/'rag2_test6545_union88294_stage5_contextual_entity_generalization_v4/manifest.json')['contract']
            self.assertEqual(d.load(root/'step5/manifest.json')['contract']['method'],baseline['method'])
            mentions=sum(len(normalized['frame'].get(role,[])) for role in ('subjects','objects'))
            d.write_json(root/'step5/summary.json',{'scope':{'frames':1,'mentions':mentions},
                'relation_or_endpoint_mutations':0,'context_parse_errors_abstained':0,'missing_entity_nodes':0})
            d.write_json(root/'step5/COMPLETE.json',{'contract_sha256':d.load(root/'step5/manifest.json')['contract_sha256']})
            d.write_rows(root/'step5/normalized_frames.jsonl',[normalized])
            d.write_rows(root/'step5/normalized_mentions.jsonl',[])
            m=adapter.step67(root);sys.argv.append('--preflight-only');m.main()
            self.assertEqual(d.load(root/'kg/manifest.json')['contract']['scope']['target_frames'],1)
            sys.argv.remove('--preflight-only');m.main()
            self.assertTrue((root/'kg/COMPLETE.json').exists())

    def test_retrieval_contract_and_mismatch(self):
        for ds in d.COUNTS:
            m=d.load(d.cache_manifest(ds));d.verify_retrieval(m)
            m['candidate_pool_policy']['pool_size']=128
            with self.assertRaises(RuntimeError):d.verify_retrieval(m)

    def test_question_local_export_and_empty_graph_retention(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'kg').mkdir()
            c=sqlite3.connect(root/'kg/graph.sqlite3')
            c.execute('CREATE TABLE binary_relation(edge_id TEXT, proposition_id TEXT, document_id TEXT)')
            c.executemany('INSERT INTO binary_relation VALUES(?,?,?)',[('e1','p1','d1'),('outside','p2','d2')]);c.commit();c.close()
            d.write_rows(root/'cohort/questions.jsonl',[{'sample_id':'q1','split':'train'},{'sample_id':'q2','split':'validation'}])
            d.write_rows(root/'retrieval/question_document_membership.jsonl',[
                {'sample_id':'q1','rank':1,'stable_id':'d1'},{'sample_id':'q2','rank':1,'stable_id':'d3'}])
            d.write_rows(root/'usable/usable_propositions.jsonl',[{'document_id':'d1','proposition_id':'p1'}])
            d.export_question_views(root)
            a,b=list(d.rows(root/'question_views.jsonl'))
            self.assertEqual(a['binary_edge_ids'],['e1']);self.assertEqual(b['status'],'no_binary_edge')
            self.assertNotIn('answer',a)

    def test_frozen_usable_policy(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            good={'stable_id':'d1','source':'pubmed','source_status':'boundary_expanded_candidate',
                'source_quality_flags':[],'parse_status':'parsed','finish_reason':'stop','parse_errors':[],
                'source_text':'A inhibits B.','text_sha256':'example',
                'propositions':[{'claim':'A inhibits B.','evidence':'A inhibits B.','evidence_exact_match':True}]}
            bad=dict(good,stable_id='d2',source_quality_flags=['source warning'])
            d.write_rows(root/'propositions/document_propositions.jsonl',[good,bad])
            d.write_rows(root/'repair/machine_valid_candidates.jsonl',[])
            d.select_usable(root)
            records=list(d.rows(root/'usable/usable_propositions.jsonl'))
            self.assertEqual(len(records),1);self.assertEqual(records[0]['document_id'],'d1')
            audit=list(d.rows(root/'usable/document_audit.jsonl'))
            self.assertEqual(audit[1]['usable_propositions'],0)

    def test_manifest_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);d.freeze(p,{'a':1});d.freeze(p,{'a':1})
            with self.assertRaises(RuntimeError):d.freeze(p,{'a':2})

    def test_explicit_train_source_scope_preflight(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            source=ROOT/'datasets/kg/rag2_test6545_top1_2_4_8_16_sentence_restored_v1/restored_chunks.jsonl'
            record=next(d.rows(source))
            p=root/'restored.jsonl';d.write_rows(p,[record])
            scope=root/'scope.json';d.write_json(scope,{'documents':1,'by_source':{record['source']:1},'input_sha256':d.sha_file(p)})
            m=importlib.import_module('extract_rag2_restored_gemma_propositions')
            m.TwoLineProgress=adapter.EventProgress
            adapter.cli('--input',p,'--source-scope',scope,'--output-dir',root/'out','--batch-size','32','--preflight-only')
            m.main()
            c=d.load(root/'out/manifest.json')['contract']
            self.assertEqual(c['documents'],1);self.assertFalse(c['uses_question_answer_label'])
            self.assertEqual(c['effective_max_model_len'],16384)

    def test_step2_exact_preparation_and_frozen_prompt(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            source=ROOT/'datasets/kg/rag2_test6545_restored_union88294_gemma_faithful_v3_usable_v2/usable_propositions.jsonl'
            record=next(d.rows(source));record['segment']='top8'
            d.write_rows(root/'usable/usable_propositions.jsonl',[record])
            ch=d.freeze(root/'usable',{'fixture':True})
            d.write_json(root/'usable/summary.json',{'documents_audited':1,'documents_with_usable_propositions':1,
                'propositions':1,'by_source':{record['source']:1}})
            d.write_json(root/'usable/COMPLETE.json',{'contract_sha256':ch,'propositions':1})
            m=adapter.step2(root);sys.argv.append('--prepare-only');m.main()
            contract=d.load(root/'step2/manifest.json')['contract']
            baseline=d.load(ROOT/'datasets/kg/rag2_test6545_union88294_stage2_relation_discovery_v2/manifest.json')['contract']
            self.assertEqual(contract['stage2'],baseline['stage2'])
            self.assertEqual(contract['generation']['max_model_len'],5120)
            self.assertEqual(contract['scope']['documents_audited'],1)
            self.assertFalse((root/'step2/generation_shards').exists())


if __name__=='__main__':unittest.main()
