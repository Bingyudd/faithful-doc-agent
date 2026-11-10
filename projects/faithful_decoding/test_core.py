import unittest
import torch
from .core import js_divergence,occurrence_attention,semantic_scores,boost_logits,DecodeConfig,prefill,_forward


class CoreTests(unittest.TestCase):
    def test_jsd_identity_symmetry_bounds(self):
        a=torch.tensor([1000.,-1000.]);b=-a
        self.assertEqual(js_divergence(a,a).item(),0)
        self.assertAlmostEqual(js_divergence(a,b).item(),1,places=6)
        self.assertEqual(js_divergence(a,b).item(),js_divergence(b,a).item())

    def test_repeated_occurrences_sum(self):
        ids,values=occurrence_attention(torch.tensor([.1,.2,.4]),torch.tensor([2,1,2]))
        self.assertEqual(ids.tolist(),[1,2])
        torch.testing.assert_close(values,torch.tensor([.2,.5]))

    def test_semantic_matches_pairwise(self):
        torch.manual_seed(4)
        emb=torch.randn(12,5);ids=torch.tensor([2,2,4,7])
        unique,score=semantic_scores(emb,ids)
        expected=torch.nn.functional.cosine_similarity(emb[unique][:,None,:],emb[ids][None,:,:],dim=-1).mean(-1)
        torch.testing.assert_close(score,expected)

    def test_author_semantic_mapping_is_explicit(self):
        emb=torch.tensor([[1.,0.],[-1.,0.],[0.,1.]])
        ids=torch.tensor([0,2,2])
        _,raw=semantic_scores(emb,ids,'paper_equations')
        _,mapped=semantic_scores(emb,ids,'author_reference')
        torch.testing.assert_close(mapped,(raw+1)/2)
        self.assertTrue(bool(((mapped>=0)&(mapped<=1)).all()))

    def test_zero_bias_and_empty_source(self):
        logits=torch.randn(10)
        torch.testing.assert_close(boost_logits(logits,torch.tensor([2,5]),0),logits)
        torch.testing.assert_close(boost_logits(logits,torch.tensor([],dtype=torch.long),5),logits)

    def test_boost_only_source_and_mean_scale(self):
        out=boost_logits(torch.zeros(8),torch.tensor([2,5]),2,torch.tensor([1.,3.]),torch.tensor([1.,1.]),1)
        torch.testing.assert_close(out,torch.tensor([0.,0.,1.,0.,0.,3.,0.,0.]))

    def test_invalid_normalization_rejected(self):
        with self.assertRaises(ValueError):boost_logits(torch.zeros(3),torch.tensor([1]),2,torch.zeros(1),torch.zeros(1))

    def test_config_validation(self):
        for kwargs in ({'method':'unknown'},{'max_new_tokens':0},{'attention_weight':1.1},{'delta_min':11}):
            with self.assertRaises(ValueError):DecodeConfig(**kwargs)

    def test_real_qwen_cache_and_last_query_attention(self):
        from transformers import Qwen2Config,Qwen2ForCausalLM
        torch.manual_seed(9)
        config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                          num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
        config._attn_implementation='eager'
        model=Qwen2ForCausalLM(config).eval()
        ids=torch.tensor([[2,4,8,3,9]])
        with torch.no_grad():
            full=model(ids,output_attentions=True,use_cache=False)
            cached=prefill(model,ids,True)
            torch.testing.assert_close(cached.logits[:,-1],full.logits[:,-1],atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(cached.attentions[-1],full.attentions[-1][:,:,-1:,:],atol=1e-6,rtol=1e-5)
            next_out=_forward(model,torch.tensor([[7]]),cached.past_key_values,True)
            whole=model(torch.cat([ids,torch.tensor([[7]])],1),use_cache=False)
            torch.testing.assert_close(next_out.logits[:,-1],whole.logits[:,-1],atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
