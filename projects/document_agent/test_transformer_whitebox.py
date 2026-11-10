import unittest

import torch
import torch.nn.functional as F

from transformer_whitebox import gqa_attention,kv_cache_bytes,rms_norm,rope,swiglu,training_state_bytes


class TransformerWhiteboxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(2)

    def test_rms_norm_unit_rms_and_gradient(self):
        x=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
        weight=torch.ones(4,dtype=torch.float64,requires_grad=True)
        y=rms_norm(x,weight,eps=0)
        torch.testing.assert_close(y.square().mean(-1),torch.ones(2,3,dtype=torch.float64))
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:rms_norm(a,b),(x,weight),fast_mode=True))

    def test_rope_preserves_norm(self):
        x=torch.randn(2,3,5,8,dtype=torch.float64)
        positions=torch.arange(5)
        torch.testing.assert_close(rope(x,positions).square().sum(-1),x.square().sum(-1))
        torch.testing.assert_close(rope(x,torch.zeros(5)),x)

    def test_rope_relative_dot_product(self):
        q,k=torch.randn(1,1,1,8,dtype=torch.float64),torch.randn(1,1,1,8,dtype=torch.float64)
        left=(rope(q,torch.tensor([3]))*rope(k,torch.tensor([7]))).sum()
        right=(q*rope(k,torch.tensor([4]))).sum()
        torch.testing.assert_close(left,right)

    def test_swiglu_gradcheck_and_zero_gate(self):
        x=torch.randn(1,2,3,dtype=torch.float64,requires_grad=True)
        gate=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
        up=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
        down=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(swiglu,(x,gate,up,down),fast_mode=True))
        self.assertTrue(torch.equal(swiglu(x,torch.zeros_like(gate),up,down),torch.zeros_like(x)))

    def test_gqa_matches_expanded_head_sdpa(self):
        q=torch.randn(1,4,5,4,dtype=torch.float64)
        k,v=torch.randn(1,2,5,4,dtype=torch.float64),torch.randn(1,2,5,4,dtype=torch.float64)
        expected=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),is_causal=True)
        torch.testing.assert_close(gqa_attention(q,k,v),expected)

    def test_causal_attention_cannot_see_future_values(self):
        q=torch.randn(1,4,5,4,dtype=torch.float64)
        k,v=torch.randn(1,2,5,4,dtype=torch.float64),torch.randn(1,2,5,4,dtype=torch.float64)
        changed=v.clone();changed[:,:,3:]+=1000
        torch.testing.assert_close(gqa_attention(q,k,v)[:,:,:3],gqa_attention(q,k,changed)[:,:,:3])

    def test_incremental_kv_cache_matches_full_sequence(self):
        q=rope(torch.randn(1,4,6,4,dtype=torch.float64),torch.arange(6))
        k=rope(torch.randn(1,2,6,4,dtype=torch.float64),torch.arange(6))
        v=torch.randn(1,2,6,4,dtype=torch.float64)
        full=gqa_attention(q,k,v)
        incremental=torch.cat([gqa_attention(q[:,:,i:i+1],k[:,:,:i+1],v[:,:,:i+1]) for i in range(6)],dim=2)
        torch.testing.assert_close(full,incremental)

    def test_kv_memory_matches_actual_tensor_storage(self):
        batch,layers,tokens,heads,dim=2,3,7,2,4
        k=torch.zeros(layers,batch,heads,tokens,dim,dtype=torch.float16)
        measured=2*k.numel()*k.element_size()
        self.assertEqual(kv_cache_bytes(batch=batch,layers=layers,tokens=tokens,kv_heads=heads,head_dim=dim),measured)
        mha=kv_cache_bytes(batch=1,layers=2,tokens=16,kv_heads=8,head_dim=4)
        mqa=kv_cache_bytes(batch=1,layers=2,tokens=16,kv_heads=1,head_dim=4)
        self.assertEqual(mha,8*mqa)

    def test_training_state_memory_lists_assumptions(self):
        result=training_state_bytes(100)
        self.assertEqual(result['total'],1600)
        self.assertEqual(result['adam_first_moment'],400)
        self.assertIn('activations',result['excludes'])

    def test_invalid_gqa_heads_and_cache_dimensions(self):
        with self.assertRaises(ValueError):
            gqa_attention(torch.zeros(1,3,2,4),torch.zeros(1,2,2,4),torch.zeros(1,2,2,4))
        with self.assertRaises(ValueError):
            kv_cache_bytes(batch=0,layers=1,tokens=1,kv_heads=1,head_dim=4)


if __name__=='__main__':
    unittest.main(verbosity=2)
