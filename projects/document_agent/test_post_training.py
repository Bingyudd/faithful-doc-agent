import math
import unittest

import torch
import torch.nn.functional as F

from post_training import dpo_loss, losses_from_logits, sequence_log_probs, sft_shift_loss, simpo_loss


class PostTrainingLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def tensors(self):
        logits = torch.randn(2,5,7,dtype=torch.float64,requires_grad=True)
        labels = torch.tensor([[-100,-100,2,3,-100],[-100,1,4,-100,5]])
        return logits,labels

    def test_shift_matches_manual_next_token_ce(self):
        logits,labels = self.tensors()
        expected = F.cross_entropy(logits[:,:-1].reshape(-1,7),labels[:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(sft_shift_loss(logits,labels),expected)

    def test_masked_prediction_positions_have_exact_zero_gradient(self):
        logits,labels = self.tensors()
        sft_shift_loss(logits,labels).backward()
        self.assertTrue(torch.equal(logits.grad[:,:-1][labels[:,1:]==-100],torch.zeros_like(logits.grad[:,:-1][labels[:,1:]==-100])))
        self.assertTrue(torch.equal(logits.grad[:,-1],torch.zeros_like(logits.grad[:,-1])))
        self.assertGreater(logits.grad[:,:-1][labels[:,1:]!=-100].abs().sum().item(),0)

    def test_all_ignored_batch_has_finite_zero_loss_and_gradient(self):
        logits,labels = self.tensors()
        loss = sft_shift_loss(logits,torch.full_like(labels,-100))
        loss.backward()
        self.assertEqual(loss.item(),0)
        self.assertTrue(torch.equal(logits.grad,torch.zeros_like(logits)))

    def test_token_and_sample_reductions_have_different_weights(self):
        logits,labels = self.tensors()
        raw = sft_shift_loss(logits,labels,reduction='none')
        counts = labels[:,1:].ne(-100).sum(-1)
        torch.testing.assert_close(sft_shift_loss(logits,labels,reduction='sample'),(raw.sum(-1)/counts).mean())
        torch.testing.assert_close(sft_shift_loss(logits,labels,reduction='token'),raw.sum()/counts.sum())
        self.assertNotAlmostEqual(sft_shift_loss(logits,labels,reduction='token').item(),sft_shift_loss(logits,labels,reduction='sample').item())

    def test_sequence_log_probs_equal_negative_ce_sum(self):
        logits,labels = self.tensors()
        logps,lengths = sequence_log_probs(logits,labels)
        raw = sft_shift_loss(logits,labels,reduction='none')
        torch.testing.assert_close(logps,-raw.sum(-1))
        normal,_ = sequence_log_probs(logits,labels,length_normalize=True)
        torch.testing.assert_close(normal,logps/lengths)

    def test_dpo_reference_correction_and_gradients(self):
        pc = torch.tensor([-2.0],requires_grad=True)
        pr = torch.tensor([-2.0],requires_grad=True)
        rc = torch.tensor([-2.0],requires_grad=True)
        rr = torch.tensor([-2.0],requires_grad=True)
        loss = dpo_loss(pc,pr,rc,rr,beta=0.2)
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)
        loss.backward()
        self.assertLess(pc.grad.item(),0)
        self.assertGreater(pr.grad.item(),0)
        self.assertIsNone(rc.grad)
        self.assertIsNone(rr.grad)

    def test_dpo_equal_policy_and_reference_margins_give_log_two(self):
        loss = dpo_loss(torch.tensor([-1.]),torch.tensor([-5.]),torch.tensor([-2.]),torch.tensor([-6.]),beta=0.3)
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)

    def test_dpo_preferring_chosen_reduces_loss(self):
        zero = torch.tensor([0.])
        self.assertLess(dpo_loss(torch.tensor([2.]),zero,zero,zero).item(),dpo_loss(zero,zero,zero,zero).item())

    def test_simpo_uses_lengths_and_unscaled_margin(self):
        cw,cr = torch.tensor([-4.]),torch.tensor([-6.])
        lw,lr = torch.tensor([2]),torch.tensor([3])
        loss = simpo_loss(cw,cr,lw,lr,beta=2,gamma=0.7)
        self.assertAlmostEqual(loss.item(),F.softplus(torch.tensor(0.7)).item(),places=6)

    def test_simpo_rejects_empty_targets(self):
        with self.assertRaises(ValueError):
            simpo_loss(torch.tensor([-1.]),torch.tensor([-2.]),torch.tensor([0]),torch.tensor([1]))

    def test_loss_functions_pass_autograd_gradcheck(self):
        logits,labels = self.tensors()
        self.assertTrue(torch.autograd.gradcheck(lambda x:sft_shift_loss(x,labels),(logits,),fast_mode=True))
        args=[torch.randn(2,dtype=torch.float64,requires_grad=True) for _ in range(2)]
        ref=[torch.zeros(2,dtype=torch.float64) for _ in range(2)]
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:dpo_loss(a,b,*ref),tuple(args),fast_mode=True))
        lengths=torch.tensor([2,3])
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:simpo_loss(a,b,lengths,lengths),tuple(args),fast_mode=True))

    def test_logits_to_preferences_is_differentiable(self):
        logits,labels = self.tensors()
        rejected=(logits.detach()+0.1*torch.randn_like(logits)).requires_grad_()
        result=losses_from_logits(logits,rejected,logits.detach(),rejected.detach(),labels,labels)
        (result['dpo']+result['simpo']).backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(rejected.grad)


if __name__=='__main__':
    unittest.main(verbosity=2)
