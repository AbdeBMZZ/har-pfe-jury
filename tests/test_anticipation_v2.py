import unittest,tempfile,copy
from pathlib import Path
import numpy as np
import torch
from src.data.anticipation_dataset import AnticipationDataset
from src.data.temporal_protocol import TemporalAnticipationDataset
from src.models.causal_forecaster import CausalForecaster,window_features
from src.models.har_model import build_model

class AnticipationTests(unittest.TestCase):
    def test_legacy_builder_kept_for_demo_but_is_not_causal_protocol(self):
        # Jury package keeps the audited legacy AnticipationDataset for historical demos.
        # Validated causal evaluation requires sessions via TemporalAnticipationDataset.
        ds = AnticipationDataset(np.zeros((20, 150, 6)), np.ones(20), np.ones(20), seq_len=5)
        self.assertGreater(len(ds), 0)
        with self.assertRaisesRegex(TypeError, 'sessions|required|missing'):
            TemporalAnticipationDataset(np.zeros((20, 150, 6)), np.ones(20), np.ones(20))
    def test_feature_shape_and_future_not_read(self):
        x=np.random.default_rng(2).normal(size=(2,5,150,6)).astype(np.float32)
        expected=window_features(x[:,:,:75]);x[:,:,75:]=1e5
        np.testing.assert_array_equal(expected,window_features(x[:,:,:75]))
        self.assertEqual(expected.shape,(2,5,36))
        with self.assertRaises(ValueError):window_features(np.zeros((1,1,6)))
    def test_checkpoint_reload_and_gradient(self):
        torch.manual_seed(2);m=CausalForecaster();x=torch.randn(8,5,36);y=torch.arange(8)%12
        opt=torch.optim.AdamW(m.parameters(),lr=.01)
        for _ in range(3):opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward();opt.step()
        self.assertTrue(torch.isfinite(loss));m.eval();expected=m(x)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.pt';torch.save(m.state_dict(),p);q=CausalForecaster();q.load_state_dict(torch.load(p,weights_only=True));q.eval();torch.testing.assert_close(expected,q(x))
    def test_recognition_weights_are_independent(self):
        torch.manual_seed(2);recognizer=build_model(d_model=16,n_blocks=1,n_heads=2);before=copy.deepcopy(recognizer.state_dict())
        ant=CausalForecaster();opt=torch.optim.AdamW(ant.parameters());loss=ant(torch.ones(2,5,36)).sum();loss.backward();opt.step()
        for key,value in before.items():torch.testing.assert_close(value,recognizer.state_dict()[key])
    def test_no_cross_recording_even_with_same_subject(self):
        x=np.zeros((12,150,6),np.float32);y=np.ones(12,int);sub=np.ones(12,int)
        sessions=np.repeat(['a','b'],6);starts=np.tile(np.arange(6)*75,2)
        ds=TemporalAnticipationDataset(x,y,sub,sessions,starts,seq_len=5)
        self.assertEqual(len(ds),0)  # each recording too short for context plus future
    def test_online_prefixes_match_offline_filter_and_chunk_boundaries(self):
        from src.data.causal_stream import CausalWindowStream
        from src.data.temporal_protocol import causal_imu
        raw=np.random.default_rng(3).normal(size=(2200,6))
        expected=CausalWindowStream().push(raw)
        stream=CausalWindowStream();actual=[]
        for chunk in np.array_split(raw,37):actual.extend(stream.push(chunk))
        self.assertEqual(len(expected),len(actual))
        filtered=causal_imu(raw,'g')
        for a,b in zip(expected,actual):
            np.testing.assert_allclose(a['context'],b['context'],atol=1e-5)
            end=a['decision_sample'];np.testing.assert_allclose(a['context'][-1],filtered[end-75:end],atol=1e-5)
            self.assertGreater(a['target_start'],end)
        stream.reset();self.assertEqual(stream.total,0)
        self.assertEqual(len(stream.context),0)

if __name__=='__main__':unittest.main()
