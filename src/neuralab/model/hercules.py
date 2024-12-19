#%%
from einops import rearrange
from flax import nnx
from neuralab.nn.hippo import HiPPO

class H0(nnx.Module):
    '''
        Sin autoevaluacion
    '''
    def __init__(self, num_features=32, *, L=128, rngs: nnx.Rngs):
        self.num_features = num_features

        self.hippo = HiPPO(N=num_features+1, L=L, dt=1/L, rngs=rngs)

        self.io_norm = nnx.BatchNorm(num_features, rngs=rngs)

    def __call__(self, x):
        feat = self.hippo.encode(x)
        dc, in_feat = rearrange(feat, 'l (1 f) -> l 1, l f', f=self.num_features)

        in_feat = self.io_norm(in_feat)

        out_feat = in_feat


        #self.hippo.decode(rearrange([dc, feat], 'l 1, l f -> l (1 f)', f=self.num_features))
#lvwp
