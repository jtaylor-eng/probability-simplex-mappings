from enum import Enum

from .adaptative_temperature import AdaptiveSoftmax
from .alpha_entmax import AlphaEntmax
from .softmax import Softmax
from .sparsemax import Sparsemax
from .stieltjes import StieltjesTransform
from .stieltjes_learnable_q import LearnableStieltjes
from .as_entmax import AdaptiveScalableEntmax
from .scalable_softmax import ScalableSoftmax
from .topk_attn import TopKAttention
from .as_stieltjes import AdaptiveScalableStieltjes

class SimplexMappingEnum(Enum):
    softmax=Softmax
    scalable_softmax = ScalableSoftmax
    topk_attn = TopKAttention
    stieltjes = StieltjesTransform
    adaptive_temperature = AdaptiveSoftmax
    sparsemax = Sparsemax
    alpha_entmax = AlphaEntmax
    as_entmax = AdaptiveScalableEntmax
    as_stieltjes = AdaptiveScalableStieltjes