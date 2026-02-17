from enum import Enum

from .adaptative_temperature import AdaptiveSoftmax
from .alpha_entmax import AlphaEntmax
from .softmax import Softmax
from .sparsemax import Sparsemax
from .stieltjes import StieltjesTransform
from .as_entmax import AdaptiveScalableEntmax
from .scalable_softmax import ScalableSoftmax
from .topk_attn import TopKAttention
from .as_stieltjes import AdaptiveScalableStieltjes
from .scalable_stieltjes import ScalableStieltjes
from .topk_stieltjes import TopKStieltjes
from .adaptive_temperature_stieltjes import AdaptiveTemperatureStieltjes

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
    scalable_stieltjes = ScalableStieltjes
    topk_stieltjes = TopKStieltjes
    adaptive_temperature_stieltjes = AdaptiveTemperatureStieltjes