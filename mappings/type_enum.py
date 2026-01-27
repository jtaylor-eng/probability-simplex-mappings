from enum import Enum

from .adaptative_temperature import AdaptiveSoftmax
from .alpha_entmax import AlphaEntmax
from .sotfmax import Softmax
from .sparsemax import Sparsemax
from .stieltjes import StieltjesTransform
from .stieltjes_learnable_q import LearnableStieltjes

class SimplexMappingEnum(Enum):
    softmax=Softmax
    stieltjes=StieltjesTransform
    stieltjes_learnable_q=LearnableStieltjes
    adaptive_temperature=AdaptiveSoftmax
    sparsemax=Sparsemax
    alpha_entmax=AlphaEntmax