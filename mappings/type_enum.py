from enum import Enum

from .adaptative_temperature import AdaptiveSoftmax
from .alpha_entmax import AlphaEntmax
from .sotfmax import Softmax
from .sparsemax import Sparsemax
from .stieltjes import StieltjesTransform


class SimplexMappingEnum(Enum):
    softmax=Softmax
    stieltjes=StieltjesTransform
    adaptive_temperature=AdaptiveSoftmax
    sparsemax=Sparsemax
    alpha_entmax=AlphaEntmax