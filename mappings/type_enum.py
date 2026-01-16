from enum import StrEnum

from adaptative_temperature import AdaptiveSoftmax
from alpha_entmax import AlphaEntmax
from sotfmax import Softmax
from sparsemax import Sparsemax
from stieltjes import StieltjesTransform


class SimplexMappingEnum(StrEnum):
    softmax=Softmax
    stieltjes=StieltjesTransform
    adaptive_temperature=AdaptiveSoftmax
    sparsemax=Sparsemax
    alpha_entmax=AlphaEntmax