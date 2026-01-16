from enum import StrEnum

class SimplexMappingEnum(StrEnum):
    softmax='softmax'
    stieltjes='stieltjes'
    adaptive_temperature='adaptive'
    sparsemax='sparsemax'
    alpha_entmax='entmax'