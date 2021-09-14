from typing import Any
from .utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from .randaug import RandAugment
from .builder import PIPELINES

@PIPELINES.register_module("pixelbert")
class pixelbert_transform:
    def __init__(self,size=800) -> None:
        
        longer = int((1333 / 800) * size)
        self.func = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ])
    def __call__(self, x):
        return self.func(x)
    


@PIPELINES.register_module("pixelbert_randaug")
class pixelbert_transform_randaug:
    def __init__(self,size=800) -> None:
        longer = int((1333 / 800) * size)
        self.trs = transforms.Compose(
            [
                MinMaxResize(shorter=size, longer=longer),
                transforms.ToTensor(),
                inception_normalize,
            ]
        )
        self.trs.transforms.insert(0, RandAugment(2, 9))
    
    def __call__(self, x: Any) -> Any:
        return self.trs(x)
