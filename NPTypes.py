import numpy.typing as npt
from nptyping import NDArray, Shape, Float, Integer, Number, Obj

NPArrayNxM = NDArray[Shape["N, M"], Obj]

NumNPArrayNxM = NDArray[Shape["N, M"], Number]
NumNPArray = NDArray[Shape["N"], Number]

FloatNPArrayNxM = NDArray[Shape["N, M"], Float]
FloatNPArrayNxN = NDArray[Shape["N, N"], Float]
FloatNPArray = NDArray[Shape["N"], Float]

IntNPArrayNxM = NDArray[Shape["N, M"], Integer]
IntNPArray = NDArray[Shape["N"], Integer]

ArrayLike = npt.ArrayLike
