import abc


class NetworkComponent(metaclass=abc.ABCMeta):
    def __init__(self, aId: str, aType: str) -> None:
        self.aId = aId
        self.aType = aType
