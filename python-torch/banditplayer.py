from abc import ABC, abstractmethod

class BanditPlayer(ABC):
    def __init__(self, n_arms):
        self.n_arms = n_arms

    @abstractmethod
    def experience(self, arm, reward) -> None:
        pass

    @abstractmethod
    def act(self) -> float:
        pass
