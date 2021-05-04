from abc import ABC, abstractmethod
from data.datasources.datasource_mode import DataSourceMode


class BaseDatasource(ABC):
    def __init__(self, config, mode: DataSourceMode):
        self.config = config
        self.mode = mode
        self.data = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def compute_length(self) -> int:
        pass

    @abstractmethod
    def get_item(self, index: int):
        pass
