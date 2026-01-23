from abc import ABC, abstractmethod

class Plot(ABC):
    def __init__(self, missing_data, metadata=None):
        self.missing_data = missing_data
        self.metadata = metadata
        self.fig = None
        self.title = None

    @abstractmethod
    def _build_figure(self):
        pass

    def show(self):
        if self.fig is None:
            self._build_figure()
        self.fig.show()

    def save(self, path: str):
        if self.fig is None:
            self._build_figure()
        self.fig.write_html(path)
