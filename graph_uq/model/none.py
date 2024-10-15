from graph_uq.data.data import Data
from graph_uq.model.base import BaseModel
from graph_uq.model.prediction import Prediction


class NoModel(BaseModel):
    """Stub for using no model"""

    def reset_cache(self): ...

    def reset_parameters(self): ...

    def forward(self, batch: Data) -> Prediction:
        return Prediction()

    @property
    def prediction_changes_at_eval(self) -> bool:
        return False
