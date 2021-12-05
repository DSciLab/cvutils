from typing import Optional, List
from torch.utils.data.dataset import Dataset
from cvutils.transform.base import Transformer


class SchedulerTF:
    def __init__(
        self,
        dataset: Dataset,
        transformer: Transformer,
        batch_size: int
    ) -> None:
        self.dataset = dataset
        self.transformer = transformer
        self.batch_size = batch_size
        self._step_cnt = 0
        self._epoch_cnt = 0
        self._data_cnt = 0
        self.strength = 1.0

    def _apply_strength(self) -> None:
        self.transformer.call_apply_strength(self.strength)

    def _step_epoch(self) -> None:
        # be called per epoch
        raise NotImplementedError

    def step_epoch(self) -> None:
        self._epoch_cnt += 1
        self._step_epoch()

    def _step(self) -> None:
        # be called per step
        raise NotImplementedError

    def step(self) -> None:
        self._data_cnt += 1
        if self._data_cnt % self.batch_size == 0:
            self._step_cnt += 1
            self._step()
            if self._step_cnt % (len(self.dataset) // self.batch_size) == 0:
                self.step_epoch()


class MultiStepSchedulerTF(SchedulerTF):
    def __init__(
        self,
        dataset: Dataset,
        transformer: Transformer,
        batch_size: int,
        epoch_milestones: Optional[List[int]]=[],
        step_milestones: Optional[List[int]]=[],
        gamma: Optional[float]=0.1
    ) -> None:
        super().__init__(dataset, transformer, batch_size)
        assert len(epoch_milestones) != 0 or len(step_milestones) != 0
        self.epoch_milestones = epoch_milestones
        self.step_milestones = step_milestones
        self.gamma = gamma

    def _adjust_strength(self) -> None:
        self.strength *= self.gamma
        self._apply_strength()

    def _step(self) -> None:
        # be called per step
        if len(self.step_milestones) != 0:
            if self._step_cnt in self.step_milestones:
                self._adjust_strength()

    def _step_epoch(self) -> None:
        # be called per epoch
        if len(self.epoch_milestones) != 0:
            if self._epoch_cnt in self.epoch_milestones:
                self._adjust_strength()


class StepSchedulerTF(SchedulerTF):
    def __init__(
        self,
        dataset: Dataset,
        transformer: Transformer,
        batch_size: int,
        epoch_size: Optional[int]=-1,
        step_size: Optional[int]=-1,
        gamma: Optional[float]=0.1
    ) -> None:
        super().__init__(dataset, transformer, batch_size)
        assert epoch_size != -1 or step_size != -1
        self.epoch_size = epoch_size
        self.step_size = step_size
        self.gamma = gamma

    def _adjust_strength(self) -> None:
        self.strength *= self.gamma
        self._apply_strength()

    def _step(self) -> None:
        # be called per step
        if self.step_size != -1:
            if self._step_cnt % self.step_size == 0:
                self._adjust_strength()

    def _step_epoch(self) -> None:
        # be called per epoch
        if self.epoch_size != -1:
            if self._epoch_cnt % self.epoch_size == 0:
                self._adjust_strength()
