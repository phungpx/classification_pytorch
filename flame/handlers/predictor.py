import numpy as np
from ..module import Module
from ignite.engine import Events


class Predictor(Module):
    def __init__(self, is_multilabel=False, output_transform=lambda x: x):
        super(Predictor, self).__init__()
        self.is_multilabel = is_multilabel
        self.output_transform = output_transform

    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.frame['engine'].engine.add_event_handler(Events.ITERATION_COMPLETED, self.compare)

    def compare(self, engine):
        preds, targets, image_paths = self.output_transform(engine.state.output)
        for pred, target, image_path in zip(preds, targets, image_paths):
            if self.is_multilabel:
                pred = pred.data.cpu().numpy().tolist()
                target = target.data.cpu().numpy().tolist()
                if np.round(pred).tolist() != target:
                    print(f'Target and prediction are different in {image_path}')
                    print(f'\t -pred: {np.round(pred).tolist()}')
                    print(f'\t -conf: {np.round(pred, decimals=3).tolist()}')
                    print(f'\t -target: {target}')
            else:
                if pred.argmax() != target:
                    print(f'Target and prediction are different in {image_path}')
                    print(f'\t -pred: {pred.argmax()}')
                    print(f'\t -conf: {pred.max().item():.3f}')
                    print(f'\t -target: {target.item()}')
