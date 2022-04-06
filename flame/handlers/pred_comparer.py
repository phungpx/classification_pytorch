import numpy as np
from ..module import Module
from ignite.engine import Events


class Comparer(Module):
    def __init__(self, is_binary=False, is_multiclass=False, is_multilabel=False, pred_threshold=0.5, output_transform=lambda x: x):
        super(Comparer, self).__init__()
        self.is_binary = is_binary
        self.is_multiclass = is_multiclass
        self.is_multilabel = is_multilabel
        self.pred_threshold = pred_threshold
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
                    print(f'\t - Pred: {np.round(pred).tolist()}')
                    print(f'\t - Target: {target}')
                    print(f'\t - Score: {np.round(pred, decimals=3).tolist()}')
            elif self.is_multiclass:
                if pred.argmax() != target:
                    print(f'Target and prediction are different in {image_path}')
                    print(f'\t - Prediction: {pred.argmax()}')
                    print(f'\t - Target: {target.item()}')
                    print(f'\t - Score: {pred.max().item():.3f}')
            elif self.is_binary:
                pred_class = int((pred > self.pred_threshold).to(target.dtype).item())
                target_class = int(target.item())
                if pred_class != target_class:
                    print(f'Target and prediction are different in {image_path}')
                    print(f'\t - Pred: {pred_class}')
                    print(f'\t - Target: {target_class}')
                    print(f'\t - Score: {pred.item():.3f}')
            else:
                raise ValueError('__Warining__ must be choose one of three type of classification.')
