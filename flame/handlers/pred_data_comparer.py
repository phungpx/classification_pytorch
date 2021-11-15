from ..module import Module
from ignite.engine import Events


class PredDataComparer(Module):
    def __init__(self, output_transform=lambda x: x):
        super(PredDataComparer, self).__init__()
        self.output_transform = output_transform

    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.frame['engine'].engine.add_event_handler(Events.ITERATION_COMPLETED, self.compare)

    def compare(self, engine):
        preds, targets, image_paths = self.output_transform(engine.state.output)
        for pred, target, image_path in zip(preds, targets, image_paths):
            if pred.argmax() != target:
                print(f'Target and prediction are different in {image_path}, pred: {pred.argmax()}, conf: {pred.max().item():.3f}, target: {target.item()}')
