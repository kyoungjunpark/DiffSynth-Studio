import os
import tempfile
import torch
from diffsynth.trainers.utils import ModelLogger


class FakeAccelerator:
    def __init__(self, save_behavior='normal'):
        self.is_main_process = True
        self.process_index = 0
        self._save_behavior = save_behavior

    def wait_for_everyone(self):
        return

    def get_state_dict(self, model):
        # return a simple state dict with a tensor so torch.save works
        return {'weight': torch.tensor([1.0])}

    def unwrap_model(self, model):
        class Unwrapper:
            def export_trainable_state_dict(self, sd, remove_prefix=None):
                return sd
        return Unwrapper()

    def save(self, obj, path, safe_serialization=True):
        if self._save_behavior == 'error':
            raise IOError('simulated save error')
        # Use torch.save to simulate accelerator.save
        torch.save(obj, path)


def test_save_ok():
    out = tempfile.mkdtemp()
    logger = ModelLogger(out)
    acc = FakeAccelerator()
    logger.save_model(acc, None, 'step-1.safetensors')
    final = os.path.join(out, 'step-1.safetensors')
    assert os.path.exists(final), f"Expected saved file: {final}"
    print('test_save_ok: PASS')


def test_save_creates_dir():
    base = tempfile.mkdtemp()
    nested = os.path.join(base, 'a', 'b', 'c')
    logger = ModelLogger(nested)
    acc = FakeAccelerator()
    logger.save_model(acc, None, 'step-2.safetensors')
    final = os.path.join(nested, 'step-2.safetensors')
    assert os.path.exists(final), f"Expected saved file in nested dir: {final}"
    print('test_save_creates_dir: PASS')


def test_save_error_wrap():
    out = tempfile.mkdtemp()
    logger = ModelLogger(out)
    acc = FakeAccelerator(save_behavior='error')
    try:
        logger.save_model(acc, None, 'step-3.safetensors')
    except RuntimeError as e:
        print('test_save_error_wrap: PASS (caught RuntimeError)')
        print('  error:', e)
    else:
        raise RuntimeError('Expected RuntimeError when accelerator.save fails')


def test_on_step_end_triggers_save_step1():
    out = tempfile.mkdtemp()
    logger = ModelLogger(out)
    acc = FakeAccelerator()
    # simulate one step: set num_steps so that on_step_end will increment to 1
    logger.num_steps = 0
    # call on_step_end with save_steps=1 should trigger save_model
    logger.on_step_end(acc, None, save_steps=1)
    final = os.path.join(out, 'step-1.safetensors')
    assert os.path.exists(final), f"Expected saved file at step-1: {final}"
    print('test_on_step_end_triggers_save_step1: PASS')


if __name__ == '__main__':
    test_save_ok()
    test_save_creates_dir()
    test_save_error_wrap()
