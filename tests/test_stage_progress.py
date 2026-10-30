import json
import shutil
from pathlib import Path

import pytest
import torch

import train_macl_ddp as training
from tests.test_macl import TinyModel, config_for, data_root


def test_patience_counts_epochs_and_resets_on_improvement():
    progress = training.StageProgress(training.parse_args([]))
    assert progress.observe(.8, 0)
    for epoch in range(1, 5):
        assert not progress.observe(.8, epoch)
        assert progress.stop_reason() is None
    restored = training.StageProgress(training.parse_args([]))
    restored.load_state_dict(progress.state_dict())
    assert restored.bad_epochs == 4
    assert restored.observe(.81, 5)
    assert restored.bad_epochs == 0
    for epoch in range(6, 11):
        restored.observe(.80, epoch)
    assert restored.stop_reason() == 'no_improvement'
    restored.advance()
    assert restored.stage == 2 and restored.epochs == restored.bad_epochs == 0
    assert restored.best_score is None
    assert restored.history[0]['best_epoch'] == 6


def test_stage_delta_and_epoch_limit():
    config = training.parse_args(['--stage_min_delta', '.001', '--stage1_epochs', '2'])
    progress = training.StageProgress(config)
    progress.observe(.8, 0)
    assert not progress.observe(.8005, 1)
    assert progress.best_epoch == 1 and progress.stop_reason() == 'epoch_limit'
    with pytest.raises(FloatingPointError):
        progress.observe(float('nan'), 2)


def test_only_learned_modalities_participate_in_stage_score():
    tests = dict.fromkeys(['DUTS-TE', 'NJU2K', 'VT5000', 'DAVIS', 'RDVS', 'VDT'])
    assert list(training.tests_for_stage(1, tests)) == ['DUTS-TE']
    assert list(training.tests_for_stage(2, tests)) == ['DUTS-TE', 'NJU2K', 'VT5000', 'DAVIS']
    assert training.tests_for_stage(3, tests) == tests


def test_early_switch_requires_validation():
    with pytest.raises(SystemExit):
        training.parse_args(['--disable_validation'])
    assert training.parse_args(['--disable_validation', '--stage_patience', '0']).stage_patience == 0


def test_best_restore_early_switch_and_mid_stage_resume(data_root, monkeypatch):
    import models.SalMamba_tri as model_module
    monkeypatch.setattr(model_module, 'Model', TinyModel)
    # Two validations per epoch must still count as one patience epoch.
    def evaluate(model, tests, epoch, save_path, config):
        return {1: .8, 2: .7, 3: .6}[config._validation_stage]
    monkeypatch.setattr(training, 'evaluate_all_tests_and_save_best', evaluate)
    original_save = training.save_checkpoint
    snapshot = data_root / 'mid_stage_resume.pth'
    checked_boundary = []

    def save(*args, **kwargs):
        original_save(*args, **kwargs)
        if kwargs.get('filename', 'last_checkpoint.pth') != 'last_checkpoint.pth':
            return
        directory = Path(args[8].save_fold)
        if args[5] == 3 and not snapshot.exists():
            shutil.copyfile(directory / 'last_checkpoint.pth', snapshot)
        if args[5] == 5:
            current = torch.load(directory / 'last_checkpoint.pth', weights_only=False)
            best = torch.load(directory / 'stage1_best.pth', weights_only=False)
            assert current['stage_progress']['stage'] == 2
            assert current['stage_progress']['epochs'] == 0
            for name in best['model']:
                torch.testing.assert_close(current['model'][name], best['model'][name], atol=0, rtol=0)
            assert current['rank_states'][0]['buffers'] == best['rank_states'][0]['buffers']
            checked_boundary.append(True)
    monkeypatch.setattr(training, 'save_checkpoint', save)

    options = ['--epoch', '21', '--stage1_epochs', '7', '--stage2_epochs', '7',
               '--max_steps_per_epoch', '2', '--validation_frequency', '1']
    config = config_for(data_root, *options)
    training.run(config)
    directory = Path(config.save_fold)
    rows = [json.loads(line) for line in (directory / 'training_log_macl.jsonl').read_text().splitlines()]
    assert [row['stage'] for row in rows] == [1] * 6 + [2] * 6 + [3] * 6
    assert [row['no_improvement_epochs'] for row in rows[:6]] == list(range(6))
    assert checked_boundary
    final = torch.load(directory / 'last_checkpoint.pth', weights_only=False)
    best = torch.load(directory / 'stage3_best.pth', weights_only=False)
    assert final['stage_progress']['finished'] and final['epoch'] == 18
    assert [item['best_epoch'] for item in final['stage_progress']['history']] == [1, 7, 13]
    assert all(item['reason'] == 'no_improvement' for item in final['stage_progress']['history'])
    assert final['rank_states'][0]['buffers'] == best['rank_states'][0]['buffers']
    for name in final['model']:
        torch.testing.assert_close(final['model'][name], best['model'][name], atol=0, rtol=0)

    resumed = config_for(data_root, *options, '--resume', str(snapshot),
                         '--save_fold', str(data_root / 'resumed'))
    training.run(resumed)
    continued = torch.load(Path(resumed.save_fold) / 'last_checkpoint.pth', weights_only=False)
    assert continued['epoch'] == 18 and continued['stage_progress']['finished']
    assert continued['stage_progress']['history'] == final['stage_progress']['history']
    for name in final['model']:
        torch.testing.assert_close(final['model'][name], continued['model'][name], atol=0, rtol=0)


def test_resume_best_at_stage_limit_advances_before_training(data_root, monkeypatch):
    import models.SalMamba_tri as model_module
    monkeypatch.setattr(model_module, 'Model', TinyModel)
    monkeypatch.setattr(training, 'evaluate_all_tests_and_save_best', lambda *args: .8)
    config = config_for(data_root, '--max_steps_per_epoch', '2')
    training.run(config)
    directory = Path(config.save_fold)
    resumed = config_for(data_root, '--max_steps_per_epoch', '2',
                         '--resume', str(directory / 'stage1_best.pth'),
                         '--save_fold', str(data_root / 'resume_best'))
    training.run(resumed)
    rows = [json.loads(line) for line in (Path(resumed.save_fold) / 'training_log_macl.jsonl').read_text().splitlines()]
    assert [row['stage'] for row in rows] == [2, 3]
    final = torch.load(Path(resumed.save_fold) / 'last_checkpoint.pth', weights_only=False)
    original = torch.load(directory / 'last_checkpoint.pth', weights_only=False)
    assert final['stage_progress']['history'] == original['stage_progress']['history']
    for name in final['model']:
        torch.testing.assert_close(final['model'][name], original['model'][name], atol=0, rtol=0)


def test_rgb_batch_override_does_not_change_later_stages(data_root, monkeypatch):
    import models.SalMamba_tri as model_module
    monkeypatch.setattr(model_module, 'Model', TinyModel)
    monkeypatch.setattr(training, 'evaluate_all_tests_and_save_best', lambda *args: .8)
    config = config_for(data_root, '--rgb_batch_size', '3', '--max_steps_per_epoch', '2')
    loaders, _ = training.setup_all_dataloaders_for_joint_training(config)
    assert loaders['DUTS-TR'].batch_size == 3
    assert loaders['train_DUT'].batch_size == loaders['DVSOD_train'].batch_size == 2
    training.run(config)
    rows = [json.loads(line) for line in (Path(config.save_fold) / 'training_log_macl.jsonl').read_text().splitlines()]
    assert [row['batch_size'] for row in rows] == [3, 2, 2]
    assert [sum(row['samples'].values()) for row in rows] == [6, 4, 4]
