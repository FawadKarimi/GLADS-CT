"""Inactive TRAIN-only timing integration; never a completed training receipt."""
import statistics
import time
import torch
from .durable_lifecycle import DurableRun
from .objectives import class_weights_from_training_counts
from .reuse import canonical
from .training_runner import TrainingRunner


class TimingProbe(TrainingRunner):
    """One consumed budget slot, ending interrupted by design; no validation/TEST.

    Production bounds are fixed at 5 warm-up + 20 measured batches, 300 seconds.
    A separately authorized outer process-group timeout must cover native hangs.
    Tiny synthetic fixtures may use 1+1 to verify recording, never production.
    """
    def run(self, *, warmup=5, measured=20, wall_seconds=300):
        if self.used:
            raise ValueError('single probe attempt; no retry')
        self.used = True
        if self.spec['scope'] == 'production' and (warmup, measured, wall_seconds) != (5, 20, 300):
            raise ValueError('fixed production timing bounds required')
        if (type(warmup) is not int or type(measured) is not int or warmup < 1 or measured < 1
                or warmup+measured > 25 or type(wall_seconds) is not int or not 1 <= wall_seconds <= 300):
            raise ValueError('bounded probe required')
        if self.spec['contract']['seed'] != 42 or self.spec['contract']['objective'] != 'HARD':
            raise ValueError('seed-42 HARD timing only')
        self._admission()
        op = self.operations
        op.check_optimizer()
        if (self._fresh_spec() != self.spec or op.epoch or op.updates or op.optimizer.state
                or op.streams.last_epoch):
            raise ValueError('exact fresh timing state required; no resume')
        plan = op.streams.epoch_plan(self.adapter.inputs.ids('train'), epoch=1)
        if len(plan['batches']) < warmup+measured:
            raise ValueError('probe may not recycle training samples or widen input scope')
        session = DurableRun(spec=self.spec, inputs=self.adapter.inputs, budget=self.budget,
                             decision_sha256=self.decision)
        started = self.clock()
        records = []
        try:
            with session.guard():
                session.begin_epoch(plan)
                op.epoch = 1
                from .optimization import set_epoch_learning_rates
                set_epoch_learning_rates(op.optimizer, 1)
                self._updates()  # permit only after exclusive reservation
                weights = class_weights_from_training_counts([
                    sum(r.label == c for r in self.adapter.inputs.records('train')) for c in range(3)])
                offset = 0
                if op.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(op.device)
                for number, ids in enumerate(plan['batches'][:warmup+measured], 1):
                    if self.clock()-started >= wall_seconds:
                        raise TimeoutError('timing wall limit reached')
                    if op.device.type == 'cuda': torch.cuda.synchronize(op.device)
                    tick = self.clock()
                    batch = self.adapter.batch(ids, split='train', traces=plan['augmentation'][offset:offset+len(ids)])
                    _, losses, norm = op.train_batch(self._batch(batch), objective='HARD', weights=weights)
                    if op.device.type == 'cuda': torch.cuda.synchronize(op.device)
                    elapsed = self.clock()-tick
                    session.progress['elapsed_seconds'] = self.clock()-started
                    session.record_batch(ids, observed_optimizer_updates=op.updates)
                    row = {'batch': number, 'n': len(ids), 'phase': 'warmup' if number <= warmup else 'measured',
                           'wall_seconds_including_read_transform_transfer_update': elapsed,
                           'loss': losses, 'gradient_norm': norm}
                    records.append(row)
                    session.store.put('timing-batch-%03d.json'%number, canonical(row))
                    offset += len(ids)
                    if self.clock()-started >= wall_seconds:
                        raise TimeoutError('timing wall limit reached after observed update')
                values = [r['wall_seconds_including_read_transform_transfer_update'] for r in records if r['phase']=='measured']
                report = {'schema': 'pv2_training_only_timing_v1', 'warmup': warmup, 'measured': measured,
                          'mean_seconds': statistics.mean(values), 'median_seconds': statistics.median(values),
                          'maximum_seconds': max(values), 'training_batches_per_full_epoch': len(plan['batches']),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(op.device) if op.device.type=='cuda' else None,
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved(op.device) if op.device.type=='cuda' else None,
                          'device_binding': op.binding, 'validation_evaluated': False, 'test_accessed': False,
                          'training_complete': False, 'eligible_for_reuse': False,
                          'pilot_execution_authorized': False, 'production_execution_authorized': False}
                session.store.put('timing-summary.json', canonical(report))
                session.terminate('interrupted', reason='bounded_training_only_timing_probe_finished_not_complete')
                return report
        finally:
            op._admitted = False
