"""Invented data only. Trust-pin monkeypatches below are test fixtures, not runtime options."""
import copy
import csv
import io
import json
from pathlib import Path
import zipfile

import pytest

from consendxformer import r3_entropy as runner
from consendxformer import r3_entropy_source as source
from consendxformer.hashing import sha256_file, sha256_json


def rows_fixture(n=183):
    rows = []
    for model in source.MODELS:
        for seed in source.SEEDS:
            for i in range(n):
                true = source.CLASSES[i % 3]
                rows.append(dict(patient_id=f"invented-p{i % 99:03}", series_instance_uid=f"series{i}",
                    nodule_id=f"invented-n{i:03}", split="test", true_class=true,
                    predicted_class=true if (i + seed) % 2 else source.CLASSES[(i + 1) % 3],
                    seed=str(seed), model=model, experiment_id=source.EXPERIMENTS[(model, seed)],
                    predictive_entropy_nats=str((i % 10) / 10), disagreement_entropy="0.4",
                    disagreement_group="low", disagreement_threshold="0.5"))
    return rows


def paired_fixture(rows):
    mapping = {(r['model'], r['seed'], r['nodule_id']): r for r in rows}
    paired = []
    for row in rows:
        if row['model'] != 'full':
            continue
        item = {field: row[field] for field in (*source.IDENTITY, 'seed', 'true_class')}
        for suffix, model in [('full', 'full'), ('off', 'disagreement_weighting_off')]:
            original = mapping[(model, row['seed'], row['nodule_id'])]
            for field in ('predicted_class', 'experiment_id', 'predictive_entropy_nats'):
                item[field + '_' + suffix] = original[field]
        paired.append(item)
    return paired


def provenance_fixture():
    sources = []
    for seed in source.SEEDS:
        fi = dict(experiment_id=source.EXPERIMENTS[('full', seed)], seed=seed, model='ConsenDxFormer',
            project_commit='e1b174490245c616e355ef7c0c6f7c3c239d3acd', completed=True,
            run_mode='final', test_used_for_selection=False, lambda_ordinal=.5, lambda_attention=.05,
            class_order=list(source.CLASSES), cohort_view='paper_aligned_multi_reader_2_to_4',
            manifest_bundle_sha256='a'*64, supervision_sha256='b'*64, ordered_supervision_sha256='c'*64,
            sample_count=1133, test_sample_count=183, phase3_input_fingerprint='d'*64,
            input_lock_sha256='e'*64, artifact_manifest_sha256='f'*64)
        full = None
        for model in source.MODELS:
            directory = '/nonexistent-do-not-open/' + source.EXPERIMENTS[(model, seed)]
            snapshot = {directory + '/checkpoints/best.pt': '1'*64,
                        directory + '/predictions/test.csv': '2'*64,
                        directory + '/artifact_manifest.json': 'f'*64}
            snapshot.update({directory + f'/invented/{i}': 'a'*64 for i in range(21 if model == 'full' else 17)})
            if model == 'full':
                identity = fi
            else:
                identity = dict(experiment_id=source.EXPERIMENTS[(model, seed)], seed=seed,
                    full_model_training_commit=fi['project_commit'], variant='disagreement_weighting',
                    lambda_ordinal=.5, lambda_attention=.05, input_lock_sha256=fi['input_lock_sha256'],
                    inputs={f: fi[f] for f in ('cohort_view', 'manifest_bundle_sha256', 'supervision_sha256',
                        'ordered_supervision_sha256', 'sample_count', 'phase3_input_fingerprint')},
                    toggles=dict(use_attention_alignment=True, use_disagreement_weighting=False,
                                 use_hierarchical_fusion=True, use_ordinal_head=True),
                    full_model_reference=dict(source_dir=full['source_dir'], descriptor=fi, best_checkpoint_sha256='1'*64))
                identity['inputs']['split_counts'] = dict(train=807, val=143, test=183)
            record = dict(identity=identity, source_dir=directory, snapshot=snapshot,
                          best_checkpoint_sha256='1'*64, test_predictions_sha256='2'*64)
            sources.append(record)
            if model == 'full':
                full = record
    return dict(analysis='uncertainty', seeds=list(source.SEEDS), sources=sources,
                input_snapshot={k: v for s in sources for k, v in s['snapshot'].items()})


@pytest.fixture
def synthetic_archive(tmp_path, monkeypatch):
    rows = rows_fixture()
    provenance = provenance_fixture()
    eid = 'phase4-evaluation-' + sha256_json(provenance)[:16]
    data = {member: b'invented evidence\n' for member in source.MEMBERS}
    data['provenance/provenance.json'] = runner._json_bytes(provenance)
    data['evaluation.json'] = runner._json_bytes(dict(evaluation_id=eid, analysis='uncertainty'))
    data['tables/uncertainty_per_nodule_per_seed.csv'] = runner._csv_bytes(rows)
    data['tables/paired_per_nodule_per_seed.csv'] = runner._csv_bytes(paired_fixture(rows))
    hashes = {k: source.digest(v) for k, v in data.items() if k not in ('artifact_manifest.json', 'COMPLETED.json')}
    data['artifact_manifest.json'] = runner._json_bytes(dict(evaluation_id=eid, files=hashes, bundle_sha256=sha256_json(hashes)))
    data['COMPLETED.json'] = runner._json_bytes(dict(evaluation_id=eid,
        artifact_manifest_sha256=source.digest(data['artifact_manifest.json']),
        source_artifacts_unchanged=True, training_performed=False, exploratory_post_hoc=True))
    path = tmp_path / 'invented.zip'
    write_zip(path, data, eid)
    # Prove that production loader rejects invented archives BEFORE test-only trust substitution.
    with pytest.raises(ValueError, match='archive SHA'):
        source.load_archive(path)
    monkeypatch.setattr(source, 'ARCHIVE_SHA256', sha256_file(path))
    monkeypatch.setattr(source, 'EVALUATION_ID', eid)
    monkeypatch.setattr(source, 'PINS', {k: source.digest(data[k]) for k in source.PINS})
    monkeypatch.setattr(source, 'PAYLOAD_DIGEST', sha256_json(hashes))
    monkeypatch.setattr(source, 'COHORT_SHA256', sha256_json(source.cohort_ledger(rows[:183])))
    proposal = tmp_path / 'proposal.md'
    proposal.write_bytes(b'invented proposal\n')
    monkeypatch.setattr(source, 'PROPOSAL_SHA256', sha256_file(proposal))
    protocol = copy.deepcopy(runner.PROTOCOL)
    protocol.update(archive_sha256=source.ARCHIVE_SHA256, member_pins=source.PINS,
                    payload_digest=source.PAYLOAD_DIGEST, cohort_sha256=source.COHORT_SHA256,
                    proposal_sha256=source.PROPOSAL_SHA256)
    monkeypatch.setattr(runner, 'PROTOCOL', protocol)
    return path, proposal, data, rows


def write_zip(path, data, eid, extra=None):
    with zipfile.ZipFile(path, 'w') as z:
        for member, content in sorted(data.items()):
            info = zipfile.ZipInfo(eid + '/' + member, (2020, 1, 1, 0, 0, 0))
            z.writestr(info, content)
        if extra:
            z.writestr(extra, b'bad')


def namespace(root):
    return root / 'artifacts/phase5/r3_entropy'


def test_summary_policy_and_traceability():
    rows = rows_fixture(4)
    original = copy.deepcopy(rows)
    result = runner.summarize(rows)
    assert rows == original
    assert len(result['summaries']) == 24 and len(result['contrasts']) == 12
    assert len(result['memberships']) == 48
    for membership in result['memberships']:
        row = rows[membership['source_row_index']]
        assert membership['stored_entropy_text'] == row['predictive_entropy_nats']
        assert membership['nodule_id'] == row['nodule_id']
    full42 = [r for r in result['summaries'] if r['model'] == 'full' and r['seed'] == 42]
    correct = next(r for r in full42 if r['group'] == 'correct')
    assert correct['mean_nats'] == pytest.approx(.2)
    assert correct['q1_nats'] == pytest.approx(.15)
    assert correct['median_nats'] == pytest.approx(.2)
    assert correct['q3_nats'] == pytest.approx(.25)
    assert correct['sample_sd_nats'] == pytest.approx(2**.5 / 10)
    indeterminate = next(r for r in full42 if r['group'] == 'indeterminate')
    assert indeterminate['nodule_count'] == 1 and indeterminate['sample_sd_nats'] is None
    for contrast in result['contrasts']:
        if contrast['axis'] == 'correctness' and contrast['seed'] == 42:
            assert contrast['mean_difference_nats'] == pytest.approx(-.1)
    # Groups change across seeds, true-class membership does not.
    a, b = [r for r in result['memberships'] if r['model'] == 'full' and r['nodule_id'] == 'invented-n000'
            and r['axis'] == 'correctness' and r['seed'] in (42, 43)]
    assert a['group'] != b['group']


def test_empty_constant_and_patient_overlap():
    rows = rows_fixture(4)
    for row in rows:
        row.update(predicted_class=row['true_class'], predictive_entropy_nats='0.5000', patient_id='same-patient')
    result = runner.summarize(rows)
    for row in result['summaries']:
        if row['group'] == 'incorrect':
            assert row['status'] == 'empty_group' and row['mean_nats'] is None and row['nodule_count'] == 0
        if row['group'] == 'correct':
            assert row['sample_sd_nats'] == 0 and row['patient_count'] == 1
    for row in result['contrasts']:
        if row['axis'] == 'correctness':
            assert row['mean_difference_nats'] is None and row['reason'] == 'empty_comparison_group'
    for row in result['patient_overlap']:
        if row['axis'] == 'true_class':
            assert row['patient_overlap_ids'] == ['same-patient'] and row['patient_union_count'] == 1


@pytest.mark.parametrize('bad', ['', 'NaN', 'inf', '-inf', 'bad', '-0.1', '1.2', None])
def test_invalid_entropy(bad):
    rows = rows_fixture(4)
    rows[0]['predictive_entropy_nats'] = bad
    with pytest.raises(ValueError, match='source row 0'):
        runner.summarize(rows)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'class', 'cohort', 'paired', 'seed', 'experiment',
                                    'paired_missing', 'paired_duplicate', 'true_label', 'disagreement'])
def test_record_corruption(synthetic_archive, mutation):
    _, _, _, rows = synthetic_archive
    paired = paired_fixture(rows)
    if mutation == 'missing': rows.pop()
    if mutation == 'duplicate': rows[1] = rows[0].copy()
    if mutation == 'class': rows[0]['true_class'] = ''
    if mutation == 'cohort': rows[200]['patient_id'] = 'different'
    if mutation == 'paired': paired[0]['predicted_class_off'] = 'invalid'
    if mutation == 'seed': rows[0]['seed'] = '42.0'
    if mutation == 'experiment': rows[0]['experiment_id'] = 'other'
    if mutation == 'paired_missing': paired.pop()
    if mutation == 'paired_duplicate': paired[1] = paired[0].copy()
    if mutation == 'true_label':
        rows[200]['true_class'] = source.CLASSES[(source.CLASSES.index(rows[200]['true_class']) + 1) % 3]
    if mutation == 'disagreement': rows[200]['disagreement_threshold'] = '0.6'
    with pytest.raises(ValueError): source.validate_rows(rows, paired)


@pytest.mark.parametrize('mutation', ['payload', 'completion', 'provenance', 'extra', 'duplicate'])
def test_archive_corruption(synthetic_archive, monkeypatch, mutation):
    path, _, data, _ = synthetic_archive
    extra = None
    if mutation == 'payload': data['README.md'] += b'changed'
    if mutation == 'completion':
        completion = source.read_json(data['COMPLETED.json'])
        completion['artifact_manifest_sha256'] = '0'*64
        data['COMPLETED.json'] = runner._json_bytes(completion)
    if mutation == 'provenance': data['provenance/provenance.json'] += b' '
    if mutation == 'extra': extra = source.EVALUATION_ID + '/../escape'
    if mutation == 'duplicate': extra = source.EVALUATION_ID + '/README.md'
    if mutation == 'duplicate':
        with pytest.warns(UserWarning): write_zip(path, data, source.EVALUATION_ID, extra)
    else: write_zip(path, data, source.EVALUATION_ID, extra)
    # Rebind only outer checksum to ensure independent inner guards are exercised.
    monkeypatch.setattr(source, 'ARCHIVE_SHA256', sha256_file(path))
    with pytest.raises(ValueError): source.load_archive(path)


@pytest.mark.parametrize('mutation', ['best', 'union', 'reference', 'lambda', 'toggle'])
def test_internal_source_binding(mutation):
    p = provenance_fixture()
    if mutation == 'best': p['sources'][0]['best_checkpoint_sha256'] = '9'*64
    if mutation == 'union': p['input_snapshot'].pop(next(iter(p['input_snapshot'])))
    if mutation == 'reference': p['sources'][1]['identity']['full_model_reference']['best_checkpoint_sha256'] = '9'*64
    if mutation == 'lambda': p['sources'][1]['identity']['lambda_attention'] = .6
    if mutation == 'toggle': p['sources'][1]['identity']['toggles']['use_disagreement_weighting'] = True
    with pytest.raises(ValueError): source.validate_bindings(p)


def test_export_determinism_limits_and_no_overwrite(synthetic_archive, tmp_path):
    archive, proposal, _, _ = synthetic_archive
    before = sha256_file(archive)
    with pytest.raises(ValueError, match='authorization'):
        runner.run(archive, namespace(tmp_path/'unauthorized'), proposal)
    first = runner.run(archive, namespace(tmp_path/'one'), proposal, authorized=True)
    second = runner.run(archive, namespace(tmp_path/'two'), proposal, authorized=True)
    files = {p.relative_to(first).as_posix(): p.read_bytes() for p in first.rglob('*') if p.is_file()}
    assert files == {p.relative_to(second).as_posix(): p.read_bytes() for p in second.rglob('*') if p.is_file()}
    assert sha256_file(archive) == before
    assert (first/'source/archive.zip').read_bytes() == archive.read_bytes()
    marker = source.read_json(files['COMPLETED.json'])
    assert marker['scope'] == runner.SCOPE and marker['model_forwards'] == 0
    for field in ('r3_review_closed', 'r4_closed', 'r5_closed', 'stage_a_certified', 'full_supplement_complete',
                  'original_prediction_bytes_reverified', 'original_supervision_bytes_reverified',
                  'checkpoint_bytes_reverified', 'live_sources_reverified'):
        assert marker[field] is False
    manifest = source.read_json(files['artifact_manifest.json'])
    assert all(source.digest(files[name]) == value for name, value in manifest['files'].items())
    assert source.digest(files['artifact_manifest.json']) == marker['artifact_manifest_sha256']
    assert len(source.read_csv(files['tables/group_summaries.csv'])) == 24
    assert len(source.read_csv(files['tables/descriptive_contrasts.csv'])) == 12
    assert len(source.read_csv(files['tables/membership.csv'])) == 2196
    with pytest.raises(ValueError, match='no overwrite'):
        runner.run(archive, namespace(tmp_path/'one'), proposal, authorized=True)


def test_protected_namespace(synthetic_archive, tmp_path):
    archive, proposal, _, _ = synthetic_archive
    with pytest.raises(ValueError, match='protected output namespace'):
        runner.run(archive, tmp_path/'existing-results', proposal, authorized=True)
    assert not (tmp_path/'existing-results').exists()


def test_no_research_imports_or_source_path_reads(synthetic_archive, monkeypatch, tmp_path):
    import builtins
    original_import, original_open = builtins.__import__, Path.open
    def guarded_import(name, *args, **kwargs):
        assert not any(word in name for word in ('torch', 'phase4_evaluation_metrics', 'model', 'phase4_execution'))
        return original_import(name, *args, **kwargs)
    def guarded_open(path, *args, **kwargs):
        assert 'nonexistent-do-not-open' not in str(path)
        assert path.suffix not in ('.pt', '.npy', '.npz', '.dcm')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(Path, 'open', guarded_open)
    archive, proposal, _, _ = synthetic_archive
    runner.run(archive, namespace(tmp_path/'no-model'), proposal, authorized=True)


def test_source_changed_and_interruption_leave_no_completion(synthetic_archive, monkeypatch, tmp_path):
    archive, proposal, _, _ = synthetic_archive
    original = runner.sha256_file
    def changed(path):
        return '0'*64 if Path(path) == archive else original(path)
    monkeypatch.setattr(runner, 'sha256_file', changed)
    with pytest.raises(ValueError, match='changed during export'):
        runner.run(archive, namespace(tmp_path/'changed'), proposal, authorized=True)
    assert not list((tmp_path/'changed').rglob('COMPLETED.json'))
    monkeypatch.setattr(runner, 'sha256_file', original)
    original_open = Path.open
    def interrupt(path, *args, **kwargs):
        if path.name == 'COMPLETED.json': raise KeyboardInterrupt()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run(archive, namespace(tmp_path/'interrupted'), proposal, authorized=True)
    assert not list((tmp_path/'interrupted').rglob('COMPLETED.json'))
