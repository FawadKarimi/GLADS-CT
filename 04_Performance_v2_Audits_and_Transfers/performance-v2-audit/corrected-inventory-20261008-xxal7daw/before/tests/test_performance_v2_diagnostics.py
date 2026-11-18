"""Invented records only; no project results, tensors or models are opened."""
import ast
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/53_extract_performance_v2_diagnostics.py"
spec = importlib.util.spec_from_file_location("v2_diagnostics", SCRIPT)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def blob(value):
    return d.canonical(value) + b"\n"


def metrics(baseline, loss, f1):
    result = {"loss": loss, "accuracy": .5}
    if baseline:
        result.update(f1_macro=f1, precision_macro=.5, recall_macro=.5)
        for key in ("precision", "recall", "f1", "support"):
            result[key + "_per_class"] = [1, 1, 1] if key == "support" else [.5, .5, .5]
    else:
        result.update(total_loss=loss, diagnostic_loss=loss, ordinal_loss=0., attention_loss=0.,
                      macro_f1=f1, macro_precision=.5, macro_recall=.5, class_order=list(d.CLASSES),
                      per_class={c: {"precision": .5, "recall": .5, "f1": .5, "support": 1} for c in d.CLASSES})
    return result


def csv_data(large=False, with_q=True):
    rows = []
    for split, n, patients in (("train", 807, 437), ("val", 143, 84)) if large else (("train", 3, 3), ("val", 3, 3)):
        for i in range(n):
            row = {"patient_id": f"invented-{split}-p{i % patients}", "nodule_id": f"invented-{split}-n{i}",
                   "series_instance_uid": f"invented-{split}-series{i % patients}", "split": split,
                   "label": d.CLASSES[i % 3], "n_valid_malignancy_ratings": "3"}
            if with_q:
                row.update(q_benign=1/3, q_indeterminate=1/3, q_malignant=1/3, disagreement_entropy=1.)
            rows.append(row)
    # If the extractor interprets this test row it will fail or leak the sentinel.
    rows.append({k: "DO_NOT_EXPORT_TEST" for k in rows[0]} | {"split": "test"})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def seal(directory, kind):
    paths = [p for p in directory.rglob('*') if p.is_file() and p.name not in ('COMPLETED.json', 'artifact_manifest.json')]
    # Synthetic poison payloads deliberately are in the manifest but must not be read.
    files = {p.relative_to(directory).as_posix(): d.digest(p.read_bytes()) for p in paths}
    manifest = {"experiment_id": directory.name, "files": files, "bundle_sha256": d.digest(d.canonical(files))}
    marker = {"experiment_id": directory.name}
    if kind == "off":
        manifest.update(schema="phase4_ablation_artifacts_v1", run_kind="phase4_component_ablation")
        marker.update(schema="phase4_ablation_completion_v1", run_kind="phase4_component_ablation", runtime_training_explicitly_authorized=True)
    else:
        manifest.update(schema_version=1, research_results_eligible=True)
        marker.update(schema_version=1, research_results_eligible=True, artifact_manifest="artifact_manifest.json")
        if kind == 'baseline':
            marker['synthetic_dry_run'] = False
        else:
            manifest['run_mode'] = marker['run_mode'] = 'final'
            manifest['test_results_eligible'] = marker['test_results_eligible'] = True
    (directory/'artifact_manifest.json').write_bytes(blob(manifest))
    marker['artifact_manifest_sha256'] = d.digest((directory/'artifact_manifest.json').read_bytes())
    (directory/'COMPLETED.json').write_bytes(blob(marker))


def make_run(repo, variant="full", seed=42, large=False):
    baseline = variant == 'B1'
    kind = 'baseline' if baseline else ('full' if variant == 'full' else 'off')
    data = csv_data(large, not baseline)
    lock = {"invented_lock": True}
    toggles = {k: True for k in ('use_attention_alignment', 'use_disagreement_weighting', 'use_ordinal_head', 'use_hierarchical_fusion')}
    if kind == 'off':
        toggles[{'attention_alignment':'use_attention_alignment','disagreement_weighting':'use_disagreement_weighting',
                 'ordinal_auxiliary':'use_ordinal_head','hierarchical_fusion':'use_hierarchical_fusion'}[variant]] = False
    config = {'data':{'mode':'real'}, 'experiment':{'seed':seed}, 'ablation':toggles,
              'dataset':{'class_names':list(d.CLASSES)}, 'predictions':{'class_order':list(d.CLASSES)}}
    base_config = json.loads(json.dumps(config))
    if kind == 'full': config['execution']={'tuning_protocol_sha256':'c'*64}
    ident = {'seed':seed, 'project_commit':'a'*40}
    if baseline:
        ident.update(baseline_variant='B1', metadata_sha256=d.digest(data), frozen_split_bundle_sha256=d.FROZEN_SPLIT,
                     baseline_source_commit=d.FROZEN_BASELINE)
    else:
        ident.update(schema_version=1, lambda_ordinal=.5, lambda_attention=.05,
                     input_lock_sha256=d.digest(d.canonical(lock)))
        if kind == 'full':
            ident['run_mode']='final'
            ident.update(manifest_bundle_sha256=d.FROZEN_SPLIT,cohort_view='paper_aligned_multi_reader_2_to_4',expected_resolution=64)
            base_hash=d.digest(d.canonical({'schema_version':1,'phase4':base_config,'b1_anchor':base_config}))
            ident['configuration_sha256']=d.digest(d.canonical({'schema_version':1,'base_combined_config_sha256':base_hash,
                'effective_config':config,'input_lock_sha256':ident['input_lock_sha256'],'tuning_protocol_sha256':'c'*64}))
        else:
            ident.pop('project_commit')
            ident.update(variant=variant, ablation_code_commit='b'*40, toggles=toggles,
                         run_kind='phase4_component_ablation', configuration_sha256=d.digest(d.canonical(config)),
                         inputs={'manifest_bundle_sha256':d.FROZEN_SPLIT,'cohort_view':'paper_aligned_multi_reader_2_to_4','expected_resolution':64})
    payload=dict(ident)
    if baseline: payload['schema_version']=1
    h=d.digest(d.canonical(payload)); ident.update(digest=h, experiment_id=f'invented-{variant}-{seed}-{h[:16]}')
    directory=repo/d.ROOTS[0 if baseline else (1 if kind=='full' else 2)]/ident['experiment_id']
    directory.mkdir(parents=True)
    run={'identity':ident, 'execution_state':'completed', 'research_results_eligible':True}
    if baseline:
        run.update(mode='real', synthetic_non_research=False, prediction_policy={'class_order':list(d.CLASSES)}, selection={'best_epoch':2})
    elif kind=='full': run['best_epoch']=2
    else: run.update(completed=True, fresh_initialization=True)
    history=[{'epoch':i,'train':metrics(baseline,l+.1,f), 'validation':metrics(baseline,l,f)}
             for i,l,f in [(1,2.,.4),(2,1.,.5),(3,1.,.7),(4,1.5,.7)]]
    values={'provenance/run.json':blob(run),'history/epoch_history.json':blob(history),
            'metrics/val.json':blob({'prediction_metrics':{'classification':{'macro_f1':.5}},'within_variant_evaluation_losses':history[1]['validation']} if kind=='off' else
                                    ({'evaluation_losses':history[1]['validation']} if kind=='full' else history[1]['validation'])),
            'checkpoints/best.pt':b'NEVER OPEN', 'predictions/test.csv':b'NEVER OPEN', 'metrics/test.json':b'NEVER OPEN'}
    if baseline:
        values.update({'config/experiment_identity.json':blob(ident),'config/input_metadata.csv':data,
                       'config/phase1_effective.yaml':b'dataset: {}\n','config/phase3_effective.yaml':b'experiment: {}\n'})
    else:
        values.update({'config/run_identity.json':blob(ident),'config/effective_config.json':blob(config),
                       'config/input_lock.json':blob(lock),'config/input_supervision.csv':data})
        if kind=='full': values.update({'config/phase4_config.json':blob(base_config),'config/b1_anchor_config.json':blob(base_config)})
    for name,value in values.items():
        path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    seal(directory,kind)
    return directory,kind


@pytest.fixture
def repo(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    subprocess.run(['git','init',str(root)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Synthetic Fixture','-c','user.email=fixture@example.invalid',
                    'commit','--allow-empty','-m','invented fixture'],check=True,capture_output=True)
    return root


@pytest.mark.parametrize('variant',['B1',*d.VARIANTS[1:]])
def test_history_adapters_and_counterfactual_ties(repo,variant):
    directory,_=make_run(repo,variant)
    source=d.load_run(d.Reader(),directory)
    rows,selection=d.histories(source)
    assert selection['minimum_loss_epoch']==2
    assert selection['maximum_macro_f1_epoch']==3
    assert selection['completed_epochs']==4
    assert selection['receipt_selected_epoch']==(None if variant in d.VARIANTS[2:] else 2)
    assert selection['per_epoch_lr']=='NOT_RECORDED'
    assert any(r['original_key'].startswith('validation.') for r in rows)
    assert 'test' not in {r['split'] for r in rows}


@pytest.mark.parametrize('relative',['predictions/test.csv','metrics/test.json','predictions/val.csv','checkpoints/best.pt','checkpoints/best.metadata.json','images/roi.png','arrays.npz','../COMPLETED.json'])
def test_forbidden_reads_before_open(tmp_path,relative,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('open should not be reached')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    with pytest.raises(d.AuditError,match='Forbidden'):
        d.Reader().read(tmp_path,relative)


def test_subset_hash_and_no_test_reads(repo,monkeypatch):
    directory,_=make_run(repo)
    original=Path.read_bytes;opened=[]
    def guarded(p):
        opened.append(p)
        assert '/checkpoints/' not in p.as_posix() and p.name!='test.csv' and p.name!='test.json'
        return original(p)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    reader=d.Reader();source=d.load_run(reader,directory);reader.recheck()
    assert all(not r['full_bundle_verified'] for r in reader.records.values())
    assert reader.records[str(directory/'artifact_manifest.json')]['recorded_manifest_sha256'] is None
    rows=d.metadata_rows(source['metadata'])
    assert len(rows)==6 and 'DO_NOT_EXPORT_TEST' not in json.dumps(rows)
    assert len(d.conflict_rows(rows))==3


@pytest.mark.parametrize('mutation',['payload','manifest_digest','link','order','identity','missing','nested'])
def test_corruption_and_schema_failure(repo,mutation):
    directory,kind=make_run(repo,'attention_alignment')
    if mutation=='payload':
        (directory/'history/epoch_history.json').write_text('[]')
    elif mutation=='manifest_digest':
        p=directory/'artifact_manifest.json';r=json.loads(p.read_text());r['bundle_sha256']='0'*64;p.write_bytes(blob(r))
        p=directory/'COMPLETED.json';r=json.loads(p.read_text());r['artifact_manifest_sha256']=d.digest((directory/'artifact_manifest.json').read_bytes());p.write_bytes(blob(r))
    elif mutation=='link':
        p=directory/'COMPLETED.json';r=json.loads(p.read_text());r['artifact_manifest_sha256']='0'*64;p.write_bytes(blob(r))
    else:
        p=directory/({'order':'config/effective_config.json','identity':'provenance/run.json','missing':'history/epoch_history.json','nested':'metrics/val.json'}[mutation])
        r=json.loads(p.read_text())
        if mutation=='order': r['predictions']['class_order'].reverse()
        if mutation=='identity': r['identity']['seed']=43
        if mutation=='missing': r[0]['validation'].pop('macro_f1')
        if mutation=='nested': r={'loss':1.}
        p.write_bytes(blob(r));seal(directory,kind)
    with pytest.raises(d.AuditError):
        d.histories(d.load_run(d.Reader(),directory))


def test_duplicate_and_patient_overlap():
    rows=d.metadata_rows(csv_data())
    for field,value in [('nodule_id',rows[0]['nodule_id']),('patient_id',rows[0]['patient_id'])]:
        altered=[dict(r) for r in rows];altered[-1][field]=value
        stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=altered[0]);w.writeheader();w.writerows(altered)
        with pytest.raises(d.AuditError): d.metadata_rows(stream.getvalue().encode())


@pytest.mark.parametrize('q,u', [([-0.01,.5,.51],.5),([1.01,0,0],.5),([.2,.2,.2],.5),([0,0,0],0),([float('nan'),0,1],0),([1,0,0],float('inf')),([1,0,0],-1)])
def test_invalid_targets(q,u):
    row=d.metadata_rows(csv_data())[0]
    row.update({f'q_{c}':str(v) for c,v in zip(d.CLASSES,q)});row['disagreement_entropy']=str(u)
    with pytest.raises(d.AuditError): d.conflict_rows([row])


def test_ties_bins_and_proxy_no_argmax_relabel():
    row=d.metadata_rows(csv_data())[0]
    rows=[]
    for i,u in enumerate([0,.25,.5,.75,1]):
        rows.append(dict(row,source_csv_line=i,disagreement_entropy=str(u)))
    out=d.conflict_rows(rows)
    assert [r['u_bin'] for r in out]==['[0,.25)','[.25,.5)','[.5,.75)','[.75,1]','[.75,1]']
    assert all(r['category']=='maximal_tie' for r in out)
    assert out[-1]['discordance_proxy']==pytest.approx(2/3)
    assert d.summarize_conflicts(out,'invented')[0]['maximal_tie']==5
    row.update(q_benign='.2',q_indeterminate='.4',q_malignant='.4')
    assert d.conflict_rows([row])[0]['category']=='hard_not_maximal'


def test_changed_input(repo):
    directory,_=make_run(repo)
    reader=d.Reader();d.load_run(reader,directory)
    (directory/'history/epoch_history.json').write_bytes(b'changed')
    with pytest.raises(d.AuditError,match='hash mismatch'): reader.recheck()


def test_path_overlap_and_symlink_guards(repo,tmp_path,monkeypatch):
    with pytest.raises(d.AuditError,match='overlap'): d.extract(repo,repo)
    # Explicit Windows reparse-point stat stub; does not pretend a real link was created.
    original=Path.lstat
    class Reparse:
        st_mode=0o100644
        st_file_attributes=0x400
    monkeypatch.setattr(Path,'lstat',lambda p: Reparse() if p==repo else original(p))
    with pytest.raises(d.AuditError,match='reparse'): d.safe_path(repo)


def test_full_inventory_export_source_immutability_and_no_overwrite(repo,tmp_path,monkeypatch):
    for variant in d.VARIANTS:
        for seed in d.SEEDS: make_run(repo,variant,seed,large=True)
    parent=tmp_path/'reports';parent.mkdir()
    before={p:p.read_bytes() for p in repo.rglob('*') if p.is_file() and '.git' not in p.parts}
    # Script never opens these; the test's fixture snapshot is not an extractor read.
    output,status=d.extract(repo,parent,argv=['synthetic'])
    assert status=='partial'  # B1 YAML-only hyperparameters intentionally pending.
    audit=json.loads((output/'audit.json').read_text())
    assert audit['model_forwards']==0 and not audit['full_source_bundle_verified']
    assert audit['git_before']==audit['git_after']
    assert 'DO_NOT_EXPORT_TEST' not in (output/'train_val_source_rows.csv').read_text()
    with (output/'run_inventory.csv').open() as s: assert len(list(csv.DictReader(s)))==18
    with (output/'checkpoint_selection_diagnostics.csv').open() as s: assert len(list(csv.DictReader(s)))==18
    mf=json.loads((output/'artifact_manifest.json').read_text())
    assert d.digest(d.canonical(mf['files']))==mf['bundle_sha256']
    for name,h in mf['files'].items(): assert d.digest((output/name).read_bytes())==h
    assert all(p.read_bytes()==b for p,b in before.items())
    with pytest.raises(FileExistsError): d.write_json(output/'audit.json',{})
    second,_=d.extract(repo,parent)
    for name in ['epoch_history_long.csv','checkpoint_selection_diagnostics.csv','train_val_source_rows.csv','train_target_conflict_summary.csv']:
        assert (output/name).read_bytes()==(second/name).read_bytes()


def test_mutation_before_export_never_completes(repo,tmp_path,monkeypatch):
    directory,_=make_run(repo)
    parent=tmp_path/'out';parent.mkdir()
    original=d.Reader.recheck
    def change(reader):
        (directory/'provenance/run.json').write_bytes(b'changed')
        original(reader)
    monkeypatch.setattr(d.Reader,'recheck',change)
    with pytest.raises(d.AuditError): d.extract(repo,parent)
    assert not list(parent.rglob('AUDIT_COMPLETED.json'))


def test_only_stdlib_and_nonlaunching_configuration():
    imports=[]
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node,ast.Import): imports.extend(n.name.split('.')[0] for n in node.names)
        if isinstance(node,ast.ImportFrom): imports.append(node.module.split('.')[0])
    assert not {'torch','numpy','pandas','yaml','consendxformer','requests'} & set(imports)
    assert set(imports)<=set(sys.stdlib_module_names)
    config=json.loads((SCRIPT.parents[1]/'configs/performance_v2/pilot_protocol_v1.json').read_text())
    assert config['pilot_execution_authorized'] is False
    assert config['budget']['total_cap']==39
    assert len(config['phase2']['recipes'])==6


def test_missing_supervision_and_inconsistent_loss_alias(repo):
    rows=d.metadata_rows(csv_data(with_q=False))
    with pytest.raises(d.AuditError,match='Missing training supervision'): d.conflict_rows(rows)
    directory,_=make_run(repo)
    source=d.load_run(d.Reader(),directory)
    source['history'][0]['validation']['loss']=99.
    with pytest.raises(d.AuditError,match='aliases'): d.histories(source)


def test_full_config_tamper_and_unsafe_manifest(repo):
    directory,kind=make_run(repo)
    p=directory/'config/effective_config.json';config=json.loads(p.read_text());config['training']={'lr':999}
    p.write_bytes(blob(config));seal(directory,kind)
    with pytest.raises(d.AuditError,match='configuration binding'): d.load_run(d.Reader(),directory)
    p=directory/'artifact_manifest.json';manifest=json.loads(p.read_text());manifest['files']['../escape']='a'*64
    manifest['bundle_sha256']=d.digest(d.canonical(manifest['files']));p.write_bytes(blob(manifest))
    p=directory/'COMPLETED.json';marker=json.loads(p.read_text());marker['artifact_manifest_sha256']=d.digest((directory/'artifact_manifest.json').read_bytes());p.write_bytes(blob(marker))
    with pytest.raises(d.AuditError,match='Unsafe manifest'): d.load_run(d.Reader(),directory)


def test_tuning_incomplete_and_failed_not_used(repo,tmp_path):
    directory,_=make_run(repo)
    p=directory/'COMPLETED.json';marker=json.loads(p.read_text());marker['run_mode']='tuning';p.write_bytes(blob(marker))
    root=directory.parent
    (root/'incomplete').mkdir()
    (root/'failed').mkdir();(root/'failed/FAILED.json').write_text('{}')
    parent=tmp_path/'reports';parent.mkdir()
    output,status=d.extract(repo,parent)
    with (output/'run_inventory.csv').open() as s: states={r['status'] for r in csv.DictReader(s)}
    assert states=={'tuning','incomplete','failed'}
    assert status=='partial'
    assert json.loads((output/'train_val_metadata_summary.json').read_text())=={}


def test_multiple_candidates_report_without_score_selection(repo,tmp_path):
    directory,kind=make_run(repo)
    other=repo/d.ROOTS[2]/directory.name
    import shutil
    shutil.copytree(directory,other)
    parent=tmp_path/'reports';parent.mkdir()
    output,status=d.extract(repo,parent)
    issues=json.loads((output/'missing_or_unverified.json').read_text())['issues']
    assert any(i.get('slot')==['full',42] and len(i.get('candidates',[]))==2 for i in issues)
    assert status=='partial'


def test_exclusive_destination_collision(repo,tmp_path,monkeypatch):
    make_run(repo)
    parent=tmp_path/'reports';parent.mkdir()
    class FrozenClock:
        @staticmethod
        def now(tz):
            class Stamp:
                def strftime(self,fmt): return 'SYNTHETIC_FIXED_CLOCK'
            return Stamp()
    monkeypatch.setattr(d,'datetime',FrozenClock)
    output,_=d.extract(repo,parent)
    before={p.name:p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError): d.extract(repo,parent)
    assert {p.name:p.read_bytes() for p in output.iterdir()}==before
    (output/'AUDIT_COMPLETED.json').unlink()  # Synthetic failed/partial destination must also be untouched.
    before={p.name:p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError): d.extract(repo,parent)
    assert {p.name:p.read_bytes() for p in output.iterdir()}==before
