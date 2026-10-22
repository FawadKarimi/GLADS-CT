"""Read-only byte/aggregate-arithmetic checks. No model imports or execution."""
from pathlib import Path
import ast, csv, hashlib, json, math, statistics, sys

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'research/r50-2026-10-22'
def check(condition, message):
    if not condition:
        raise ValueError(message)
def close(a, b, label):
    check(math.isfinite(float(a)) and abs(float(a)-float(b)) < 1e-10, label)
def metrics(cm):
    check(len(cm)==3 and all(len(r)==3 for r in cm), 'matrix shape')
    check(all(type(x) is int and x>=0 for row in cm for x in row), 'matrix counts')
    support = [sum(r) for r in cm]
    columns = [sum(row[c] for row in cm) for c in range(3)]
    f1 = [2*cm[c][c]/(support[c]+columns[c]) if support[c]+columns[c] else 0.0 for c in range(3)]
    return sum(cm[c][c] for c in range(3))/sum(support), sum(f1)/3, f1
def main():
    ledger = json.loads((RELEASE/'SHA256-LEDGER.json').read_text(encoding='utf-8'))
    for rel, expected in ledger['files'].items():
        p = ROOT/rel
        check(p.is_file() and not p.is_symlink(), 'missing/symlink: '+rel)
        check(hashlib.sha256(p.read_bytes()).hexdigest()==expected, 'hash: '+rel)
    lock = json.loads((RELEASE/'provenance/replication-source-lock.original.json').read_text(encoding='utf-8'))
    count=0
    for rel, expected in lock.items():
        p = RELEASE/'historical-replication'/rel
        if rel.startswith(('src/', 'tests/')):
            check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected, 'historical source: '+rel)
            ast.parse(p.read_text(encoding='utf-8'), filename=rel)
            count += 1
    original = json.loads((RELEASE/'results/original807-replication.json').read_text(encoding='utf-8'))
    check(original['ddof']==1 and original['synthetic_results_used'] is False, 'analysis scope')
    check(len(original['every_seed'])==6, 'six original results')
    for row in original['every_seed']:
        acc, macro, f1 = metrics(row['confusion_matrix'])
        close(acc,row['accuracy'],'original accuracy');close(macro,row['macro_f1'],'original macro-F1')
        for c,name in enumerate(('benign','indeterminate','malignant')):
            close(f1[c],row[name+'_f1'],'original '+name+' F1')
    ensembles=json.loads((RELEASE/'results/exploratory-validation-ensembles.json').read_text(encoding='utf-8'))
    for label,row in ensembles['configurations'].items():
        acc,macro,f1=metrics(row['confusion_matrix'])
        close(acc,row['accuracy'],'ensemble '+label+' accuracy');close(macro,row['macro_f1'],'ensemble '+label+' Macro-F1')
        close(sum(row['weights'].values()),1.0,'weight sum')
    with (RELEASE/'results/expanded1297/TABLE-1-PER-SEED-VALIDATION.csv').open(encoding='utf-8',newline='') as f:
        expanded=list(csv.DictReader(f))
    check({(r['objective'],int(r['seed'])) for r in expanded}=={(o,s) for o in ('HARD','LEGACY') for s in (42,43,44)},'expanded identities')
    names=('benign','indeterminate','malignant')
    matrices={r['run']:[[0]*3 for _ in names] for r in expanded}
    with (RELEASE/'results/expanded1297/TABLE-S2-CONFUSION-MATRICES.csv').open(encoding='utf-8',newline='') as f:
        entries=list(csv.DictReader(f))
    check(len(entries)==54,'six complete expanded matrices')
    cells=set()
    for row in entries:
        key=(row['run'],row['truth'],row['prediction'])
        check(key not in cells,'duplicate matrix cell');cells.add(key)
        matrices[row['run']][names.index(row['truth'])][names.index(row['prediction'])]=int(row['count'])
    for row in expanded:
        cm=matrices[row['run']]
        check([sum(r) for r in cm]==[65,38,40],'validation class support')
        acc,macro,_=metrics(cm)
        close(acc,row['accuracy'],'expanded accuracy');close(macro,row['macro_f1'],'expanded Macro-F1')
        old=next(r for r in original['every_seed'] if r['objective']==row['objective'] and r['seed']==int(row['seed']))
        check(macro<old['macro_f1'],'reported negative expanded comparison')
    with (RELEASE/'results/expanded1297/TABLE-S1-PER-CLASS-VALIDATION.csv').open(encoding='utf-8',newline='') as f:
        class_rows=list(csv.DictReader(f))
    check(len(class_rows)==18,'expanded class inventory')
    for row in class_rows:
        c=names.index(row['class']);cm=matrices[row['run']]
        support=sum(cm[c]);column=sum(r[c] for r in cm)
        close(cm[c][c]/column,row['precision'],'expanded precision')
        close(cm[c][c]/support,row['recall'],'expanded recall')
        close(2*cm[c][c]/(support+column),row['f1'],'expanded class F1')
        check(int(row['support'])==support,'class support')
    summary={}
    for objective in ('HARD','LEGACY'):
        values=[float(r['macro_f1']) for r in expanded if r['objective']==objective]
        summary[objective]={'validation_macro_f1_mean':statistics.mean(values),'sample_sd':statistics.stdev(values)}
    print(json.dumps({'status':'PASS','released_file_hashes':len(ledger['files']),
                      'historical_source_and_test_bindings':count,'original_matrices_verified':6,
                      'exploratory_ensemble_matrices_verified':3,'expanded_matrices_verified':6,
                      'expanded_class_records_verified':18,'expanded_summary':summary,
                      'scope':'saved bytes and arithmetic only; no models, images, logits replay or CE verification'},indent=2))
if __name__=='__main__':
    try: main()
    except Exception as error:
        print(json.dumps({'status':'FAIL','error':str(error)}),file=sys.stderr);sys.exit(1)
