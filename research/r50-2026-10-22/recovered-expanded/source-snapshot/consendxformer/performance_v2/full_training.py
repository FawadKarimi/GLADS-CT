"""Single-use coordinator: durable reservation precedes device setup and reads."""
from .durable_lifecycle import DurableRun
from .reuse import digest

def interrupted_receipt(budget,parent,*,read):
    snapshot=budget.inspect()
    rows=snapshot['attempts']
    if (snapshot['issues'] or type(parent) is not int or not rows or parent!=rows[-1]['number']
        or rows[-1]['state']!='interrupted' or rows[-1]['exported']!=1):
        raise ValueError('latest reconciled interrupted receipt required; completion is not resume admission')
    raw=read(budget.receipt_path(parent))
    if digest(raw)!=rows[-1]['receipt_sha']:raise ValueError('interrupted receipt hash disagrees with durable budget')
    return raw

def execute_reserved(*, spec, adapter, budget, decision, setup, runner_factory,
                     resume=None, parent_attempt=None):
    session=DurableRun(spec=spec,inputs=adapter.inputs,budget=budget,
                       decision_sha256=decision,resume=resume,parent_attempt=parent_attempt)
    with session.guard():
        operations=setup()
        runner=runner_factory(operations)
        return runner.run(resume=resume,parent_attempt=parent_attempt,reserved_session=session)
