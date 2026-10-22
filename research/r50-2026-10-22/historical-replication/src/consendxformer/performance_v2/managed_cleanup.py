"""Cleanup only the group created for this child, including surviving children."""
import signal,time,subprocess
def cleanup(child,*,observe,send,clock=time.monotonic,sleep=time.sleep):
    if child is None:return []
    members=observe(child.pid)
    if child.poll() is None or members:
        send(child.pid,signal.SIGTERM)
        deadline=clock()+15
        while clock()<deadline:
            try:child.wait(timeout=.05)
            except subprocess.TimeoutExpired:pass
            members=observe(child.pid)
            if child.poll() is not None and not members:return []
            sleep(.05)
        send(child.pid,signal.SIGKILL)
        deadline=clock()+15
        while clock()<deadline:
            try:child.wait(timeout=.05)
            except subprocess.TimeoutExpired:pass
            members=observe(child.pid)
            if child.poll() is not None and not members:return []
            sleep(.05)
    members=observe(child.pid)
    if child.poll() is None or members:raise RuntimeError('managed group remains after bounded cleanup')
    return members
