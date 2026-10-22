"""Reuse accepted import-compatible audit denial; no class replacement or OS-isolation claim."""
import sys

def deny(*args, **kwargs):
    raise PermissionError('network/child/pickle access forbidden')

def audit(event, args):
    if (event.startswith(('socket.', 'subprocess.')) or event in
            ('os.system','os.exec','os.spawn','os.posix_spawn','os.fork','os.forkpty',
             'os.startfile','os.startfile/2','_winapi.CreateProcess')):
        deny()

def install():
    # Classes remain intact for import-time inheritance. Same event policy as
    # accepted qualification_guards; frozen input adapters own file admission.
    sys.addaudithook(audit)
