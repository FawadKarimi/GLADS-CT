import hashlib,json,pathlib,shutil,subprocess
R=pathlib.Path(__file__).resolve().parent;S=R.parent/'ConsenDxFormer-performance-v2'
def git(*args):return subprocess.check_output(['git',*args],cwd=S)
sha=lambda x:hashlib.sha256(x).hexdigest()
assert not (R/'before.json').exists()
names=sorted(filter(None,git('ls-files','--cached','--others','--exclude-standard','-z').decode().split('\0')))
untracked=list(filter(None,git('ls-files','--others','--exclude-standard','-z').decode().split('\0')))
data={'root':str(S),'branch':git('branch','--show-current').decode().strip(),'head':git('rev-parse','HEAD').decode().strip(),
 'status':git('status','--porcelain=v1','--untracked-files=all').decode(),'hashes':{n:sha((S/n).read_bytes()) for n in names},
 'untracked':untracked,'staged_diff':sha(git('diff','--cached','--binary')),'tracked_diff':sha(git('diff','--binary'))}
for n in untracked:
 p=R/'before'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(S/n,p)
(R/'before.json').write_text(json.dumps(data,indent=2)+'\n',newline='\n')
print(len(names),len(untracked),data['branch'],data['head'])
