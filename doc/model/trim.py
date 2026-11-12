import sys,random
path=sys.argv[1]
with open(path) as fh:
    lines=[l for l in fh if not l.startswith('%')]
n,m=map(int,lines[0].split()[:2])
adj=[[int(x)-1 for x in l.split()] for l in lines[1:n+1]]
deg=[len(a) for a in adj]
def run(perm,p,label):
    # vertex v gets id perm[v]; contiguous id ranges, balanced by edges like KaGen balance-edges (approx: by vertex count)
    owner=[0]*n
    order=sorted(range(n),key=lambda v:perm[v])
    tot=sum(deg); acc=0
    for v in order:
        owner[v]=min(p-1,acc*p//tot); acc+=deg[v]
    key=lambda v:(deg[v],perm[v])
    full=trim=0; cnt=0
    for v in range(n):
        out=[u for u in adj[v] if key(u)>key(v)]
        if not out: continue
        out.sort(key=key)                       # orientation order
        first={}                                # first position of each remote rank in orientation order
        for i,u in enumerate(out):
            r=owner[u]
            if r!=owner[v] and r not in first: first[r]=i
        for r,i in first.items():
            full+=len(out); trim+=len(out)-i-1; cnt+=1
    print(f"{label:10} p={p:5} messages={cnt:8d} volume full={full:9d} trimmed={trim:9d} ratio={trim/max(full,1):.2f}")
ident=list(range(n)); rnd=list(range(n)); random.seed(1); random.shuffle(rnd)
for p in (16,64,256,1024):
    run(ident,p,"original"); run(rnd,p,"permuted")
