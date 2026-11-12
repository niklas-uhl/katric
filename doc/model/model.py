import sys,random
from collections import defaultdict
random.seed(7)
def rmat(scale,ef,a=.57,b=.19,c=.19):
    n=1<<scale; E=set()
    for _ in range(ef*n):
        u=v=0
        for _ in range(scale):
            r=random.random(); u<<=1; v<<=1
            if r<a: pass
            elif r<a+b: v|=1
            elif r<a+b+c: u|=1
            else: u|=1; v|=1
        if u!=v: E.add((min(u,v),max(u,v)))
    adj=[[] for _ in range(n)]
    for u,v in E: adj[u].append(v); adj[v].append(u)
    return adj
def metis(path):
    L=[l for l in open(path) if not l.startswith('%')]; n=int(L[0].split()[0])
    return [[int(x)-1 for x in l.split()] for l in L[1:n+1]]
def ranges(weights,p):
    tot=sum(weights) or 1; acc=0; own=[0]*len(weights)
    for v,w in enumerate(weights):
        own[v]=min(p-1,acc*p//tot); acc+=w
    return own
def model(adj,p,name):
    n=len(adj); deg=[len(a) for a in adj]; key=lambda v:(deg[v],v)
    out=[sorted(u for u in adj[v] if key(u)>key(v)) for v in range(n)]
    dp=[len(o) for o in out]
    indeg=[deg[v]-dp[v] for v in range(n)]
    tail=[sum(dp[v]+dp[u] for u in out[v]) for v in range(n)]
    W=sum(tail)
    def stats(own,mode):
        work=[0]*p; send=[0]*p; recv=[0]*p
        if mode=="push":
            for v in range(n):
                seen=set()
                for u in out[v]:
                    work[own[u]]+=dp[v]+dp[u]
                    r=own[u]
                    if r!=own[v] and r not in seen:
                        seen.add(r); send[own[v]]+=dp[v]; recv[r]+=dp[v]
        else:
            need=[set() for _ in range(p)]
            for v in range(n):
                work[own[v]]+=tail[v]
                for u in out[v]:
                    if own[u]!=own[v]: need[own[v]].add(u)
            for r in range(p):
                for u in need[r]: send[own[u]]+=dp[u]; recv[r]+=dp[u]
        ma=lambda x:max(x)/(sum(x)/p) if sum(x) else 0
        return ma(work),ma(send),ma(recv),sum(send)
    edge_bal=ranges(deg,p)
    rows=[("push, edge-bal",stats(edge_bal,"push")),("tail, edge-bal",stats(edge_bal,"tail")),
          ("tail, DPD",stats(ranges(tail,p),"tail")),
          ("tail, DPD+reply",stats(ranges([tail[v]+min(indeg[v],p)*dp[v] for v in range(n)],p),"tail"))]
    maxv=max(tail)
    print(f"{name} p={p}: n={n} m={sum(deg)//2} max deg={max(deg)} max d+={max(dp)} sqrt(2m)={int((sum(deg))**.5)} max vertex tail cost/(W/p)={maxv/(W/p):.3f}")
    for lab,(w,s,r,vol) in rows:
        print(f"   {lab:16} work max/avg={w:6.2f}  send max/avg={s:6.2f}  recv max/avg={r:6.2f}  total vol={vol:9d}")
G={"rmat16":rmat(16,16)}
G["com-amazon"]=metis(sys.argv[1])
for name,adj in G.items():
    for p in (16,64,256):
        model(adj,p,name)
