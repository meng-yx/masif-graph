"""Holo-only HH sanity: frozen vs learned retrieval on the clean set (DB = all chains), no AF3 needed.
Reproduces the MaSIF-search holo->holo cell at real scale before AF3 arrives."""
import os, sys, json
import numpy as np, torch
torch.set_num_threads(8)
sys.path.insert(0, "src")
from masif_graph.p4.eval_af3 import build_encoder, normalize
from masif_graph.p4.eval_af3 import load_state_chain

DATA="/work/upthomae/Meng/phase5/npz"; CKPT="/work/upthomae/Meng/phase4/ret_full_ctr_best.pt"
ids=[l.strip() for l in open("logs/phase5/_holo_ok.txt") if l.strip()]
POS="pos"; MINP=8
class R:  # minimal holo rec
    pass
recs=[]
for cid in ids:
    h1=load_state_chain(DATA,cid,"holo","p1","cpu"); h2=load_state_chain(DATA,cid,"holo","p2","cpu")
    cf=os.path.join(DATA,f"{cid}__contacts.npz")
    if h1 is None or h2 is None or not os.path.exists(cf): continue
    pos=np.load(cf)[POS]
    if len(np.unique(pos[:,0]))<MINP or len(np.unique(pos[:,1]))<MINP: continue
    r=R(); r.cid=cid; r.hg1,_=h1; r.hg2,_=h2; r.pos=pos; recs.append(r)
print(f"usable holo complexes: {len(recs)} -> DB {2*len(recs)} chains", flush=True)
enc,comp,src=build_encoder(recs,CKPT,"cpu"); enc.eval(); T=comp.T
raw={}; allrows=[]
with torch.no_grad():
    for r in recs:
        e={"h1":enc(r.hg1),"h2":enc(r.hg2)}; raw[r.cid]=e; allrows.append(torch.cat(list(e.values()),0))
    mu=torch.cat(allrows,0).mean(0,keepdim=True)
    emb={c:{k:normalize(v-mu) for k,v in e.items()} for c,e in raw.items()}
def U(a): return np.unique(a) if len(a) else np.zeros(0,np.int64)
pat={}
for r in recs:
    I1=U(r.pos[:,0]); I2=U(r.pos[:,1])
    pat[r.cid]={"p1":{"z":emb[r.cid]["h1"][I1],"ds":r.hg1["desc_straight"][I1],"df":r.hg1["desc_flipped"][I1]},
                "p2":{"z":emb[r.cid]["h2"][I2],"ds":r.hg2["desc_straight"][I2],"df":r.hg2["desc_flipped"][I2]}}
dbk=[(c,role) for c in pat for role in ("p1","p2")]
def ls(zq,zd): return -1e9 if zq.shape[0]==0 or zd.shape[0]==0 else float((zq@T@zd.t()).max(1).values.median())
def fs(qs,df):
    if qs.shape[0]==0 or df.shape[0]==0: return 1e9
    d=torch.sqrt(((qs[:,None,:]-df[None,:,:])**2).sum(-1)+1e-12); return float(d.min(1).values.median())
def retr(method):
    ranks=[]
    for cid in pat:
        for qr,pr in (("p1","p2"),("p2","p1")):
            qp=pat[cid][qr]; scored=[]
            for (dc,dr) in dbk:
                if dc==cid and dr==qr: continue
                dp=pat[dc][dr]
                scored.append(((dc,dr), -ls(qp["z"],dp["z"]) if method=="learned" else fs(qp["ds"],dp["df"])))
            scored.sort(key=lambda x:x[1]); order=[k for k,_ in scored]
            if (cid,pr) in order: ranks.append(order.index((cid,pr))+1)
    r=np.array(ranks,float)
    return {"n":len(r),"top1":float((r<=1).mean()),"top5":float((r<=5).mean()),"top10":float((r<=10).mean()),
            "mrr":float((1/r).mean()),"medrank":float(np.median(r))}
with torch.no_grad():
    out={"src":os.path.basename(src),"db":len(dbk),"HH_frozen":retr("frozen"),"HH_learned":retr("learned")}
print(json.dumps(out,indent=2))
json.dump(out,open("logs/phase5/hh_holo_clean.json","w"),indent=2)
