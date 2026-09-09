"""Train, test, or predict strict future windows; test never selects a model.

RawData -> prepare_hapt_protocol.py -> train -> test (once) -> predict.
Recognition code and recognition checkpoints are never modified.
"""
import argparse,copy,hashlib,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.data.temporal_protocol import load_protocol,TemporalAnticipationDataset
from src.models.causal_forecaster import CausalForecaster,window_features

LABELS=list(range(1,13))

def metrics(y,p):
    if len(y)==0:return dict(n=0,accuracy=None,macro_f1=None)
    return dict(n=len(y),accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0)))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def make_split(data,manifest,split,temporal):
    mask=np.isin(data['subjects'],manifest['splits'][split]);y=data['y'][mask]
    ds=TemporalAnticipationDataset(data['X'][mask],y,data['subjects'][mask],data['sessions'][mask],data['starts'][mask],**temporal)
    if not len(ds):raise ValueError(f'No valid temporal examples in {split}')
    return ds,window_features(ds.X),dict(subjects=data['subjects'][mask][ds.target_indices],
        sessions=data['sessions'][mask][ds.target_indices],starts=data['starts'][mask][ds.target_indices],
        current=y[ds.last_context_indices])

def neural_prob(model,x,device,batch=128):
    model.eval();parts=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):parts.append(torch.softmax(model(torch.from_numpy(np.ascontiguousarray(x[start:start+batch])).to(device)),dim=-1).cpu().numpy())
    return np.concatenate(parts)

def rf_prob(model,x):
    p=np.zeros((len(x),12),dtype=np.float32)
    p[:,np.asarray(model.classes_,dtype=int)-1]=model.predict_proba(x)
    return p

def train(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a new output directory')
    data,manifest=load_protocol(args.protocol)
    temporal=dict(obs_ratio=args.ratio,seq_len=args.seq_len,horizon_samples=args.horizon_samples,stride=manifest['config']['stride'])
    tr,xtr,_=make_split(data,manifest,'train',temporal);va,xva,_=make_split(data,manifest,'validation',temporal)
    mean=xtr.mean(axis=(0,1),dtype=np.float64).astype(np.float32);std=np.maximum(xtr.std(axis=(0,1)),1e-6).astype(np.float32)
    ztr=((xtr-mean)/std).astype(np.float32);zva=((xva-mean)/std).astype(np.float32)
    out.mkdir(parents=True);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    model=CausalForecaster().to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    counts=np.bincount(tr.y,minlength=13)[1:];weights=np.zeros(12,dtype=np.float32);present=counts>0
    weights[present]=np.sqrt(counts[present].sum()/(present.sum()*counts[present]));weights=np.clip(weights,0,4)
    wt=torch.from_numpy(weights).to(args.device)
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(ztr),torch.from_numpy(tr.y-1)),batch_size=args.batch_size,shuffle=True,generator=generator)
    best,best_f1,best_epoch,stale=None,-1,0,0;history=[];started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0
        for x,y in loader:
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x.to(args.device)),y.to(args.device),weight=wt)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss_sum+=float(loss.detach())*len(y)
        p=neural_prob(model,zva,args.device);m=metrics(va.y,p.argmax(1)+1);m.update(epoch=epoch,loss=loss_sum/len(tr));history.append(m)
        print('epoch',epoch,m,flush=True)
        if m['macro_f1']>best_f1:
            best=copy.deepcopy(model.cpu().state_dict());model.to(args.device);best_f1=m['macro_f1'];best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=args.patience:break
    model.load_state_dict(best);pnn=neural_prob(model,zva,args.device)
    # Fixed reference families; choose on validation only. No test is loaded here.
    models={};probs={'gru':pnn}
    for name,tx,vx in [('last',xtr[:,-1],xva[:,-1]),('context',xtr.reshape(len(xtr),-1),xva.reshape(len(xva),-1))]:
        rf=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced_subsample',random_state=args.seed,n_jobs=2)
        rf.fit(tx,tr.y);models[name]=rf;probs[name]=rf_prob(rf,vx)
    probs['ensemble']=(probs['gru']+probs['context'])/2
    scores={k:metrics(va.y,p.argmax(1)+1) for k,p in probs.items()}
    selected=max(scores,key=lambda k:scores[k]['macro_f1'])
    # Current-activity classifier for a deployable persistence baseline, trained without future test labels.
    mask=np.isin(data['subjects'],manifest['splits']['train']);p_len=tr.obs_len
    current=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced_subsample',random_state=args.seed,n_jobs=2)
    current.fit(window_features(data['X'][mask,:p_len]),data['y'][mask]);models['persistence']=current
    torch.save(dict(schema_version=1,state=best,mean=mean.tolist(),std=std.tolist(),temporal=temporal,protocol_id=manifest['protocol_id'],seed=args.seed),out/'forecaster.pt')
    joblib.dump(models,out/'references.joblib',compress=3)
    metadata=dict(schema_version=1,protocol_id=manifest['protocol_id'],temporal=temporal,seed=args.seed,selected=selected,
        validation=scores,best_epoch=best_epoch,history=history,train_examples=len(tr),validation_examples=len(va),
        training_seconds=time.perf_counter()-started,majority_label=int(np.bincount(tr.y).argmax()),
        unit=manifest['config']['acceleration_unit'],hz=manifest['config']['hz'],window=manifest['config']['window'],
        files={n:sha(out/n) for n in ['forecaster.pt','references.joblib']},
        environment=dict(torch=torch.__version__,numpy=np.__version__),
        scope='future-window label on all valid contexts; not fall or event-onset prediction')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));print('Selected on validation:',selected,scores,flush=True)

def load_bundle(folder,device,inference_only=False):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text())
    if meta.get('schema_version')!=1:raise ValueError('Unsupported model bundle')
    need_references = not inference_only or meta['selected'] != 'gru'
    for name,digest in meta['files'].items():
        if name == 'references.joblib' and not need_references:
            continue
        if sha(folder/name)!=digest:raise ValueError('Model artifact checksum mismatch')
    ck=torch.load(folder/'forecaster.pt',map_location='cpu',weights_only=True)
    if ck['protocol_id']!=meta['protocol_id'] or ck['temporal']!=meta['temporal']:raise ValueError('Checkpoint metadata mismatch')
    model=CausalForecaster().to(device);model.load_state_dict(ck['state'])
    # Only load bundles you created/trust: joblib, like pickle, can execute code.
    refs=joblib.load(folder/'references.joblib') if need_references else None
    return meta,ck,model,refs

def all_probs(x,ck,model,refs,device):
    z=((x-np.asarray(ck['mean'],dtype=np.float32))/np.asarray(ck['std'],dtype=np.float32)).astype(np.float32)
    probs=dict(gru=neural_prob(model,z,device),last=rf_prob(refs['last'],x[:,-1]),context=rf_prob(refs['context'],x.reshape(len(x),-1)),persistence=rf_prob(refs['persistence'],x[:,-1]))
    probs['ensemble']=(probs['gru']+probs['context'])/2
    return probs

def selected_prob(x,ck,model,refs,device,selected):
    if selected == 'gru':
        z=((x-np.asarray(ck['mean'],dtype=np.float32))/np.asarray(ck['std'],dtype=np.float32)).astype(np.float32)
        return neural_prob(model,z,device)
    return all_probs(x,ck,model,refs,device)[selected]


def test(args):
    path=Path(args.out)
    if path.exists():raise ValueError('Test output exists; do not overwrite a held-out evaluation')
    meta,ck,model,refs=load_bundle(args.bundle,args.device);data,manifest=load_protocol(args.protocol)
    if manifest['protocol_id']!=meta['protocol_id']:raise ValueError('Test protocol differs from training')
    ds,x,provenance=make_split(data,manifest,'test',meta['temporal']);probs=all_probs(x,ck,model,refs,args.device)
    changed=ds.y!=provenance['current'];result=dict(protocol_id=meta['protocol_id'],selected=meta['selected'],temporal=meta['temporal'],seed=meta['seed'],
        horizon_seconds_min=float(ds.actual_horizon_samples.min()/meta['hz']),horizon_seconds_max=float(ds.actual_horizon_samples.max()/meta['hz']),methods={})
    for name,p in probs.items():
        pred=p.argmax(1)+1;result['methods'][name]=dict(all=metrics(ds.y,pred),changed_windows=metrics(ds.y[changed],pred[changed]),stable_windows=metrics(ds.y[~changed],pred[~changed]),per_subject={str(u):metrics(ds.y[provenance['subjects']==u],pred[provenance['subjects']==u]) for u in np.unique(provenance['subjects'])},confusion_matrix=confusion_matrix(ds.y,pred,labels=LABELS).tolist())
    result['constant']=metrics(ds.y,np.full(len(ds.y),meta['majority_label']))
    result['oracle_persistence']=metrics(ds.y,provenance['current'])
    result['transition_definition']='Diagnostic only: target-window label differs from last full context-window label, not annotated event onset.'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    np.savez_compressed(str(path)+'.predictions.npz',y=ds.y,changed=changed,**provenance,**{k:p for k,p in probs.items()})
    print(json.dumps({k:v['all'] for k,v in result['methods'].items()},indent=2),flush=True)

def predict(args):
    meta,ck,model,refs=load_bundle(args.bundle,args.device,inference_only=True);x=np.load(args.input,allow_pickle=False)
    expected=(meta['temporal']['seq_len'],max(1,int(meta['window']*meta['temporal']['obs_ratio'])),6)
    if x.ndim==3:x=x[None]
    if x.ndim!=4 or tuple(x.shape[1:])!=expected:raise ValueError(f'Expected prepared causal prefixes (batch, {expected}); no future/full-window inputs')
    p=selected_prob(window_features(x),ck,model,refs,args.device,meta['selected'])
    result=dict(labels=(p.argmax(1)+1).tolist(),probabilities=p.tolist(),class_ids=LABELS,selected=meta['selected'],temporal=meta['temporal'],warning='Uncalibrated probabilities; future-window labels, not fall alerts')
    path=Path(args.out)
    if path.exists():raise ValueError('Output exists')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))

def stream(args):
    from src.data.causal_stream import CausalWindowStream
    meta,ck,model,refs=load_bundle(args.bundle,args.device,inference_only=True)
    raw=np.load(args.input,allow_pickle=False)
    t=meta['temporal'];builder=CausalWindowStream(unit=meta['unit'],window=meta['window'],stride=t['stride'],ratio=t['obs_ratio'],seq_len=t['seq_len'],horizon_samples=t['horizon_samples'])
    outputs=[]
    for start in range(0,len(raw),250):
        items=builder.push(raw[start:start+250])
        if not items:continue
        probs=selected_prob(window_features(np.stack([q['context'] for q in items])),ck,model,refs,args.device,meta['selected'])
        for item,p in zip(items,probs):
            outputs.append(dict(decision_sample=item['decision_sample'],target_start=item['target_start'],horizon_seconds=item['horizon_seconds'],label=int(p.argmax()+1),probabilities=p.tolist()))
    path=Path(args.out)
    if path.exists():raise ValueError('Output exists')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(unit=meta['unit'],hz=meta['hz'],predictions=outputs),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--protocol',required=True);tr.add_argument('--out',required=True);tr.add_argument('--ratio',type=float,default=.5);tr.add_argument('--seq-len',type=int,default=5);tr.add_argument('--horizon-samples',type=int,default=50);tr.add_argument('--seed',type=int,default=42);tr.add_argument('--epochs',type=int,default=30);tr.add_argument('--patience',type=int,default=7);tr.add_argument('--batch-size',type=int,default=128);tr.add_argument('--lr',type=float,default=.001)
    te=sub.add_parser('test');te.add_argument('--protocol',required=True);te.add_argument('--bundle',required=True);te.add_argument('--out',required=True)
    pr=sub.add_parser('predict');pr.add_argument('--bundle',required=True);pr.add_argument('--input',required=True);pr.add_argument('--out',required=True)
    sr=sub.add_parser('stream');sr.add_argument('--bundle',required=True);sr.add_argument('--input',required=True);sr.add_argument('--out',required=True)
    for q in [tr,te,pr,sr]:q.add_argument('--device',default='cpu');q.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    if a.command=='train' and min(a.epochs,a.patience,a.batch_size,a.lr)<=0:p.error('Training parameters must be positive')
    {'train':train,'test':test,'predict':predict,'stream':stream}[a.command](a)
if __name__=='__main__':main()
