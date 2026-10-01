# scripts/scoring_rules.py
#按照腿法打分
import numpy as np
from typing import Dict, Tuple
HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)
VTH = 0.2

def robust_scale(k, v):
    sw, hw = [], []
    for t in range(k.shape[0]):
        if v[t,LS]>VTH and v[t,RS]>VTH: sw.append(np.linalg.norm(k[t,LS]-k[t,RS]))
        if v[t,LH]>VTH and v[t,RH]>VTH: hw.append(np.linalg.norm(k[t,LH]-k[t,RH]))
    if sw: return float(np.median(sw))
    if hw: return float(np.median(hw))
    return 80.0

def normalize_xy(k, v):
    s = robust_scale(k,v)
    pel = (k[:,LH,:] + k[:,RH,:]) / 2.0
    x = (k - pel[:,None,:]) / max(s, 1e-3)
    return x

def choose_kicking_leg(x):
    radL = np.max(np.linalg.norm(x[:, LA, :], axis=-1), initial=0.0)
    radR = np.max(np.linalg.norm(x[:, RA, :], axis=-1), initial=0.0)
    return (LA, RA, LK, LH) if radL >= radR else (RA, LA, RK, RH)

def series_speed(x, fps):
    dx = np.diff(x, axis=0)
    sp = np.linalg.norm(dx, axis=-1) * float(max(fps,1.0))
    if sp.size >= 3: sp = np.convolve(sp, np.ones(3)/3.0, mode='same')
    return sp

def series_acc(sp, fps):
    if len(sp)<2: return np.array([0.0], np.float32)
    a = np.abs(np.diff(sp))*float(max(fps,1.0))
    if a.size>=3: a=np.convolve(a,np.ones(3)/3.0,mode='same')
    return a

def knee_angle_deg(hip,knee,ank):
    v1,v2=hip-knee,ank-knee; n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6: return 180.0
    c=np.clip(np.dot(v1,v2)/(n1*n2),-1,1); return float(np.degrees(np.arccos(c)))

def path_straightness(x):
    if len(x)<3: return 1.0
    seg=np.linalg.norm(np.diff(x,axis=0),axis=-1).sum(); chord=np.linalg.norm(x[-1]-x[0])
    return 1.0 if seg<1e-6 else float(np.clip(chord/seg,0,1))

def guard_closeness_and_stability(x,v):
    head=x[:,HEAD,:]; ok=v[:,HEAD]>VTH
    if not np.any(ok): return 0.5,0.5
    hmed=np.median(head[ok],axis=0)
    dists,rngs=[],[]
    for k in (LW,RW):
        o=v[:,k]>VTH
        if np.any(o):
            d=np.linalg.norm(x[o,k,:]-hmed,axis=-1); dists.append(np.median(d))
            p=x[o,k,:]; rng=np.linalg.norm(np.percentile(p,95,axis=0)-np.percentile(p,5,axis=0))
            rngs.append(rng)
    dist_med=float(np.median(dists)) if dists else 0.6
    rng_med=float(np.median(rngs)) if rngs else 0.6
    close=float(np.clip(1.0-(dist_med-0.3)/0.5,0,1))
    stab=float(np.clip(1.0-(rng_med-0.2)/0.4,0,1))
    return close,stab

def compute_raws(k,v,fps)->Dict:
    x=normalize_xy(k,v)
    kick_ank,_,kick_knee,kick_hip=choose_kicking_leg(x)
    sp=series_speed(x[:,kick_ank,:],fps); ac=series_acc(sp,fps)
    speed_raw=float(np.percentile(sp,90)) if len(sp) else 0.0
    accel_raw=float(np.percentile(ac,90)) if len(ac) else 0.0
    power_raw=0.7*speed_raw+0.3*accel_raw
    height_raw=float(np.max(-x[:,kick_ank,1])) if len(x)>0 else 0.0
    # 下劈：下行速度
    if len(x)>1:
        vy=np.diff(x[:,kick_ank,1])*float(max(fps,1.0))
        down_speed=float(np.percentile(np.maximum(vy,0),90))
    else:
        down_speed=0.0
    t_peak=int(np.argmax(-x[:,kick_ank,1])) if len(x)>0 else 0
    ang=knee_angle_deg(x[t_peak,kick_hip],x[t_peak,kick_knee],x[t_peak,kick_ank]) if len(x)>0 else 180.0
    knee_score=float(np.clip((ang-120.0)/60.0,0,1))
    t0=max(0,t_peak-3); t1=min(x.shape[0],t_peak+4)
    straight=path_straightness(x[t0:t1,kick_ank,:])
    close,stab=guard_closeness_and_stability(x,v)
    return dict(speed_raw=speed_raw,power_raw=power_raw,height_raw=height_raw,
                down_speed=down_speed,knee_score=knee_score,straight=straight,
                guard_close=close,guard_stab=stab)

def robust_minmax(vals, lo=10, hi=90):
    vals=np.array(vals,np.float32)
    if len(vals)==0: return 0.0,1.0
    vmin=float(np.percentile(vals,lo)); vmax=float(np.percentile(vals,hi))
    if vmax-vmin<1e-6: vmax=vmin+1.0
    return vmin,vmax

def to_unit(x,lo,hi): return float(np.clip((x-lo)/max(hi-lo,1e-6),0,1))

DEFAULT_WEIGHTS = {
    "front":      {"acc_corr":[0.7,0.3],        "expr":[0.35,0.25,0.40], "axe_down":0.0},
    "roundhouse": {"acc_corr":[0.6,"arc:0.4"],  "expr":[0.35,0.35,0.30], "axe_down":0.0},
    "side":       {"acc_corr":[0.6,0.4],        "expr":[0.30,0.30,0.40], "axe_down":0.0},
    "axe":        {"acc_corr":[0.6,0.4],        "expr":[0.25,0.25,0.40], "axe_down":0.10},
}

def score_one_video(raw:Dict, ktype:str, ranges:Dict):
    w = DEFAULT_WEIGHTS.get(ktype, DEFAULT_WEIGHTS["front"])
    s_min,s_max=ranges["speed"]; p_min,p_max=ranges["power"]; h_min,h_max=ranges["height"]; d_min,d_max=ranges["down"]
    knee=raw["knee_score"]; straight=raw["straight"]; arc=float(np.clip((0.9-straight)/0.4,0,1))
    a0,a1=w["acc_corr"][0], w["acc_corr"][1]
    acc_corr = a0*knee + (float(a1)*straight if not isinstance(a1,str) else float(a1.split(":")[1])*arc)
    hand = 0.6*raw["guard_close"] + 0.4*raw["guard_stab"]
    acc4 = float(np.clip(2.0*acc_corr + 2.0*hand, 0.0, 4.0))
    sp=2.0*to_unit(raw["speed_raw"],s_min,s_max)
    pw=2.0*to_unit(raw["power_raw"],p_min,p_max)
    ht=2.0*to_unit(raw["height_raw"],h_min,h_max)
    expr6=float(np.clip(w["expr"][0]*sp + w["expr"][1]*pw + w["expr"][2]*ht, 0.0, 6.0))
    if ktype=="axe" and w.get("axe_down",0.0)>0:
        ds=2.0*to_unit(raw["down_speed"],d_min,d_max)
        expr6=float(np.clip(expr6 + w["axe_down"]*ds, 0.0, 6.0))
    total=float(np.clip(acc4+expr6,0.0,10.0))
    return acc4, expr6, total
