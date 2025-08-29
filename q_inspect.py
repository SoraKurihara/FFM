# tools/q_inspect.py
import pickle
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _detect_action_size(Q, scan_limit=100000):
    sizes = Counter()
    for i, (_, v) in enumerate(Q.items()):
        try: sizes[len(v)] += 1
        except: pass
        if i >= scan_limit: break
    if sizes: return sizes.most_common(1)[0][0]
    first = next(iter(Q.keys()))
    return len(Q[first])

def _schema(k):
    if isinstance(k, tuple):
        if len(k)==3 and isinstance(k[0],(bytes,bytearray)) and isinstance(k[1],(bytes,bytearray)) and isinstance(k[2],tuple):
            return "new3"  # (map_bytes, occ_bytes, (bx,by))
        if len(k)==2 and isinstance(k[1],tuple):
            return "tuple_block2"
    if isinstance(k,(bytes,bytearray)) and len(k)>=4:
        return "bytes_with_block_tail"
    return "unknown"

def _block(k):
    s=_schema(k)
    if s=="new3": return k[2]
    if s=="tuple_block2": return k[1]
    if s=="bytes_with_block_tail":
        try:
            bx=int.from_bytes(k[-4:-2],"little",signed=False)
            by=int.from_bytes(k[-2:],  "little",signed=False)
            return (bx,by)
        except: return None
    return None

def _occ3x3(k):
    if _schema(k)!="new3": return None
    a = np.frombuffer(k[1], dtype=np.uint8)
    return a.reshape(3,3) if a.size==9 else None

def _softmax(q):
    x = q - np.max(q)
    ex = np.exp(x); s = ex.sum()
    return ex/s if (s>0 and np.isfinite(s)) else np.ones_like(ex)/len(ex)

def analyze_q(q_path, sample_limit=200_000, rng_seed=42):
    with open(q_path,"rb") as f:
        Q = pickle.load(f)
    action_size = _detect_action_size(Q)
    stop_idx = action_size - 1

    # reservoir sample
    rng = random.Random(rng_seed)
    sample = []
    for idx, item in enumerate(Q.items()):
        if idx < sample_limit: sample.append(item)
        else:
            j = rng.randint(0, idx)
            if j < sample_limit: sample[j] = item

    entropies=[]; top_probs=[]; q_ranges=[]; adv_gaps=[]; stop_top=[]; p_stop=[]
    per_block = defaultdict(lambda: {"n":0,"stop_top":0,"low_range":0,"high_topprob":0})
    front_bins=[]; front_stop_top=Counter()

    for k,v in sample:
        q = np.asarray(v, dtype=np.float64).ravel()
        if q.size != action_size: continue
        p = _softmax(q)
        entropies.append(float(-(p*np.log(p+1e-12)).sum()))
        top_probs.append(float(p.max()))
        q_ranges.append(float(q.max()-q.min()))
        idxs = np.argsort(-q)
        if q.size>=2: adv_gaps.append(float(q[idxs[0]]-q[idxs[1]]))
        stop_top.append(int(np.argmax(p)==stop_idx))
        p_stop.append(float(p[stop_idx]) if stop_idx < q.size else np.nan)

        b = _block(k)
        if b is not None:
            per_block[b]["n"] += 1
            per_block[b]["stop_top"] += int(np.argmax(p)==stop_idx)
            per_block[b]["low_range"] += int((q.max()-q.min()) < 0.5)
            per_block[b]["high_topprob"] += int(p.max() >= 0.8)

        occ = _occ3x3(k)
        if occ is not None:
            up=int(occ[0].sum()); down=int(occ[2].sum()); left=int(occ[:,0].sum()); right=int(occ[:,2].sum())
            binned = min(max(up,down,left,right), 4)  # 0..4
            front_bins.append(binned)
            if int(np.argmax(p))==stop_idx: front_stop_top[binned]+=1

    sr=np.array(q_ranges); tp=np.array(top_probs); ent=np.array(entropies); gap=np.array(adv_gaps); st=np.array(stop_top); pst=np.array(p_stop)

    summary = pd.DataFrame([{
        "n_states_total": len(Q),
        "n_states_sampled": len(sample),
        "action_size": action_size,
        "q_range_p50": float(np.percentile(sr,50)) if sr.size else None,
        "q_range_p90": float(np.percentile(sr,90)) if sr.size else None,
        "entropy_p50": float(np.percentile(ent,50)) if ent.size else None,
        "entropy_p90": float(np.percentile(ent,90)) if ent.size else None,
        "top_prob_p50": float(np.percentile(tp,50)) if tp.size else None,
        "top_prob_p90": float(np.percentile(tp,90)) if tp.size else None,
        "adv_gap_p50": float(np.percentile(gap,50)) if gap.size else None,
        "adv_gap_p90": float(np.percentile(gap,90)) if gap.size else None,
        "share_stop_is_top": float(st.mean()) if st.size else None,
        "mean_p_stop": float(np.nanmean(pst)) if pst.size else None,
    }])

    block_rows=[]
    for (bx,by), d in per_block.items():
        n=d["n"]
        block_rows.append({
            "block_x": bx, "block_y": by, "n_states": n,
            "stop_top_share": d["stop_top"]/n if n else None,
            "low_range_share(<0.5)": d["low_range"]/n if n else None,
            "high_topprob_share(>=0.8)": d["high_topprob"]/n if n else None,
        })
    per_block_df = pd.DataFrame(block_rows).sort_values(["block_x","block_y"]).reset_index(drop=True)

    crowd_df = None
    if front_bins:
        total = Counter(front_bins)
        rows=[]
        for b in range(0,5):
            denom = max(1,total.get(b,0))
            rows.append({
                "front_crowding_bin(0-4cap)": b,
                "states_count": total.get(b,0),
                "stop_top_count": front_stop_top.get(b,0),
                "stop_top_share": front_stop_top.get(b,0)/denom,
            })
        crowd_df = pd.DataFrame(rows)

    return summary, per_block_df, crowd_df

if __name__=="__main__":
    import sys
    qpath = sys.argv[1] if len(sys.argv)>1 else "Q.pkl"
    summary, per_block_df, crowd_df = analyze_q(qpath)
    print(summary.to_string(index=False))
    if not per_block_df.empty:
        print("\\nPer-block flags (head):")
        print(per_block_df.head(12).to_string(index=False))
    if crowd_df is not None:
        print("\\nFront-crowding vs stop-top:")
        print(crowd_df.to_string(index=False))
    if crowd_df is not None:
        print("\\nFront-crowding vs stop-top:")
        print(crowd_df.to_string(index=False))
