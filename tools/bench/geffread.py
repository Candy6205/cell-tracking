import json, os, glob, numpy as np, sys
try:
    import zstandard as zstd
    dec = lambda b: zstd.ZstdDecompressor().decompress(b) if False else zstd.ZstdDecompressor().decompressobj().decompress(b)
except ImportError:
    from numcodecs import Zstd
    dec = lambda b: Zstd().decode(b)
def arr(path):
    meta = json.load(open(os.path.join(path, "zarr.json")))
    shape = meta["shape"]; dt = np.dtype(meta["data_type"]).newbyteorder("<")
    codecs = [c["name"] for c in meta["codecs"]]
    raw = open(os.path.join(path, "c", *(["0"]*len(shape))), "rb").read()
    if "zstd" in codecs: raw = dec(raw)
    elif "gzip" in codecs:
        import gzip; raw = gzip.decompress(raw)
    return np.frombuffer(raw, dtype=dt).reshape(shape)
def load(p):
    n = lambda k: arr(os.path.join(p, k))
    return dict(nid=n("nodes/ids"), nsol=n("nodes/props/solution/values").astype(bool), t=n("nodes/props/t/values"),
                eid=n("edges/ids"), esol=n("edges/props/solution/values").astype(bool), ep=n("edges/props/edge_prob/values"))
if __name__ == "__main__":
    for p in sorted(glob.glob("nb948_out/tracking_repo/predictions/unknown/unet_transformer/split_0/*.geff")):
        g = load(p)
        solids = set(g["nid"][g["nsol"]].tolist()); se = g["eid"][g["esol"]]
        touched = set(se[:,0].tolist()) | set(se[:,1].tolist())
        print(os.path.basename(p), "nodes", len(g["nid"]), "sol_nodes", int(g["nsol"].sum()), "edges", len(g["eid"]), "sol_edges", int(g["esol"].sum()),
              "sol_nodes_isolated", len(solids - touched), "nodes_touched_by_any_edge", len(set(g["eid"][:,0].tolist())|set(g["eid"][:,1].tolist())),
              "sol p med %.3f" % np.median(g["ep"][g["esol"]]), "nonsol p med %.3f" % (np.median(g["ep"][~g["esol"]]) if (~g["esol"]).any() else -1))
