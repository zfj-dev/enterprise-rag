"""诊断（当前逻辑）：具体表号查询注入+过滤后的最终候选（LLM 实际看到什么）。
用法: .venv\\Scripts\\python.exe diagnose_named.py
"""
from __future__ import annotations
import os, re, sys

BACKEND = os.path.dirname(os.path.abspath(__file__))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
for k, v in {
    "HF_HUB_OFFLINE": "0", "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_DISABLE_SYMLINKS": "1", "HF_HUB_DISABLE_XET": "1",
    "USE_REAL": "true", "EMBEDDING_PROVIDER": "bge", "RERANKER_PROVIDER": "bge",
    "EMBEDDING_DEVICE": "cuda", "RERANKER_DEVICE": "cuda", "VECTOR_STORE": "inmemory",
}.items():
    os.environ.setdefault(k, v)

def looks_table(c):
    lines=(c or "").splitlines()
    return sum(1 for ln in lines if ln.lstrip().startswith("|"))>=2

def main():
    from app.core.parser import ParserRouter
    from app.core.chunker import ParentChildChunker
    from app.services.chat_service import _named_ref_intent

    parsed = ParserRouter().parse(os.path.join(BACKEND,"paper.pdf"), "paper.pdf")
    chunks=[]
    for pidx,pg in enumerate(parsed.pages,1):
        if not pg.strip(): continue
        chunks.extend(ParentChildChunker().chunk(pg,doc_id="diag",page_num=pidx))
    childs=[c for c in chunks if c["chunk_type"]=="child"]
    print(f"pages={len(parsed.pages)} child={len(childs)}")

    for q in ["表3.1的内容是什么","表3.2的内容是什么","表3.3的内容是什么"]:
        refs=_named_ref_intent(q)
        print(f"\n===== {q}  引用: {refs} =====")
        keys=[]
        merged=[]
        for prefix,n1,n2 in refs:
            key=f"{prefix}{n1}.{n2}" if n2 else f"{prefix}{n1}"
            keys.append(key)
            tables=[];refs_l=[]
            for c in childs:
                compact=re.sub(r"\s+","",c["content"] or "")
                if key not in compact: continue
                entry={"chunk_id":c["id"],"page":c["page_num"],"content":c["content"]}
                (tables if "|" in c["content"] else refs_l).append(entry)
            named=(tables[:2]+refs_l[:1])[:3]
            print(f"  注入 {len(named)} 个:")
            for n in named:
                print(f"    [{n['chunk_id']}] page={n['page']} head={(n['content'] or '')[:55]!r}")
            for n in named:
                if not any(x["chunk_id"]==n["chunk_id"] for x in merged):
                    merged.append(n)
        # 过滤：只保留含 key 的
        merged=[c for c in merged if any(k in re.sub(r"\s+","",c["content"] or "") for k in keys)]
        print(f"  过滤后保留 {len(merged)} 个:")
        for i,m in enumerate(merged,1):
            print(f"    {i}. [{m['chunk_id']}] page={m['page']} head={(m['content'] or '')[:60]!r}")

if __name__=="__main__":
    main()
