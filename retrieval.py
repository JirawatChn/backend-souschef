# -*- coding: utf-8 -*-
from typing import List, Dict, Any
from config import embed_query, index, child_meta, docstore


def retrieve_parents(query: str, k_children: int = 40, top_parents: int = 3) -> List[Dict[str, Any]]:
    q = embed_query(query)
    D, I = index.search(q, k_children)

    seen, parents = set(), []
    for idx in I[0]:
        pid = child_meta[idx]["parent_id"]
        if pid in seen:
            continue
        seen.add(pid)
        par = docstore.get(pid)
        if par:
            parents.append(par)
        if len(parents) >= top_parents:
            break
    return parents


def build_context_blocks(parents: List[Dict[str, Any]]) -> str:
    blocks = []
    for p in parents:
        name = p.get("menu_name", "")
        ing  = p.get("ingredients", "")
        mth  = p.get("method", "")
        nut  = p.get("nutrition_text", "")
        tags = p.get("tags", "")
        block = (
f"""[MENU]
- ชื่อ: {name}
- แท็ก: {tags or "-"}
- ส่วนผสมหลัก: {ing if ing else "-"}
- วิธีทำ: {mth if mth else "-"}
- โภชนาการ: {nut or "ไม่มีข้อมูล"}"""
        )
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else ""
