# -*- coding: utf-8 -*-
import re
from typing import Dict, Optional, Any
from config import DAILY_GUIDE


def parse_nutrition_text(nut_text: str) -> Dict[str, Optional[float]]:
    if not nut_text:
        return {"kcal": None, "protein_g": None, "fat_g": None, "carb_g": None, "sugar_g": None, "sodium_mg": None}
    text = nut_text.lower()

    def find_number(patterns, unit_multiplier=1.0):
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1)) * unit_multiplier
                except:
                    continue
        return None

    kcal = find_number([
        r"kcal\s*([0-9]+(?:\.[0-9]+)?)",
        r"พลังงาน\s*([0-9]+(?:\.[0-9]+)?)\s*kcal",
        r"พลังงาน\s*([0-9]+(?:\.[0-9]+)?)",
    ])
    protein_g = find_number([
        r"โปรตีน\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"protein\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"โปรตีน\s*([0-9]+(?:\.[0-9]+)?)",
    ])
    fat_g = find_number([
        r"ไขมัน\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"fat\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"ไขมัน\s*([0-9]+(?:\.[0-9]+)?)",
    ])
    carb_g = find_number([
        r"คาร์บ(?:โฮไฮเดรต)?\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"carb(?:ohydrate)?s?\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"คาร์บ(?:โฮไฮเดรต)?\s*([0-9]+(?:\.[0-9]+)?)",
    ])
    sugar_g = find_number([
        r"น้ำตาล\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"sugar\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        r"น้ำตาล\s*([0-9]+(?:\.[0-9]+)?)",
    ])
    sodium_mg = find_number([
        r"โซเดียม\s*([0-9]+(?:\.[0-9]+)?)\s*mg",
        r"sodium\s*([0-9]+(?:\.[0-9]+)?)\s*mg",
    ])
    if sodium_mg is None:
        sodium_mg = find_number([
            r"โซเดียม\s*([0-9]+(?:\.[0-9]+)?)\s*g",
            r"sodium\s*([0-9]+(?:\.[0-9]+)?)\s*g",
        ], unit_multiplier=1000.0)

    return {"kcal": kcal, "protein_g": protein_g, "fat_g": fat_g, "carb_g": carb_g, "sugar_g": sugar_g, "sodium_mg": sodium_mg}


def evaluate_against_guidelines(nut: Dict[str, Optional[float]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if nut.get("kcal") is not None:
        pct = (nut["kcal"] / DAILY_GUIDE["energy_kcal"]) * 100.0
        out["energy_kcal"] = {"value": round(nut["kcal"], 2), "percent_of_daily": round(pct, 1), "status": "OK" if pct <= 100 else "HIGH"}
    if nut.get("fat_g") is not None:
        pct = (nut["fat_g"] / DAILY_GUIDE["fat_g_max"]) * 100.0
        out["fat_g"] = {"value": round(nut["fat_g"], 2), "percent_of_daily_max": round(pct, 1), "status": "OK" if pct <= 100 else "HIGH"}
    if nut.get("protein_g") is not None:
        v = nut["protein_g"]
        status = "LOW" if v < DAILY_GUIDE["protein_g_min"] else ("HIGH" if v > DAILY_GUIDE["protein_g_max"] else "OK")
        out["protein_g"] = {"value": round(v, 2), "range_reco_g": [DAILY_GUIDE["protein_g_min"], DAILY_GUIDE["protein_g_max"]], "status": status}
    if nut.get("sugar_g") is not None:
        v = nut["sugar_g"]
        pct10 = (v / DAILY_GUIDE["sugar_g_max"]) * 100.0
        pct5  = (v / DAILY_GUIDE["sugar_g_ideal"]) * 100.0
        status = "GREAT" if v <= DAILY_GUIDE["sugar_g_ideal"] else ("OK" if v <= DAILY_GUIDE["sugar_g_max"] else "HIGH")
        out["sugar_g"] = {"value": round(v, 2), "percent_of_10pct_cap": round(pct10, 1), "percent_of_5pct_ideal": round(pct5, 1), "status": status}
    if nut.get("sodium_mg") is not None:
        pct = (nut["sodium_mg"] / DAILY_GUIDE["sodium_mg_max"]) * 100.0
        out["sodium_mg"] = {"value": round(nut["sodium_mg"], 2), "percent_of_daily_max": round(pct, 1), "status": "OK" if pct <= 100 else "HIGH"}
    return out
