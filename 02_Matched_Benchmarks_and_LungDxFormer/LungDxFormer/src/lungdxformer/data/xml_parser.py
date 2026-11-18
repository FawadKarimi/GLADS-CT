from __future__ import annotations
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

def parse_generic_lidc_xml(xml_path: str | Path) -> list[dict[str, Any]]:
    """
    Generic XML parser for LIDC-style annotation exports.
    Because local XML exports may vary, this parser intentionally extracts
    commonly used fields and may need minor adaptation for a specific archive.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodules: list[dict[str, Any]] = []
    for elem in root.iter():
        tag = elem.tag.lower()
        if tag.endswith("unblindedreadnodule") or tag.endswith("nodule"):
            item = {"roi_points": [], "malignancy": None}
            for child in elem.iter():
                ctag = child.tag.lower()
                text = (child.text or "").strip()
                if ctag.endswith("malignancy") and text:
                    try:
                        item["malignancy"] = float(text)
                    except ValueError:
                        pass
                if ctag.endswith("xcoord") and text:
                    item.setdefault("_last_x", []).append(int(float(text)))
                if ctag.endswith("ycoord") and text:
                    item.setdefault("_last_y", []).append(int(float(text)))
                if ctag.endswith("imagezposition") and text:
                    item.setdefault("_z", []).append(float(text))
            xs = item.pop("_last_x", [])
            ys = item.pop("_last_y", [])
            zs = item.pop("_z", [])
            item["roi_points"] = list(zip(xs, ys, zs[:len(xs)] if zs else [0.0] * len(xs)))
            nodules.append(item)
    return nodules
