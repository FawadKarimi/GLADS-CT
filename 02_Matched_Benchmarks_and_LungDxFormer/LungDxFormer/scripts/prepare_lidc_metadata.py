from __future__ import annotations
"""
Utility script scaffold for building ROI-level metadata from a local LIDC-IDRI style export.

Because LIDC XML and folder exports can differ, this script is intentionally conservative:
- parses generic XML annotations
- maps malignancy score to 3 classes
- writes a CSV template you can refine if needed

You can customize it for your local archive structure.
"""
import argparse
from pathlib import Path
import pandas as pd
from lungdxformer.data.xml_parser import parse_generic_lidc_xml
from lungdxformer.data.label_mapping import map_malignancy_score_to_class, ID_TO_LABEL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_root", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    rows = []
    xml_files = list(Path(args.xml_root).rglob("*.xml"))
    for xml_path in xml_files:
        patient_id = xml_path.parent.name
        try:
            nodules = parse_generic_lidc_xml(xml_path)
        except Exception:
            continue
        for idx, nodule in enumerate(nodules):
            malignancy = nodule.get("malignancy", None)
            if malignancy is None:
                continue
            label_id = map_malignancy_score_to_class(malignancy)
            rows.append({
                "patient_id": patient_id,
                "image_path": "",  # fill ROI path after extraction
                "label": ID_TO_LABEL[label_id],
                "malignancy_score": malignancy,
                "xml_path": str(xml_path),
                "nodule_index": idx,
            })

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved metadata template: {args.output_csv}")

if __name__ == "__main__":
    main()
