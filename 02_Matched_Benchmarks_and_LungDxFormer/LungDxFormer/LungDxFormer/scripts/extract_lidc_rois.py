from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm
import sys

# Add project root and src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from lungdxformer.data.xml_parser import parse_generic_lidc_xml
from lungdxformer.data.label_mapping import map_malignancy_score_to_class, ID_TO_LABEL
from lungdxformer.data.preprocessing import hu_clip, minmax_normalize
from lungdxformer.data.roi_extraction import crop_centered_roi

def find_closest_slice(dicom_files: list[Path], target_z: float) -> tuple[pydicom.dataset.FileDataset, float]:
    best_file = None
    min_diff = float("inf")
    
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if not hasattr(ds, "ImagePositionPatient"):
                continue
            z = float(ds.ImagePositionPatient[2])
            diff = abs(z - target_z)
            if diff < min_diff:
                min_diff = diff
                best_file = f
        except Exception:
            continue
            
    if best_file is None:
        raise ValueError("Could not find any valid DICOM file with ImagePositionPatient")
        
    return pydicom.dcmread(str(best_file)), min_diff

def main():
    parser = argparse.ArgumentParser(description="Extract 2D nodule ROIs from LIDC-IDRI DICOMs using XMLs")
    parser.add_argument("--lidc_root", required=True, help="Root directory containing LIDC-IDRI-XXXX folders")
    parser.add_argument("--output_dir", required=True, help="Output directory to save extracted patches")
    parser.add_argument("--image_size", type=int, default=96, help="Target patch image size")
    parser.add_argument("--max_patients", type=int, default=None, help="Limit number of patients to process (for testing/partial runs)")
    args = parser.parse_args()

    lidc_root = Path(args.lidc_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all XML files
    print("Scanning for XML annotation files...")
    xml_files = sorted(list(lidc_root.rglob("*.xml")))
    print(f"Found {len(xml_files)} XML files.")

    records = []
    processed_patients = set()
    
    for xml_path in tqdm(xml_files, desc="Extracting ROIs"):
        # The parent directory of the XML file contains the DICOM slice files
        series_dir = xml_path.parent
        
        # Walk up the path to find the LIDC-IDRI-XXXX patient folder
        patient_id = None
        for parent in xml_path.parents:
            if parent.name.startswith("LIDC-IDRI-") and parent.name != "LIDC-IDRI":
                patient_id = parent.name
                break
        if patient_id is None:
            continue
                
        if args.max_patients is not None:
            if patient_id not in processed_patients and len(processed_patients) >= args.max_patients:
                continue
            processed_patients.add(patient_id)

        # Get list of DICOM files in the same directory
        dicom_files = list(series_dir.glob("*.dcm"))
        if not dicom_files:
            continue

        try:
            nodules = parse_generic_lidc_xml(xml_path)
        except Exception as e:
            print(f"Error parsing XML {xml_path.name}: {e}")
            continue

        for idx, nodule in enumerate(nodules):
            malignancy = nodule.get("malignancy", None)
            if malignancy is None:
                continue
            
            roi_points = nodule.get("roi_points", [])
            if not roi_points:
                continue

            # Calculate average coordinates
            xs = [pt[0] for pt in roi_points]
            ys = [pt[1] for pt in roi_points]
            zs = [pt[2] for pt in roi_points]
            
            avg_x = int(np.mean(xs))
            avg_y = int(np.mean(ys))
            avg_z = float(np.mean(zs))

            try:
                # Find the closest slice
                ds, z_diff = find_closest_slice(dicom_files, avg_z)
                if z_diff > 3.0: # Skip if the closest slice is more than 3mm away
                    continue
                
                # Extract image array
                slice_img = ds.pixel_array.astype(np.float32)
                
                # Apply rescale slope/intercept to get HU
                if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                    slope = float(ds.RescaleSlope)
                    intercept = float(ds.RescaleIntercept)
                    slice_img = slice_img * slope + intercept
                
                # Clip HU to lung window [-1000, 400]
                slice_img = hu_clip(slice_img, min_hu=-1000, max_hu=400)
                
                # Normalize to [0, 255]
                slice_img = minmax_normalize(slice_img)
                slice_img = (slice_img * 255).astype(np.uint8)

                # Crop centered patch
                roi = crop_centered_roi(slice_img, avg_x, avg_y, patch_size=args.image_size)
                
                # Save patch
                patient_output_dir = output_dir / patient_id
                patient_output_dir.mkdir(parents=True, exist_ok=True)
                
                file_name = f"nodule_{idx}_{int(avg_z)}.png"
                cv2.imwrite(str(patient_output_dir / file_name), roi)
                
                label_id = map_malignancy_score_to_class(malignancy)
                label_name = ID_TO_LABEL[label_id]
                
                rel_path = str(Path(patient_id) / file_name)
                records.append({
                    "patient_id": patient_id,
                    "image_path": rel_path,
                    "label": label_name,
                    "malignancy_score": malignancy
                })
                
            except Exception as e:
                # print(f"Error extracting nodule from {patient_id}: {e}")
                continue

    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_dir / "metadata.csv", index=False)
        print(f"\nSuccessfully extracted {len(df)} nodule crops.")
        print(f"Metadata file written to: {output_dir / 'metadata.csv'}")
    else:
        print("\nNo nodules were successfully extracted.")

if __name__ == "__main__":
    main()
