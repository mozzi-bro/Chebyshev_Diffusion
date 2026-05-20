
import os
import sys
import argparse
import json
from glob import glob
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

try:
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


@dataclass
class FileValidationResult:
    filename: str
    is_valid: bool = True
    has_radius: bool = False
    n_points: int = 0
    n_segments: int = 0
    n_valid_segments: int = 0
    coord_range: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    radius_range: Tuple[float, float] = (0, 0)
    estimated_scale: str = "unknown"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DatasetValidationResult:
    dataset_name: str
    vtp_dir: str
    total_files: int = 0
    valid_files: int = 0
    files_with_radius: int = 0
    total_points: int = 0
    total_segments: int = 0
    avg_points_per_file: float = 0
    avg_segments_per_file: float = 0
    coord_range_global: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    radius_range_global: Tuple[float, float] = (0, 0)
    estimated_scale: str = "unknown"
    is_compatible: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    file_results: List[FileValidationResult] = field(default_factory=list)


def load_and_validate_vtp(vtp_path: str) -> FileValidationResult:
    filename = os.path.basename(vtp_path)
    result = FileValidationResult(filename=filename)
    
    try:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_path)
        reader.Update()
        polydata = reader.GetOutput()
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"Failed to read VTP: {e}")
        return result
    
    if polydata is None:
        result.is_valid = False
        result.errors.append("VTP file is empty or corrupted")
        return result
    
    points = polydata.GetPoints()
    if points is None:
        result.is_valid = False
        result.errors.append("No Points data")
        return result
    
    n_pts = points.GetNumberOfPoints()
    result.n_points = n_pts
    
    if n_pts < 2:
        result.is_valid = False
        result.errors.append(f"Too few points: {n_pts} (minimum 2 required)")
        return result
    
    if n_pts < 10:
        result.warnings.append(f"Low point count: {n_pts}")
    
    coords = np.array([points.GetPoint(i) for i in range(n_pts)], dtype=np.float64)
    
    if np.any(np.isnan(coords)):
        result.is_valid = False
        result.errors.append("NaN values in coordinates")
        return result
    
    if np.any(np.isinf(coords)):
        result.is_valid = False
        result.errors.append("Inf values in coordinates")
        return result
    
    result.coord_range = (
        float(coords[:, 0].min()), float(coords[:, 0].max()),
        float(coords[:, 1].min()), float(coords[:, 1].max()),
        float(coords[:, 2].min()), float(coords[:, 2].max())
    )
    
    radius_arr = polydata.GetPointData().GetArray("Radius")
    if radius_arr is not None:
        result.has_radius = True
        radii = np.array([radius_arr.GetValue(i) for i in range(n_pts)], dtype=np.float64)
        
        if np.any(np.isnan(radii)):
            result.warnings.append("NaN values in Radius")
        elif np.any(np.isinf(radii)):
            result.warnings.append("Inf values in Radius")
        else:
            valid_radii = radii[~np.isnan(radii) & ~np.isinf(radii)]
            if len(valid_radii) > 0:
                result.radius_range = (float(valid_radii.min()), float(valid_radii.max()))
                
                if valid_radii.min() <= 0:
                    result.warnings.append(f"Non-positive radius values: min={valid_radii.min():.4f}")
    else:
        result.warnings.append("No Radius field - default values will be used")
    
    lines = polydata.GetLines()
    if lines is None:
        result.is_valid = False
        result.errors.append("No Lines (segment) data")
        return result
    
    n_lines = lines.GetNumberOfCells()
    if n_lines == 0:
        result.is_valid = False
        result.errors.append("Zero segments")
        return result
    
    result.n_segments = n_lines
    
    lines.InitTraversal()
    idList = vtk.vtkIdList()
    valid_segments = 0
    
    while lines.GetNextCell(idList):
        n_seg_pts = idList.GetNumberOfIds()
        if n_seg_pts >= 2:
            valid_segments += 1
    
    result.n_valid_segments = valid_segments
    
    if valid_segments == 0:
        result.is_valid = False
        result.errors.append("No valid segments")
    
    extent_x = result.coord_range[1] - result.coord_range[0]
    extent_y = result.coord_range[3] - result.coord_range[2]
    extent_z = result.coord_range[5] - result.coord_range[4]
    max_extent = max(extent_x, extent_y, extent_z)
    
    if max_extent < 0.1:
        result.estimated_scale = "micro"
    elif max_extent < 10:
        result.estimated_scale = "small"
    elif max_extent < 500:
        result.estimated_scale = "mm"
    elif max_extent < 5000:
        result.estimated_scale = "large"
    else:
        result.estimated_scale = "very_large"
    
    return result


def validate_dataset(dataset_name: str, vtp_dir: str) -> DatasetValidationResult:
    result = DatasetValidationResult(dataset_name=dataset_name, vtp_dir=vtp_dir)
    
    if not os.path.exists(vtp_dir):
        result.is_compatible = False
        result.errors.append(f"Directory does not exist: {vtp_dir}")
        return result
    
    vtp_files = sorted(glob(os.path.join(vtp_dir, "*.vtp")))
    result.total_files = len(vtp_files)
    
    if len(vtp_files) == 0:
        result.is_compatible = False
        result.errors.append(f"No VTP files found: {vtp_dir}")
        return result
    
    if len(vtp_files) < 10:
        result.warnings.append(f"Low VTP file count: {len(vtp_files)}")
    
    print(f"  Validating: {len(vtp_files)} files...")
    
    all_coords = []
    all_radii = []
    
    for i, vtp_path in enumerate(vtp_files):
        file_result = load_and_validate_vtp(vtp_path)
        result.file_results.append(file_result)
        
        if file_result.is_valid:
            result.valid_files += 1
            result.total_points += file_result.n_points
            result.total_segments += file_result.n_valid_segments
            all_coords.append(file_result.coord_range)
            
            if file_result.has_radius:
                result.files_with_radius += 1
                if file_result.radius_range[1] > 0:
                    all_radii.append(file_result.radius_range)
        
        if (i + 1) % 100 == 0 or i == len(vtp_files) - 1:
            print(f"    [{i+1}/{len(vtp_files)}] valid: {result.valid_files}, invalid: {len(result.file_results) - result.valid_files}")
    
    if result.valid_files > 0:
        result.avg_points_per_file = result.total_points / result.valid_files
        result.avg_segments_per_file = result.total_segments / result.valid_files
        
        if all_coords:
            min_x = min(c[0] for c in all_coords)
            max_x = max(c[1] for c in all_coords)
            min_y = min(c[2] for c in all_coords)
            max_y = max(c[3] for c in all_coords)
            min_z = min(c[4] for c in all_coords)
            max_z = max(c[5] for c in all_coords)
            result.coord_range_global = (min_x, max_x, min_y, max_y, min_z, max_z)
        
        if all_radii:
            min_r = min(r[0] for r in all_radii)
            max_r = max(r[1] for r in all_radii)
            result.radius_range_global = (min_r, max_r)
    
    if result.coord_range_global[1] != 0:
        extent_x = result.coord_range_global[1] - result.coord_range_global[0]
        extent_y = result.coord_range_global[3] - result.coord_range_global[2]
        extent_z = result.coord_range_global[5] - result.coord_range_global[4]
        max_extent = max(extent_x, extent_y, extent_z)
        
        if max_extent < 0.1:
            result.estimated_scale = "micro"
            result.warnings.append("Very small data scale - check units")
        elif max_extent < 10:
            result.estimated_scale = "small"
            result.warnings.append("Small data scale - may be normalized units")
        elif max_extent < 500:
            result.estimated_scale = "mm"
        elif max_extent < 5000:
            result.estimated_scale = "large"
            result.warnings.append("Large data scale - hint_deviation_threshold may need adjustment")
        else:
            result.estimated_scale = "very_large"
            result.warnings.append("Very large data scale - check units")
    
    valid_ratio = result.valid_files / result.total_files if result.total_files > 0 else 0
    
    if valid_ratio < 0.5:
        result.is_compatible = False
        result.errors.append(f"Valid file ratio too low: {valid_ratio*100:.1f}%")
    
    radius_ratio = result.files_with_radius / result.valid_files if result.valid_files > 0 else 0
    if radius_ratio < 0.5:
        result.warnings.append(f"Few files with Radius field: {radius_ratio*100:.1f}%")
    
    return result


def print_validation_report(result: DatasetValidationResult) -> None:

    print("\n" + "=" * 70)
    print(f"  Dataset Validation Result: {result.dataset_name}")
    print("=" * 70)

    print(f"\n  Directory: {result.vtp_dir}")
    print(f"\n  File Statistics:")
    print(f"   - Total VTP files:     {result.total_files}")
    pct = result.valid_files/result.total_files*100 if result.total_files > 0 else 0
    print(f"   - Valid files:         {result.valid_files} ({pct:.1f}%)")
    pct2 = result.files_with_radius/result.valid_files*100 if result.valid_files > 0 else 0
    print(f"   - Files with Radius:   {result.files_with_radius} ({pct2:.1f}%)")

    if result.valid_files > 0:
        print(f"\n  Data Statistics:")
        print(f"   - Total Points:        {result.total_points:,}")
        print(f"   - Total Segments:      {result.total_segments:,}")
        print(f"   - Avg Points/file:     {result.avg_points_per_file:.1f}")
        print(f"   - Avg Segments/file:   {result.avg_segments_per_file:.1f}")

        extent = max(
            result.coord_range_global[1] - result.coord_range_global[0],
            result.coord_range_global[3] - result.coord_range_global[2],
            result.coord_range_global[5] - result.coord_range_global[4]
        )
        print(f"\n  Coordinate Range:")
        print(f"   - Max Extent: {extent:.2f}")
        print(f"   - Estimated scale: {result.estimated_scale}")

        if result.radius_range_global[1] > 0:
            print(f"\n  Radius range: [{result.radius_range_global[0]:.4f}, {result.radius_range_global[1]:.4f}]")

    if result.warnings:
        print(f"\n  Warnings ({len(result.warnings)}):")
        for w in result.warnings[:5]:
            print(f"   - {w}")
        if len(result.warnings) > 5:
            print(f"   ... and {len(result.warnings) - 5} more")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for e in result.errors[:5]:
            print(f"   - {e}")
        if len(result.errors) > 5:
            print(f"   ... and {len(result.errors) - 5} more")

    print("\n" + "-" * 70)
    if result.is_compatible:
        print("  PASS: Pipeline compatible")
    else:
        print("  FAIL: Pipeline incompatible")
    print("-" * 70)


def save_validation_report(result: DatasetValidationResult, output_path: str) -> None:
    report = {
        "dataset_name": result.dataset_name,
        "vtp_dir": result.vtp_dir,
        "is_compatible": result.is_compatible,
        "statistics": {
            "total_files": result.total_files,
            "valid_files": result.valid_files,
            "files_with_radius": result.files_with_radius,
            "total_points": result.total_points,
            "total_segments": result.total_segments,
        },
        "coord_range_global": result.coord_range_global,
        "radius_range_global": result.radius_range_global,
        "estimated_scale": result.estimated_scale,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VTP Dataset Pipeline Compatibility Validation")

    parser.add_argument('--dataset', type=str, help='Single dataset name')
    parser.add_argument('--datasets', nargs='+', help='Multiple dataset names')
    parser.add_argument('--vtp-dir', type=str, help='Directly specify VTP directory')
    parser.add_argument('--name', type=str, default='custom', help='Dataset name when using --vtp-dir')
    parser.add_argument('--root', type=str, default='.', help='Project root directory')
    parser.add_argument('--save-report', action='store_true', help='Save JSON report')
    
    args = parser.parse_args()
    
    if not HAS_VTK:
        print("ERROR: VTK is not installed. pip install vtk")
        sys.exit(1)
    
    datasets_to_validate = []
    
    if args.vtp_dir:
        datasets_to_validate.append((args.name, args.vtp_dir))
    elif args.datasets:
        for ds in args.datasets:
            vtp_dir = os.path.join(args.root, 'data', ds, 'raw_vtp')
            datasets_to_validate.append((ds, vtp_dir))
    elif args.dataset:
        vtp_dir = os.path.join(args.root, 'data', args.dataset, 'raw_vtp')
        datasets_to_validate.append((args.dataset, vtp_dir))
    else:
        parser.print_help()
        print("\nERROR: Specify one of --dataset, --datasets, or --vtp-dir.")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("  VTP Dataset Pipeline Compatibility Validation")
    print("=" * 70)
    
    results = []
    all_compatible = True
    
    for ds_name, vtp_dir in datasets_to_validate:
        result = validate_dataset(ds_name, vtp_dir)
        results.append(result)
        print_validation_report(result)
        
        if args.save_report:
            report_path = os.path.join(args.root, 'data', ds_name, f'validation_report.json')
            save_validation_report(result, report_path)
        
        if not result.is_compatible:
            all_compatible = False
    
    print("\n" + "=" * 70)
    print("  Final Summary")
    print("=" * 70)

    for result in results:
        status = "PASS" if result.is_compatible else "FAIL"
        print(f"  [{status}] {result.dataset_name}: {result.valid_files}/{result.total_files} valid")
    
    print("=" * 70)
    
    if all_compatible:
        print("\n  All datasets are pipeline compatible!")
        sys.exit(0)
    else:
        print("\n  Some datasets are not compatible.")
        sys.exit(1)


if __name__ == '__main__':
    main()
