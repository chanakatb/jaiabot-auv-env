#!/usr/bin/env python3
"""
Inspect both USD files to understand the reference issue
"""

import os
from pxr import Usd, UsdGeom

def inspect_usd_file(usd_path, name):
    """Inspect a USD file and show its structure"""
    print(f"\n{'='*60}")
    print(f"Inspecting {name}: {usd_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(usd_path):
        print(f"❌ File not found: {usd_path}")
        return
    
    try:
        # Check file size
        file_size = os.path.getsize(usd_path)
        print(f"📁 File size: {file_size} bytes")
        
        # Try to open the stage
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print("❌ Could not open USD stage")
            return
        
        print("✅ USD stage opened successfully")
        
        # Show basic info
        print(f"🎭 Default prim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else 'None'}")
        
        # Count prims
        prim_count = len(list(stage.Traverse()))
        print(f"📦 Total prims: {prim_count}")
        
        # Show prim structure
        print("\n🌳 Prim structure:")
        for prim in stage.Traverse():
            depth = len(str(prim.GetPath()).split('/')) - 2
            indent = "  " * depth
            prim_type = prim.GetTypeName()
            
            print(f"{indent}├─ {prim.GetPath().name} ({prim_type})")
            
            # Check for references
            if prim.HasAuthoredReferences():
                print(f"{indent}   📎 Has references")
                
            # Check for meshes
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr()
                if points_attr.HasValue():
                    points = points_attr.Get()
                    print(f"{indent}   🔺 Mesh with {len(points) if points else 0} vertices")
                else:
                    print(f"{indent}   🔺 Mesh (no points data)")
        
        # Check for any materials
        print(f"\n🎨 Materials and shaders:")
        material_count = 0
        for prim in stage.Traverse():
            if prim.GetTypeName() in ['Material', 'Shader']:
                material_count += 1
                print(f"  📄 {prim.GetPath()}: {prim.GetTypeName()}")
        
        if material_count == 0:
            print("  No materials found")
            
    except Exception as e:
        print(f"❌ Error inspecting file: {e}")

def check_reference_paths():
    """Check the reference paths in the main USD file"""
    main_usd = "./data/warpauv/warpauv.usd"
    
    print(f"\n{'='*60}")
    print("🔍 Checking reference paths in main USD file")
    print(f"{'='*60}")
    
    try:
        with open(main_usd, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'warpauv_new_visual.usd' in line:
                print(f"Line {i+1}: {line.strip()}")
                # Show context (lines before and after)
                start = max(0, i-2)
                end = min(len(lines), i+3)
                print("Context:")
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"  {marker} {j+1}: {lines[j]}")
                print()
                
    except Exception as e:
        print(f"❌ Error reading main USD file: {e}")

if __name__ == "__main__":
    # Inspect both files
    inspect_usd_file("./data/warpauv/warpauv.usd", "Main WarpAUV USD")
    inspect_usd_file("./data/warpauv/warpauv_new_visual.usd", "Visual USD")
    
    # Check reference paths
    check_reference_paths()
    
    print(f"\n{'='*60}")
    print("💡 RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("1. If visual USD file looks good, the issue might be path resolution")
    print("2. Try using absolute paths in the reference")
    print("3. Ensure both files are in the same directory")
    print("4. Check file permissions")