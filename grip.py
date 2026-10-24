"""Reference-relative grip deformation; all dimensions are millimetres."""
from pathlib import Path
import hashlib, json
import numpy as np
import trimesh
import pymeshlab
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/smoothed_mergedhandle_V2.stl'
BASE = ROOT / 'data/baseline.npz'
ALGORITHM = 'grip-centreline-1'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def meshset(mesh):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)))
    return ms

def edge_counts(mesh):
    e = np.sort(mesh.edges, axis=1)
    unique, counts = np.unique(e, axis=0, return_counts=True)
    return unique, counts

def validate(mesh):
    finite = bool(np.isfinite(mesh.vertices).all())
    edges, counts = edge_counts(mesh)
    bad_edges = edges[counts != 2]
    area_bad = np.where(mesh.area_faces < 1e-10)[0]
    intersections = []
    error = None
    if finite:
        try:
            ms = meshset(mesh)
            ms.compute_selection_by_self_intersections_per_face()
            intersections = np.flatnonzero(ms.current_mesh().face_selection_array()).tolist()
        except Exception as exc:
            error = str(exc)
    else:
        error = 'Non-finite vertex coordinates'
    edge_faces = np.flatnonzero(np.isin(mesh.faces, bad_edges.ravel()).any(axis=1)) if len(bad_edges) else []
    bad = sorted(set(intersections) | set(map(int, area_bad)) | set(map(int, edge_faces)))
    consistent = bool(mesh.is_winding_consistent)
    volume_positive = bool(mesh.volume > 0)
    if not consistent or not volume_positive: bad = list(range(len(mesh.faces)))
    ok = finite and mesh.is_watertight and consistent and volume_positive and not len(area_bad) and not intersections and error is None
    return dict(valid=bool(ok), watertight=bool(mesh.is_watertight), consistent_normals=consistent,
                positive_volume=volume_positive, boundary_edges=int(sum(counts == 1)),
                nonmanifold_edges=int(sum(counts > 2)), degenerate_faces=len(area_bad),
                intersecting_faces=len(intersections), intersection_check_error=error,
                bad_faces=bad, vertices=len(mesh.vertices), faces=len(mesh.faces),
                dimensions_mm=mesh.extents.tolist(), volume_mm3=float(mesh.volume))

def prepare():
    sha = digest(SOURCE)
    report_path = ROOT / 'data/preparation.json'
    if BASE.exists() and report_path.exists() and json.loads(report_path.read_text()).get('source_sha256') == sha:
        return
    original = trimesh.load(SOURCE, force='mesh', process=True)
    original.update_faces(original.unique_faces())
    original.update_faces(original.nondegenerate_faces())
    original.remove_unreferenced_vertices()
    # Connectivity without automatic hole-filling, preserving a reliable before/after comparison.
    components = trimesh.graph.connected_components(original.face_adjacency, nodes=np.arange(len(original.faces)), min_len=1)
    keep = max(components, key=len)
    cleaned = original.submesh([keep], append=True, repair=False)
    trimesh.repair.fix_normals(cleaned, multibody=True)
    before = validate(cleaned)
    ms = meshset(cleaned)
    ms.meshing_close_holes(maxholesize=10000, refinehole=False, selfintersection=True)
    cm = ms.current_mesh()
    repaired = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=False)
    trimesh.repair.fix_normals(repaired, multibody=True)
    after = validate(repaired)
    removed_intersection_faces = 0
    if after['intersecting_faces']:
        mask = np.zeros(len(repaired.faces),dtype=bool)
        mask[after['bad_faces']] = True
        for _ in range(2):
            pairs = repaired.face_adjacency
            mask[np.unique(pairs[mask[pairs].any(axis=1)])] = True
        removed_intersection_faces = int(mask.sum())
        repaired.update_faces(~mask)
        repaired.remove_unreferenced_vertices()
        ms = meshset(repaired)
        ms.meshing_close_holes(maxholesize=10000,refinehole=False,selfintersection=True)
        cm = ms.current_mesh()
        repaired = trimesh.Trimesh(cm.vertex_matrix(),cm.face_matrix(),process=False)
        trimesh.repair.fix_normals(repaired,multibody=True)
        after = validate(repaired)
    distances, _ = cKDTree(cleaned.triangles_center).query(repaired.triangles_center)
    repair_faces = np.flatnonzero(distances > 1e-7)
    # PCA basis with deterministic signs; right handed local x = long axis.
    origin = repaired.vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(repaired.vertices - origin, full_matrices=False)
    basis = vh.T
    for j in range(2):
        if basis[np.argmax(np.abs(basis[:,j])),j] < 0: basis[:,j] *= -1
    basis[:,2] = np.cross(basis[:,0], basis[:,1])
    local = (repaired.vertices - origin) @ basis
    raw_local = (original.vertices - origin) @ basis
    # Sections give a geometric centreline, independent of vertex density around the circumference.
    lo, hi = local[:,0].min(), local[:,0].max()
    knots = np.linspace(lo, hi, 41)
    centres = []
    for x in knots:
        points = local[np.abs(local[:,0]-x) < (hi-lo)/35]
        if len(points) < 3: points = local[np.argsort(abs(local[:,0]-x))[:30]]
        centres.append((np.quantile(points[:,1:], .05, axis=0)+np.quantile(points[:,1:], .95, axis=0))/2)
    centres = gaussian_filter1d(np.array(centres), 2, axis=0, mode='nearest')
    np.savez_compressed(BASE, vertices=local, faces=repaired.faces, original_vertices=raw_local,
                        original_faces=original.faces, knots=knots, centres=centres, origin=origin, basis=basis,
                        repair_faces=repair_faces)
    report = dict(source=SOURCE.name, source_sha256=sha, removed_fragment_faces=len(original.faces)-len(cleaned.faces),
                  added_faces=len(repair_faces), removed_intersection_faces=removed_intersection_faces, before=before, after=after,
                  local_bounds=[local.min(0).tolist(),local.max(0).tolist()], algorithm=ALGORITHM)
    report_path.write_text(json.dumps(report, indent=2))

def baseline():
    with np.load(BASE) as z: return {k:z[k].copy() for k in z.files}

def number(value, label, minimum=0, maximum=1000):
    if isinstance(value, bool): raise ValueError(f'{label} must be a number')
    try: v = float(value)
    except (ValueError, TypeError): raise ValueError(f'{label} must be a number')
    if not np.isfinite(v) or not minimum < v <= maximum: raise ValueError(f'{label} must be greater than {minimum} and at most {maximum}')
    return v

def measurements(data):
    if not isinstance(data,dict): raise ValueError('Measurements must be an object')
    labels = {'hand_length':'Hand length','hand_breadth':'Hand breadth','palm_length':'Palm length'}
    for k,label in labels.items():
        if data.get(k) in (None,''): raise ValueError(f'Enter {label.lower()} in millimetres, or use the reference measurements')
    m = {k:number(data.get(k), label) for k,label in labels.items()}
    if m['hand_length'] <= m['palm_length']: raise ValueError('Hand length must exceed palm length')
    for k in ['hand_circumference','thumb_length','index_length','middle_length','ring_length','little_length',
              'thumb_breadth','index_breadth','middle_breadth','ring_breadth','little_breadth']:
        if data.get(k) not in (None,''): m[k] = number(data[k],k)
    return m

def calibration(data):
    if not isinstance(data,dict): raise ValueError('Calibration must be an object')
    if data.get('algorithm',ALGORITHM) != ALGORITHM: raise ValueError('Unsupported calibration algorithm')
    if data.get('accepted_repairs') is not True: raise ValueError('Review and accept the repair preview first')
    if data.get('source_sha256') != digest(SOURCE): raise ValueError('Calibration belongs to a different source model')
    if data.get('thumb_end') not in (-1,1): raise ValueError('Choose the thumb end')
    start = number(data.get('usable_start'), 'Usable start',-0.00001, 1)
    end = number(data.get('usable_end'), 'Usable end',0,1)
    if end-start < .1: raise ValueError('Usable region must span at least 10% of the model')
    angle = number(data.get('palm_angle'), 'Palm angle',-0.00001,360)
    if not data.get('landmarks_confirmed'): raise ValueError('Confirm the palm/finger labels and grip region')
    return dict(reference=measurements(data.get('reference')), thumb_end=int(data['thumb_end']),
                palm_angle=angle,usable_start=start,usable_end=end,accepted_repairs=True,landmarks_confirmed=True,
                source_sha256=data['source_sha256'],schema_version=1,algorithm=ALGORITHM)

def profile(data):
    if not isinstance(data,dict): raise ValueError('Profile must be an object')
    target = measurements(data.get('target'))
    if data.get('algorithm',ALGORITHM) != ALGORITHM: raise ValueError('Unsupported profile algorithm')
    adjustment = data.get('adjustments',{})
    if not isinstance(adjustment,dict): raise ValueError('Adjustments must be an object')
    a = {k:number(adjustment.get(k,1), k,.49,1.8) for k in ['length','thickness','ovality','palm_fullness']}
    if any(abs(v-1)>1e-12 for v in a.values()): raise ValueError('Independent shape adjustments are no longer supported. Use hand measurements to deform the existing imprint.')
    pose=data.get('hand_pose',{})
    if not isinstance(pose,dict): raise ValueError('Hand pose must be an object')
    hand_pose={}
    for k in ('x','y','z','rx','ry','rz','palm_angle'):
        limit=180 if k.startswith('r') or k=='palm_angle' else 150
        hand_pose[k]=number(pose.get(k,0),f'Hand position {k}',-limit-.00001,limit)
    return dict(name=str(data.get('name','Personalized grip'))[:100],target=target,adjustments=a,hand_pose=hand_pose,imprint_mode='fixed-existing-surface',
                schema_version=1, algorithm=ALGORITHM,fit_status='Unvalidated — physical fit testing required')

def warp_points(v, cal, pro, b=None):
    b = baseline() if b is None else b
    v = np.asarray(v,dtype=float); x = v[:,0]
    ref, target, a = cal['reference'], pro['target'], pro['adjustments']
    length = target['hand_breadth']/ref['hand_breadth']*a['length']
    wrap = (target['hand_length']-target['palm_length'])/(ref['hand_length']-ref['palm_length'])
    radial = wrap*a['thickness']
    palm = (target['palm_length']/ref['palm_length'])/wrap*a['palm_fullness']
    centre = CubicSpline(b['knots'],b['centres'],axis=0)(x)
    offsets = v[:,1:] - centre
    theta = np.deg2rad(cal['palm_angle']); direction = np.array([np.cos(theta),np.sin(theta)])
    side = np.column_stack([direction,[-direction[1],direction[0]]])
    q = offsets @ side
    # Smooth half-space weighting: palm side = 1, finger side = 0; no seams at the transition.
    cosine = np.divide(q[:,0], np.linalg.norm(q,axis=1),out=np.zeros(len(q)),where=np.linalg.norm(q,axis=1)>1e-12)
    t = np.clip(cosine,0,1); w = t*t*(3-2*t)
    u = (x-b['knots'][0])/np.ptp(b['knots'])
    start,end = cal['usable_start'],cal['usable_end']
    def smooth(t):
        t=np.clip(t,0,1);return t*t*(3-2*t)
    longitudinal = smooth((u-start)/.08)*smooth((end-u)/.08)
    factor = 1+(palm-1)*w*longitudinal
    q[:,0] *= radial*np.sqrt(a['ovality'])*factor
    q[:,1] *= radial/np.sqrt(a['ovality'])
    out = v.copy()
    anchor = b['knots'][0] if cal['thumb_end']==-1 else b['knots'][-1]
    out[:,0] = anchor+(x-anchor)*length
    out[:,1:] = centre + q @ side.T
    if not np.isfinite(out).all() or np.abs(out).max() > 1e7: raise ValueError('These measurement ratios produce an unusable model; check the reference and target dimensions')
    return out, dict(length_scale=length,wrap_scale=wrap,radial_scale=radial,palm_relative_scale=palm,
        usable_grip_length_mm=(end-start)*np.ptp(b['knots'])*length)


def deform(cal, pro):
    b = baseline()
    vertices, factors = warp_points(b['vertices'],cal,pro,b)
    return trimesh.Trimesh(vertices,b['faces'],process=False), factors
