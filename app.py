"""Local-only web application. No model or measurement data leaves this process."""
import io, json, zipfile, hashlib, threading
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import trimesh
from flask import Flask, request, jsonify, send_file
import grip
import hand

app = Flask(__name__,static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 2*1024*1024
lock = threading.RLock()
CAL = grip.ROOT/'data/calibration.json'
cache = {}

def saved_cal():
    if not CAL.exists(): raise ValueError('Complete baseline setup before personalizing the grip')
    return grip.calibration(json.loads(CAL.read_text()))

def serialize(mesh):
    return dict(vertices=np.asarray(mesh.vertices).ravel().tolist(), faces=np.asarray(mesh.faces).ravel().tolist())

@app.before_request
def local_only():
    if request.host.split(':')[0] not in ('localhost','127.0.0.1','[::1]'): return jsonify(error='Local access only'),403
    if request.method=='POST' and request.headers.get('Origin') not in (None,'http://'+request.host): return jsonify(error='Origin mismatch'),403

@app.errorhandler(ValueError)
def invalid(exc): return jsonify(error=str(exc)),400

@app.errorhandler(400)
def bad_request(exc): return jsonify(error='Invalid request data'),400

@app.route('/')
def index(): return app.send_static_file('index.html')

@app.get('/api/state')
def state():
    b=grip.baseline()
    report=json.loads((grip.ROOT/'data/preparation.json').read_text())
    cal=json.loads(CAL.read_text()) if CAL.exists() else None
    return jsonify(preparation=report,calibration=cal,hand=serialize(hand.render_mesh(cal)) if cal else None,
      baseline=serialize(trimesh.Trimesh(b['vertices'],b['faces'],process=False)),
      original=serialize(trimesh.Trimesh(b['original_vertices'],b['original_faces'],process=False)),
      repair_faces=b['repair_faces'].tolist(), centreline=dict(knots=b['knots'].tolist(),centres=b['centres'].tolist()))

@app.post('/api/calibration')
def save_cal():
    cal=grip.calibration(request.get_json())
    if CAL.exists():
        previous=saved_cal()
        for key in ('thumb_end','palm_angle','usable_start','usable_end'):
            if cal[key]!=previous[key]: raise ValueError('The existing imprint orientation and contact region are locked for this scan.')
    temp=CAL.with_suffix('.tmp'); temp.write_text(json.dumps(cal,indent=2)); temp.replace(CAL)
    cache.clear()
    return jsonify(calibration=cal,hand=serialize(hand.render_mesh(cal)))

@app.get('/api/calibration/download')
def download_cal():
    saved_cal()
    return send_file(CAL,as_attachment=True,download_name='grip-baseline-calibration.json')

def generate(data):
    cal=saved_cal(); pro=grip.profile(data)
    if data.get('source_sha256') and data['source_sha256']!=cal['source_sha256']: raise ValueError('Profile belongs to a different source model')
    if data.get('calibration') and data['calibration']!=cal: raise ValueError('This profile uses a different baseline calibration. Restore its calibration in Setup before generating.')
    key=hashlib.sha256(json.dumps([cal,pro],sort_keys=True).encode()).hexdigest()
    with lock:
        if key not in cache:
            mesh,factors=grip.deform(cal,pro)
            checks=grip.validate(mesh)
            b=grip.baseline(); base_mesh=trimesh.Trimesh(b['vertices'],b['faces'],process=False)
            report=dict(name=pro['name'],source=grip.SOURCE.name,source_sha256=cal['source_sha256'],algorithm=grip.ALGORITHM,
                fit_status=pro['fit_status'],dimensions_mm=mesh.extents.tolist(),
                baseline_dimensions_mm=base_mesh.extents.tolist(),dimension_changes_mm=(mesh.extents-base_mesh.extents).tolist(),
                factors=factors,validation=checks,calibration=cal,profile=pro)
            cache.clear(); cache[key]=(mesh,report)
        return cache[key]

@app.post('/api/preview')
def preview():
    mesh,report=generate(request.get_json())
    return jsonify(mesh=serialize(mesh),hand=serialize(hand.render_mesh(report['calibration'],report['profile'])),report=report)

@app.post('/api/export')
def export():
    mesh,report=generate(request.get_json())
    if not report['validation']['valid']: return jsonify(error='Export blocked: resolve the highlighted mesh integrity issues first',validation=report['validation']),422
    stl_bytes=mesh.export(file_type='stl')
    with lock:
        written=trimesh.load(io.BytesIO(stl_bytes),file_type='stl',force='mesh',process=True)
        export_check=grip.validate(written)
    if not export_check['valid']: return jsonify(error='Export blocked: STL precision introduced a mesh defect. Adjust the shape and regenerate.',validation=export_check),422
    report=dict(report,export_validation=export_check)
    pro=report['profile']|dict(source_sha256=report['source_sha256'],calibration=report['calibration'])
    lines=['PERSONALIZED PRYBAR GRIP',f"Profile: {pro['name']}",f"Source: {grip.SOURCE.name}",'Units: millimetres',
        f"Fit status: {pro['fit_status']}",'Dimensions (local axes): '+', '.join(f'{v:.2f}' for v in report['dimensions_mm']),
        'Dimension changes: '+', '.join(f'{v:+.2f}' for v in report['dimension_changes_mm']),
        'Geometry checks: closed, consistent normals, positive volume, no degenerate faces or detected self-intersections.',
        'Mapping: hand breadth -> axial length; hand length minus palm length -> finger-wrap scaling; palm length -> palm surface adjustment.',
        'These are experimental design mappings, not validated ANSUR fit equations.',
        'Physical fit checklist: finger closure; thumb placement; pressure hotspots; slipping; comfort; compare baseline and personalized variants.',
        'Test with the reference user and at least two users with different hand proportions. Record adjustments and results.',
        'Mesh checks do not establish load-bearing suitability.']
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('personalized-grip.stl',stl_bytes)
        z.writestr('profile.json',json.dumps(pro,indent=2))
        z.writestr('baseline-calibration.json',json.dumps(report['calibration'],indent=2))
        z.writestr('report.json',json.dumps(report,indent=2))
        z.writestr('report.txt','\n'.join(lines))
    stream.seek(0)
    return send_file(stream,mimetype='application/zip',as_attachment=True,download_name='personalized-grip.zip')

if __name__=='__main__':
    grip.prepare()
    print('Open http://127.0.0.1:8765 — keep this terminal open.',flush=True)
    app.run(host='127.0.0.1',port=8765,debug=False,threaded=False)
