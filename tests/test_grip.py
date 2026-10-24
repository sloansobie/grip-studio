import copy, io, json, zipfile
import numpy as np
import pytest
import trimesh
import grip
import app as web

@pytest.fixture(scope='session',autouse=True)
def prepared(): grip.prepare()

@pytest.fixture
def cal():
    return grip.calibration(dict(reference=dict(hand_length=190,hand_breadth=90,palm_length=105),thumb_end=-1,
        palm_angle=0,usable_start=.1,usable_end=.9,accepted_repairs=True,landmarks_confirmed=True,source_sha256=grip.digest(grip.SOURCE)))

@pytest.fixture
def pro(cal): return grip.profile(dict(name='TEST ONLY',target=cal['reference'],adjustments={}))

def test_repaired_baseline():
    b=grip.baseline();m=trimesh.Trimesh(b['vertices'],b['faces'],process=False)
    assert grip.validate(m)['valid']
    report=json.loads((grip.ROOT/'data/preparation.json').read_text())
    assert report['source']=='smoothed_mergedhandle_V2.stl'
    assert report['after']['intersecting_faces']==0
    assert len(b['repair_faces'])>0

def test_identity(cal,pro):
    m,_=grip.deform(cal,pro)
    assert np.max(np.linalg.norm(m.vertices-grip.baseline()['vertices'],axis=1))<.01

def test_length_only(cal,pro):
    b=grip.baseline()['vertices'];pro['target']['hand_breadth']*=1.1
    m,f=grip.deform(cal,pro)
    assert np.isclose(m.extents[0]/np.ptp(b[:,0]),1.1)
    assert np.allclose(m.vertices[:,1:],b[:,1:])
    assert grip.validate(m)['valid']

def test_wrap_and_local_palm(cal,pro):
    b=grip.baseline();v=b['vertices'];c=grip.CubicSpline(b['knots'],b['centres'],axis=0)(v[:,0])
    pro['target']['hand_length']+=10
    m,f=grip.deform(cal,pro)
    assert np.allclose(m.vertices[:,0],v[:,0])
    assert f['radial_scale']>1
    # Increase palm and hand equally to keep finger-wrap fixed: finger surface stays fixed.
    pro['target']=dict(hand_length=200,hand_breadth=90,palm_length=115)
    m,f=grip.deform(cal,pro)
    finger=v[:,1]<c[:,0]
    assert np.allclose(m.vertices[finger],v[finger])
    assert np.max(np.linalg.norm(m.vertices[~finger]-v[~finger],axis=1))>.1
    assert grip.validate(m)['valid']

def test_repeatability(cal,pro):
    pro['target']['hand_length']+=8
    a,_=grip.deform(cal,pro);b,_=grip.deform(json.loads(json.dumps(cal)),json.loads(json.dumps(pro)))
    assert np.array_equal(a.vertices,b.vertices)
    assert a.export(file_type='stl')==b.export(file_type='stl')

@pytest.mark.parametrize('data',[dict(hand_length=100,hand_breadth=80,palm_length=100),dict(hand_length=0,hand_breadth=80,palm_length=90),dict(hand_length=float('nan'),hand_breadth=80,palm_length=90),dict(hand_length=190,hand_breadth=-1,palm_length=100)])
def test_invalid_measurements(data):
    with pytest.raises(ValueError): grip.measurements(data)

def test_integrity_failures():
    box=trimesh.creation.box();box.update_faces(np.arange(len(box.faces))!=0)
    assert not grip.validate(box)['valid']
    overlapping=trimesh.util.concatenate([trimesh.creation.box(),trimesh.creation.box(transform=trimesh.transformations.translation_matrix([.3,.3,.3]))])
    check=grip.validate(overlapping)
    assert not check['valid'] and check['intersecting_faces']>0
    assert check['bad_faces']

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(web,'CAL',tmp_path/'calibration.json');web.cache.clear()
    with web.app.test_client() as c: yield c

def test_requires_baseline_and_review(client,cal,pro):
    assert client.post('/api/preview',json=pro).status_code==400
    bad=copy.deepcopy(cal);bad['accepted_repairs']=False
    assert client.post('/api/calibration',json=bad).status_code==400
    assert client.post('/api/calibration',json=cal).status_code==200

def test_full_export_and_restore(client,cal,pro):
    assert client.post('/api/calibration',json=cal).status_code==200
    preview=client.post('/api/preview',json=pro).get_json()
    assert preview['report']['validation']['valid']
    res=client.post('/api/export',json=pro)
    assert res.status_code==200
    z=zipfile.ZipFile(io.BytesIO(res.data))
    assert set(z.namelist())=={'personalized-grip.stl','profile.json','baseline-calibration.json','report.json','report.txt'}
    m=trimesh.load(io.BytesIO(z.read('personalized-grip.stl')),file_type='stl',force='mesh')
    assert grip.validate(m)['valid']
    restored=json.loads(z.read('profile.json'))
    rerun=client.post('/api/preview',json=restored).get_json()
    assert rerun['mesh']['vertices']==preview['mesh']['vertices']
    restored['calibration']['palm_angle']=180
    assert client.post('/api/preview',json=restored).status_code==400

def test_export_blocked(client,cal,pro,monkeypatch):
    client.post('/api/calibration',json=cal)
    real=grip.validate
    def bad(m):
        out=real(m);out['valid']=False;out['intersecting_faces']=2;return out
    monkeypatch.setattr(grip,'validate',bad)
    assert client.post('/api/export',json=pro).status_code==422

def test_local_guard(client):
    assert client.get('/api/state',headers={'Host':'attacker.example'}).status_code==403
    assert client.post('/api/calibration',headers={'Origin':'https://example.com'},json={}).status_code==403


def test_existing_imprint_locked(client,cal,pro):
    client.post('/api/calibration',json=cal)
    changed=copy.deepcopy(cal);changed['palm_angle']=90
    response=client.post('/api/calibration',json=changed)
    assert response.status_code==400 and 'locked' in response.get_json()['error']
    pro['adjustments']['palm_fullness']=1.2
    assert client.post('/api/preview',json=pro).status_code==400

def test_hand_follows_identical_field(cal,pro):
    import hand
    before=hand.render_mesh(cal)
    same=hand.render_mesh(cal,pro)
    assert np.allclose(before.vertices,same.vertices,atol=1e-10)
    pro['target']['hand_breadth']*=1.1
    pro['target']['hand_length']+=10
    after=hand.render_mesh(cal,pro)
    expected,_=grip.warp_points(before.vertices,cal,pro)
    assert np.array_equal(after.faces,before.faces)
    assert np.allclose(after.vertices,expected)
    assert np.isfinite(after.vertices).all()
    mesh,_=grip.deform(cal,pro)
    assert np.array_equal(mesh.faces,grip.baseline()['faces'])

def test_preview_hand_is_separate_from_export(client,cal,pro):
    client.post('/api/calibration',json=cal)
    preview=client.post('/api/preview',json=pro).get_json()
    assert len(preview['hand']['vertices'])>0
    res=client.post('/api/export',json=pro)
    z=zipfile.ZipFile(io.BytesIO(res.data))
    m=trimesh.load(io.BytesIO(z.read('personalized-grip.stl')),file_type='stl')
    assert len(m.faces)==len(grip.baseline()['faces'])


def test_hand_pose_does_not_change_grip(client,cal,pro):
    client.post('/api/calibration',json=cal)
    original=client.post('/api/export',json=pro)
    pro['hand_pose']=dict(x=20,y=-10,z=5,rx=90,ry=30,rz=-12,palm_angle=45)
    moved=client.post('/api/export',json=pro)
    a=zipfile.ZipFile(io.BytesIO(original.data));b=zipfile.ZipFile(io.BytesIO(moved.data))
    assert a.read('personalized-grip.stl')==b.read('personalized-grip.stl')
    assert json.loads(b.read('profile.json'))['hand_pose']==pro['hand_pose']
    pro['hand_pose']['x']=float('nan')
    assert client.post('/api/preview',json=pro).status_code==400

def test_improved_hand_is_one_closed_surface(cal):
    import hand
    mesh=hand.render_mesh(cal)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert len(mesh.split(only_watertight=False))==1
