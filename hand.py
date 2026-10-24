"""Illustrative gripping hand. A display aid, not a scan or contact simulation.
Build once in reference coordinates; apply exactly the grip's deformation field.
The hand never participates in imprint generation or STL exports.
"""
import json
from functools import lru_cache
import numpy as np
import trimesh
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from skimage.measure import marching_cubes
import grip

@lru_cache(maxsize=8)
def _reference(key):
    cal=json.loads(key); b=grip.baseline(); ref=cal['reference']
    scan=trimesh.Trimesh(b['vertices'],b['faces'],process=False)
    curve=CubicSpline(b['knots'],b['centres'],axis=0)
    theta=np.deg2rad(cal['palm_angle']); n=np.array([0,np.cos(theta),np.sin(theta)])
    tangent=np.array([0,-np.sin(theta),np.cos(theta)])
    xaxis=np.array([1,0,0]); frame=np.column_stack([xaxis,n,tangent])
    lo,hi=b['knots'][0],b['knots'][-1]
    a=lo+(hi-lo)*cal['usable_start']; z=lo+(hi-lo)*cal['usable_end']; mid=(a+z)/2
    width=min(ref['hand_breadth'],(z-a)*.95)
    thumb_sign=cal['thumb_end']; wrap=ref['hand_length']-ref['palm_length']
    def centres(xs): return np.column_stack([xs,curve(xs)])
    def contact(xs,angles,clearance):
        xs=np.array(xs);angles=np.array(angles)
        origin=centres(xs); direction=np.cos(angles)[:,None]*n+np.sin(angles)[:,None]*tangent
        points,rays,_=scan.ray.intersects_location(origin,direction,multiple_hits=True)
        distances=np.full(len(xs),np.inf)
        if len(points): np.minimum.at(distances,rays,np.linalg.norm(points-origin[rays],axis=1))
        distances[~np.isfinite(distances)]=np.mean(scan.extents[1:])/2
        distances=gaussian_filter1d(distances,1)
        return origin+direction*(distances+clearance)[:,None]
    # A continuous implicit palm/wrist replaces the overlapping ellipsoid blobs.
    palm_shift=tangent*6.0
    palm_center=contact([mid],[0],8)[0]+palm_shift
    strokes=[]
    def tube(points,radii):
        strokes.append(((points-palm_center)@frame,np.asarray(radii)))
    # Index through little finger, ordered away from the thumb end.
    # Traced contact outlines: a short index pad, deeper ring pad, and
    # individual distal sweeps. Palm, thumb and knuckle roots stay fixed.
    for i,(offset,length_ratio,radius_ratio) in enumerate(zip([.38,.12,-.10,-.36],[.94,1,.93,.76],[.093,.099,.092,.077])):
        x=mid+thumb_sign*width*offset
        radius=width*radius_ratio
        end=.75+wrap*length_ratio/(np.mean(scan.extents[1:])/2+radius)
        u=np.linspace(0,1,32)
        curl_limit=[3.45,3.50,3.48,3.24][i]
        tip_adjustment=[-.53,.20,.20,-.20][i]
        angles=-np.linspace(.75,min(end+.12,curl_limit)+tip_adjustment,len(u))
        # Preserve knuckle attachment while varying distal sweep by digit.
        distal=np.clip((u-.25)/.75,0,1)
        sweep=distal*distal*(3-2*distal)
        slant=[.04,.115,.075,.08][i]
        xs=x-thumb_sign*width*slant*sweep
        path=contact(xs,angles,radius*(.94-.20*sweep))
        root_blend=np.clip(1-u/.30,0,1)
        path+=palm_shift*(root_blend*root_blend*(3-2*root_blend))[:,None]
        radii=radius*(1-.34*u+.065*np.exp(-((u-.32)/.065)**2)+.045*np.exp(-((u-.66)/.06)**2))
        tube(path,radii)
    # Marked reference: the thumb pad lies on the broad positive-tangent face,
    # below the rim. The four fingers curl from the opposite face.
    u=np.linspace(0,1,32)
    # Seat the elongated thumb in the marked groove, below the upper rim.
    angles=.20+1.18*(u*u*(3-2*u))
    xs=mid+thumb_sign*width*(-.10+.49*u)
    thumb_radius=width*.104
    thumb_path=contact(xs,angles,thumb_radius)
    # Keep the base connected to the lowered palm.
    base_blend=np.clip(1-u/.65,0,1)
    thumb_path+=palm_shift*(base_blend*base_blend*(3-2*base_blend))[:,None]
    tube(thumb_path,np.linspace(thumb_radius,thumb_radius*.69,len(angles)))
    # Smoothly join palm, tapered wrist, thumb and fingers on a 0.9 mm field.
    pitch=.9
    length=ref['palm_length']
    stations=np.array([-.70,-.52,-.36,-.14,.10,.28,.36])*length
    widths=width*np.array([.25,.245,.29,.44,.49,.46,.37])
    depths=np.array([9.0,9.2,10.5,11.0,10.0,8.5,5.0])*width/90
    centre_y=np.array([2,1,0,0,-1,-2,-3])*width/90
    # Sweep the heel and wrist toward the little-finger side without
    # moving the knuckle edge or any digit contact paths.
    centre_x=-thumb_sign*width*np.array([.58,.48,.32,.12,0,0,0])
    allpoints=np.concatenate([p for p,r in strokes])
    lower=np.minimum(allpoints.min(0)-14,[-width*.95,-15,-stations[-1]-5])
    upper=np.maximum(allpoints.max(0)+14,[width*.95,16,-stations[0]+5])
    axes=[np.arange(lower[i],upper[i]+pitch,pitch) for i in range(3)]
    xx,yy,zz=np.meshgrid(*axes,indexing='ij')
    # The knuckle edge follows the fingers on the negative-tangent side;
    # the wrist and thumb base emerge on the positive-tangent side.
    longitudinal=-zz
    halfwidth=np.interp(longitudinal,stations,widths)
    halfdepth=np.interp(longitudinal,stations,depths)
    cy=np.interp(longitudinal,stations,centre_y)
    cx=PchipInterpolator(stations,centre_x)(np.clip(longitudinal,stations[0],stations[-1]))
    # Rounded rectangular cross-sections give a flatter palm and a cuff-shaped wrist.
    radial=((np.abs((xx-cx)/halfwidth)**2.5+np.abs((yy-cy)/halfdepth)**2.5)**.4-1)*halfdepth
    axial=np.maximum(stations[0]-longitudinal,longitudinal-stations[-1])
    field=np.maximum(radial,axial).astype(np.float32)
    del xx,yy,zz
    blend=2.4
    for points,radii in strokes:
        for a,b,ra,rb in zip(points[:-1],points[1:],radii[:-1],radii[1:]):
            low=np.maximum(0,np.floor((np.minimum(a,b)-max(ra,rb)-blend-lower)/pitch).astype(int))
            high=np.minimum(field.shape,np.ceil((np.maximum(a,b)+max(ra,rb)+blend-lower)/pitch).astype(int)+1)
            box=tuple(slice(low[i],high[i]) for i in range(3))
            grid=np.stack(np.meshgrid(*[axes[i][box[i]] for i in range(3)],indexing='ij'),axis=-1)
            delta=b-a; u=np.clip(((grid-a)@delta)/(delta@delta),0,1)
            distance=np.linalg.norm(grid-a-u[...,None]*delta,axis=-1)-(ra+(rb-ra)*u)
            old=field[box];h=np.maximum(blend-np.abs(old-distance),0)/blend
            field[box]=np.minimum(old,distance)-h*h*blend*.25
    verts,faces,_,_=marching_cubes(field,level=0,spacing=(pitch,pitch,pitch))
    verts=(verts+lower)@frame.T+palm_center
    mesh=trimesh.Trimesh(verts,faces,process=True)
    trimesh.repair.fix_normals(mesh)
    trimesh.smoothing.filter_taubin(mesh,iterations=6)
    return mesh

def render_mesh(cal,pro=None):
    key=json.dumps(cal,sort_keys=True)
    reference=_reference(key)
    vertices=reference.vertices.copy()
    if pro is not None: vertices,_=grip.warp_points(vertices,cal,pro)
    return trimesh.Trimesh(vertices,reference.faces,process=False)
