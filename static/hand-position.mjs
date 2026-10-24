import {Vector3,MathUtils,Quaternion} from './vendor/three.module.js';
// Display-only rigid transform about the hand centre. Never touches mesh vertices.
export function placeHand(object,pose){
  const orbit=new Quaternion().setFromAxisAngle(new Vector3(1,0,0),MathUtils.degToRad(pose.palm_angle||0));
  const axis=object.userData.gripPivot||new Vector3();
  object.position.copy(object.userData.pivot).sub(axis).applyQuaternion(orbit).add(axis).add(new Vector3(pose.x,pose.y,pose.z));
  object.rotation.set(...['rx','ry','rz'].map(k=>MathUtils.degToRad(pose[k])));
  object.quaternion.premultiply(orbit);
  object.updateMatrixWorld(true);
}
