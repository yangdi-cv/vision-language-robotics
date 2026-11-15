/* Port of PLAN-AN YbImuProtocol.kt: body axis -> world XY -> compass bearing. */
(function(root) {
  const axes={'+Z':[0,0,1],'-Z':[0,0,-1],'+X':[1,0,0],'-X':[-1,0,0],'+Y':[0,1,0],'-Y':[0,-1,0]};
  const normalize=value=>((value%360)+360)%360;
  function bearing(values,axis){
    if(!Array.isArray(values)||values.length!==4||!values.every(Number.isFinite)||!axes[axis])return null;
    const norm=Math.hypot(...values);
    if(norm<.5||norm>1.5)return null;
    const [w,x,y,z]=values.map(v=>v/norm);
    const [a,b,c]=axes[axis];
    const horizontalX=(1-2*(y*y+z*z))*a+2*(x*y-w*z)*b+2*(x*z+w*y)*c;
    const horizontalY=2*(x*y+w*z)*a+(1-2*(x*x+z*z))*b+2*(y*z-w*x)*c;
    if(Math.hypot(horizontalX,horizontalY)<.2)return null;
    return normalize(-Math.atan2(horizontalY,horizontalX)*180/Math.PI);
  }
  const api={axes,normalize,bearing};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.ImuHeading=api;
})(typeof globalThis!=='undefined'?globalThis:this);
