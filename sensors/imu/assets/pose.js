/* Proper body-to-world rotation plus model mounting and Z-up -> Y-up mapping. */
(function(root){
 const unit=q=>{if(!Array.isArray(q)||q.length!==4||!q.every(Number.isFinite))return null;const n=Math.hypot(...q);return n<.5||n>1.5?null:q.map(v=>v/n);};
 function multiply(a,b){const[w,x,y,z]=a,[v,i,j,k]=b;return[w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v];}
 const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 function fromBasis(r,u,f){
  const m00=r[0],m01=u[0],m02=f[0],m10=r[1],m11=u[1],m12=f[1],m20=r[2],m21=u[2],m22=f[2];
  const tr=m00+m11+m22;let w,x,y,z,s;
  if(tr>0){s=Math.sqrt(tr+1)*2;w=s/4;x=(m21-m12)/s;y=(m02-m20)/s;z=(m10-m01)/s;}
  else if(m00>m11&&m00>m22){s=Math.sqrt(1+m00-m11-m22)*2;w=(m21-m12)/s;x=s/4;y=(m01+m10)/s;z=(m02+m20)/s;}
  else if(m11>m22){s=Math.sqrt(1+m11-m00-m22)*2;w=(m02-m20)/s;x=(m01+m10)/s;y=s/4;z=(m12+m21)/s;}
  else{s=Math.sqrt(1+m22-m00-m11)*2;w=(m10-m01)/s;x=(m02+m20)/s;y=(m12+m21)/s;z=s/4;}
  return unit([w,x,y,z]);
 }
 function orientation(values,forward,up,headingOffset=0){
  const q=unit(values);if(!q||!forward||!up||Math.abs(forward.reduce((s,v,i)=>s+v*up[i],0))>.001)return null;
  const mount=fromBasis(cross(up,forward),up,forward);
  const t=Math.SQRT1_2,world=[t,-t,0,0];
  const angle=-headingOffset*Math.PI/360,offset=[Math.cos(angle),0,Math.sin(angle),0];
  return unit(multiply(offset,multiply(world,multiply(q,mount))));
 }
 const api={unit,multiply,orientation};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ImuPose=api;
})(typeof globalThis!=='undefined'?globalThis:this);
