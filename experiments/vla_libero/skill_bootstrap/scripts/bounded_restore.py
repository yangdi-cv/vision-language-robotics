"""Read existing Orbax OCDBT arrays one storage chunk at a time, directly to GPU."""
import hashlib,itertools,json,time,gc,functools
from pathlib import Path
import numpy as np
from chunk_math import merge_chunk


def restore(directory,adapter=None,progress=None):
    import tensorstore as ts
    import jax,jax.numpy as jnp
    from flax.traverse_util import unflatten_dict
    from orbax.checkpoint._src.serialization import tensorstore_utils
    directory=Path(directory);meta=json.loads((directory/'_METADATA').read_text())
    context=ts.Context({'cache_pool#ocdbt':{'total_bytes_limit':16*1024**2},'cache_pool':{'total_bytes_limit':32*1024**2},'data_copy_concurrency':{'limit':2},'file_io_concurrency':{'limit':2}})
    pairs={};seen=set()
    if adapter:
        with np.load(adapter) as f:
            assert f.files and all('lora' in k for k in f.files)
            for k in f.files:
                if not k.endswith('lora_a'):continue
                bkey=k[:-1]+'b';base=k[:-len('/lora_a')]+'/w' if k.endswith('/lora_a') else k[:-len('_lora_a')]
                pairs[base]=(f[k].copy(),f[bkey].copy())
            assert len(pairs)*2==len(f.files)
    @functools.partial(jax.jit,donate_argnums=(0,))
    def update(destination,block,offsets):
        return jax.lax.dynamic_update_slice(destination,block,offsets)
    flat={};fingerprints={};started=time.monotonic();max_source_bytes=0;max_output_bytes=0
    for entry in meta['tree_metadata'].values():
        keys=[x['key'] for x in entry['key_metadata']]
        if keys[0]!='params':continue
        logical=keys[1:-1] if keys[-1]=='value' else keys[1:];name='/'.join(logical)
        spec={'driver':'zarr','kvstore':tensorstore_utils.build_kvstore_tspec(str(directory),name='.'.join(keys),use_ocdbt=True),'fill_missing_data_reads':False,'recheck_cached_data':False,'recheck_cached_metadata':False}
        store=ts.open(spec,open=True,context=context).result();shape=tuple(store.shape);chunk=tuple(store.chunk_layout.read_chunk.shape)
        # Match Orbax's TensorStore cast during normal BF16 parameter restore.
        cast=ts.cast(store,'bfloat16');destination=jnp.empty(shape,dtype=jnp.bfloat16)
        ranges=[range(0,n,max(1,c)) for n,c in zip(shape,chunk,strict=True)]
        for offsets in itertools.product(*ranges):
            slices=tuple(slice(start,min(start+c,n)) for start,c,n in zip(offsets,chunk,shape,strict=True))
            block=np.ascontiguousarray(cast[slices].read().result())
            max_source_bytes=max(max_source_bytes,int(block.size*4));max_output_bytes=max(max_output_bytes,block.nbytes)
            if name in pairs:
                a,b=pairs[name];seen.add(name)
                if np.any(b):block=merge_chunk(block,a,b,slices)
            key=name+'@'+','.join(map(str,offsets));fingerprints[key]=hashlib.sha256(memoryview(block.reshape(-1).view(np.uint8))).hexdigest()
            destination=update(destination,jnp.asarray(block),tuple(offsets));destination.block_until_ready();del block
        flat[tuple(logical)]=destination;del store,cast,destination
        if progress:
            progress(dict(stage='reading_chunks',arrays=len(flat),latest=name,seconds=time.monotonic()-started,max_source_chunk_bytes=max_source_bytes,max_output_chunk_bytes=max_output_bytes))
    assert seen==set(pairs),(seen,set(pairs))
    gc.collect()
    return unflatten_dict(flat),dict(chunk_sha256=fingerprints,arrays=len(flat),chunks=len(fingerprints),max_source_chunk_bytes=max_source_bytes,max_output_chunk_bytes=max_output_bytes,seconds=time.monotonic()-started,adapter_weight_groups=len(pairs))
