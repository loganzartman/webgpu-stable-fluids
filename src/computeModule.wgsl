struct Uniforms {
  N: u32,
  time: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> densityBuffer: array<f32>;

const k = u32(1103515245);

fn hash(x: vec3<u32>) -> vec3<f32> {
  var y = x;
  y = ((y>>vec3<u32>(8))^y.yzx)*k;
  y = ((y>>vec3<u32>(8))^y.yzx)*k;
  y = ((y>>vec3<u32>(8))^y.yzx)*k;
  return vec3<f32>(y)*(1.0/f32(u32(0xffffffff)));
}

@compute @workgroup_size(64) fn fillWithJunk(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let id = global_id.x;
  if (id < uniforms.N * uniforms.N) {
    densityBuffer[id] = hash(vec3<u32>(id, u32(uniforms.time * 1000.0), 0)).x;
  }
}
