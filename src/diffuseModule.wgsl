struct DiffuseUniforms {
  N: u32,
  diff: f32,
  dt: f32,
};

@group(0) @binding(0) var<uniform> uniforms: DiffuseUniforms;
@group(0) @binding(1) var densityReadTex: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var densityWriteTex: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var densityPrevTex: texture_storage_2d<r32float, read>;

@compute @workgroup_size(16, 16) fn diffuseStep(
  @builtin(global_invocation_id) id: vec3<u32>,
) {
  if (all(id.xy >= vec2u(1)) && all(id.xy <= vec2u(uniforms.N))) {
    let a = uniforms.dt * uniforms.diff * f32(uniforms.N) * f32(uniforms.N);

    let value = (textureLoad(densityPrevTex, id.xy) + 
      a * (
        textureLoad(densityReadTex, vec2u(id.x - 1, id.y)) +
        textureLoad(densityReadTex, vec2u(id.x + 1, id.y)) +
        textureLoad(densityReadTex, vec2u(id.x, id.y - 1)) +
        textureLoad(densityReadTex, vec2u(id.x, id.y + 1))
      )) / (1 + 4 * a);
    textureStore(densityWriteTex, id.xy, vec4f(value));
  }
}
