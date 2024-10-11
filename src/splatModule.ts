import { wgsl } from "./wgsl";

export function splatModuleCode({
  workgroupDim,
}: {
  workgroupDim: number;
}): string {
  return wgsl/*wgsl*/ `
    struct SplatUniforms {
      N: u32,
      dt: f32,
      position: vec2f,
      velocity: vec2f,
      radius: f32,
      amount: f32,
    };

    @group(0) @binding(0) var<uniform> uniforms: SplatUniforms;
    @group(0) @binding(1) var densityReadTex: texture_2d<f32>;
    @group(0) @binding(2) var densityWriteTex: texture_storage_2d<r32float, write>;
    @group(0) @binding(3) var velocityReadTex: texture_2d<f32>;
    @group(0) @binding(4) var velocityWriteTex: texture_storage_2d<rg32float, write>;

    @compute @workgroup_size(${workgroupDim}, ${workgroupDim}) fn splat(
      @builtin(global_invocation_id) id: vec3u,
    ) {
      if (uniforms.radius == 0) {
        return;
      }
      if (!(all(id.xy >= vec2u(1)) && all(id.xy <= vec2u(uniforms.N)))) {
        return;
      }

      var density = textureLoad(densityReadTex, id.xy, 0).x;
      var velocity = textureLoad(velocityReadTex, id.xy, 0).xy;
      
      let dx = vec2f(id.xy) - uniforms.position;
      let dist = length(dx);
      if (dist < uniforms.radius) {
        let f = 1.0 - dist / uniforms.radius;
        
        density += f * uniforms.amount;
        velocity += f * uniforms.velocity;
      }

      textureStore(densityWriteTex, id.xy, vec4(density, 0, 0, 0));
      textureStore(velocityWriteTex, id.xy, vec4(velocity, 0, 0));
    }
  `;
}
