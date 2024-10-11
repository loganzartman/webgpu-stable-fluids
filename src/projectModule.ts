import { wgsl } from "./wgsl";

export function projectModuleCode({
  workgroupDim,
}: {
  workgroupDim: number;
}): string {
  return wgsl/*wgsl*/ `
    struct ProjectUniforms {
      N: u32,
      dt: f32,
    };

    @group(0) @binding(0) var<uniform> uniforms: ProjectUniforms;
    @group(0) @binding(1) var velReadTex: texture_2d<f32>;
    @group(0) @binding(2) var velWriteTex: texture_storage_2d<rg32float, write>;
    @group(0) @binding(3) var velSampler: sampler;
    @group(0) @binding(4) var divReadTex: texture_2d<f32>;
    @group(0) @binding(5) var divWriteTex: texture_storage_2d<r32float, write>;
    @group(0) @binding(6) var presReadTex: texture_2d<f32>;
    @group(0) @binding(7) var presWriteTex: texture_storage_2d<r32float, write>;

    @compute @workgroup_size(${workgroupDim}, ${workgroupDim}) fn projectInit(
      @builtin(global_invocation_id) id: vec3u,
    ) {
      _ = velWriteTex;
      _ = divReadTex;
      _ = presReadTex;

      if (!(all(id.xy >= vec2u(1)) && all(id.xy <= vec2u(uniforms.N)))) {
        return;
      }

      let h = vec2f(1) / vec2f(f32(uniforms.N));
      
      let texSize = vec2f(textureDimensions(velReadTex));
      let samplePos = (vec2f(id.xy) + 0.5) / texSize;
      let ddx = 1.0 / texSize;
      let gx = textureSampleGrad(velReadTex, velSampler, samplePos, vec2f(1, 0) * ddx, vec2f(0));
      let gy = textureSampleGrad(velReadTex, velSampler, samplePos, vec2f(0), vec2f(0, 1) * ddx);
      let divergence = -0.5 * (h.x * gx.x + h.y * gy.y);
      
      textureStore(divWriteTex, id.xy, vec4f(divergence, 0.5, 0, 0));
      textureStore(presWriteTex, id.xy, vec4f(0));
      
      // TODO: boundary
    }

    @compute @workgroup_size(${workgroupDim}, ${workgroupDim}) fn projectSolve(
      @builtin(global_invocation_id) id: vec3u,
    ) {
      _ = velReadTex;
      _ = velWriteTex;
      _ = velSampler;
      _ = divWriteTex;

      if (!(all(id.xy >= vec2u(1)) && all(id.xy <= vec2u(uniforms.N)))) {
        return;
      }
      
      let x = i32(id.x);
      let y = i32(id.y);
      let divergence = textureLoad(divReadTex, id.xy, 0).r;
      let pressure = (
        divergence +
        textureLoad(presReadTex, vec2i(x - 1, y), 0) + 
        textureLoad(presReadTex, vec2i(x + 1, y), 0) + 
        textureLoad(presReadTex, vec2i(x, y - 1), 0) + 
        textureLoad(presReadTex, vec2i(x, y + 1), 0)
      ) / 4.0;
      
      textureStore(presWriteTex, id.xy, pressure);
    }
    
    @compute @workgroup_size(${workgroupDim}, ${workgroupDim}) fn projectApply(
      @builtin(global_invocation_id) id: vec3u,
    ) {}
  `;
}
