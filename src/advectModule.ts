import { wgsl } from "./wgsl";

export function advectModuleCode({
  texFormat,
  workgroupDim,
}: {
  texFormat: GPUTextureFormat;
  workgroupDim: number;
}): string {
  return wgsl/*wgsl*/ `
    struct AdvectUniforms {
      N: u32,
      dt: f32,
    };

    @group(0) @binding(0) var<uniform> uniforms: AdvectUniforms;
    @group(0) @binding(1) var readTex: texture_2d<f32>;
    @group(0) @binding(2) var readSampler: sampler;
    @group(0) @binding(3) var writeTex: texture_storage_2d<${texFormat}, write>;
    @group(0) @binding(4) var velocityTex: texture_2d<f32>;

    @compute @workgroup_size(${workgroupDim}, ${workgroupDim}) fn advect(
      @builtin(global_invocation_id) id: vec3u,
    ) {
      if (all(id.xy >= vec2u(1)) && all(id.xy <= vec2u(uniforms.N))) {
        // advectStam(id);
        advectMacCormack(id);
      }
    }

    fn advectStam(
      id: vec3u,
    ) {
      let dt0 = vec2f(uniforms.dt * f32(uniforms.N));
      let velocity = textureLoad(velocityTex, id.xy, 0).xy;
      let backPos = clamp(
        vec2f(id.xy) - dt0 * velocity,
        vec2f(0.5),
        vec2f(f32(uniforms.N)) + vec2f(0.5),
      );
      
      let texSize = vec2f(textureDimensions(readTex));
      let samplePos = (backPos + 0.5) / texSize;
      let value = textureSampleLevel(readTex, readSampler, samplePos, 0);
      
      textureStore(writeTex, id.xy, value);
    }
    
    fn advectMacCormack(
      id: vec3u,
    ) {
      let texSize = vec2f(textureDimensions(readTex));

      let dt0 = vec2f(uniforms.dt * f32(uniforms.N));
      let velocity = textureLoad(velocityTex, id.xy, 0).xy;
      let backwardPos = clamp(
        vec2f(id.xy) - dt0 * velocity,
        vec2f(0.5),
        vec2f(f32(uniforms.N)) + vec2f(0.5),
      );
      let backwardSamplePos = (backwardPos + 0.5) / texSize;
      let backwardValue = textureSampleLevel(readTex, readSampler, backwardSamplePos, 0);

      let forwardPos = clamp(
        vec2f(id.xy) + dt0 * velocity,
        vec2f(0.5),
        vec2f(f32(uniforms.N)) + vec2f(0.5),
      );
      let forwardSamplePos = (forwardPos + 0.5) / texSize;
      let forwardVelocity = textureSampleLevel(velocityTex, readSampler, forwardSamplePos, 0).xy;

      let forwardBackPos = clamp(
        forwardPos - dt0 * forwardVelocity,
        vec2f(0.5),
        vec2f(f32(uniforms.N)) + vec2f(0.5),
      );
      let forwardBackSamplePos = (forwardBackPos + 0.5) / texSize;
      let forwardBackValue = textureSampleLevel(readTex, readSampler, forwardBackSamplePos, 0);

      let error = forwardBackValue - backwardValue;
      let value = backwardValue + error * 0.5;

      textureStore(writeTex, id.xy, value);
    }
  `;
}
