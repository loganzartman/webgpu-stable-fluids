import { wgsl } from "./wgsl";

export function diffuseModuleCode({
  workgroupDim,
}: {
  workgroupDim: number;
}): string {
  return wgsl/*wgsl*/ `
    struct DiffuseUniforms {
      N: u32,
      diff: f32,
      dt: f32,
    };

    @group(0) @binding(0) var<uniform> uniforms: DiffuseUniforms;
    @group(0) @binding(1) var readTex: texture_storage_2d<r32float, read>;
    @group(0) @binding(2) var writeTex: texture_storage_2d<r32float, write>;
    @group(0) @binding(3) var prevTex: texture_storage_2d<r32float, read>;

    @compute @workgroup_size(${workgroupDim}, ${workgroupDim}) fn diffuseStep(
      @builtin(global_invocation_id) id: vec3<u32>,
    ) {
      if (all(id.xy >= vec2u(1)) && all(id.xy <= vec2u(uniforms.N))) {
        let a = uniforms.dt * uniforms.diff * f32(uniforms.N) * f32(uniforms.N);

        let value = (textureLoad(prevTex, id.xy) + 
          a * (
            textureLoad(readTex, vec2u(id.x - 1, id.y)) +
            textureLoad(readTex, vec2u(id.x + 1, id.y)) +
            textureLoad(readTex, vec2u(id.x, id.y - 1)) +
            textureLoad(readTex, vec2u(id.x, id.y + 1))
          )) / (1.0 + 4.0 * a);
        textureStore(writeTex, id.xy, vec4f(value));
      }
    }
  `;
}
