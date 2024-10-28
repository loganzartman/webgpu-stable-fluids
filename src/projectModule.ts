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
    @group(0) @binding(3) var divReadTex: texture_2d<f32>;
    @group(0) @binding(4) var divWriteTex: texture_storage_2d<r32float, write>;
    @group(0) @binding(5) var presReadTex: texture_2d<f32>;
    @group(0) @binding(6) var presWriteTex: texture_storage_2d<r32float, write>;

    @compute @workgroup_size(${workgroupDim}, ${workgroupDim}) fn projectInit(
      @builtin(global_invocation_id) id: vec3u,
    ) {
      _ = velWriteTex;

      if (!(all(id.xy >= vec2u(1)) && all(id.xy <= vec2u(uniforms.N)))) {
        return;
      }

      let h = vec2f(1) / vec2f(f32(uniforms.N));
      
      let oldDivergence = textureLoad(divReadTex, id.xy, 0).r;

      let gx = 
        textureLoad(velReadTex, vec2u(id.x + 1, id.y), 0).x - 
        textureLoad(velReadTex, vec2u(id.x - 1, id.y), 0).x;
      let gy = 
        textureLoad(velReadTex, vec2u(id.x, id.y + 1), 0).y - 
        textureLoad(velReadTex, vec2u(id.x, id.y - 1), 0).y;
      let divergence = -0.5 * (h.x * gx + h.y * gy);
      
      let pressure = textureLoad(presReadTex, id.xy, 0).r;
      
      textureStore(divWriteTex, id.xy, vec4f(divergence, 0, 0, 0));
      textureStore(presWriteTex, id.xy, vec4f(pressure * 0.99, 0, 0, 0));
      // textureStore(presWriteTex, id.xy, vec4f(0));
      
      // TODO: boundary
    }

    @compute @workgroup_size(${workgroupDim}, ${workgroupDim}) fn projectSolve(
      @builtin(global_invocation_id) id: vec3u,
    ) {
      _ = velReadTex;
      _ = velWriteTex;
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
      
      textureStore(presWriteTex, id.xy, pressure * 0.99);
      
      // TODO: boundary
    }
    
    @compute @workgroup_size(${workgroupDim}, ${workgroupDim}) fn projectApply(
      @builtin(global_invocation_id) id: vec3u,
    ) {
      _ = divReadTex;
      _ = divWriteTex;
      _ = presWriteTex;

      if (!(all(id.xy >= vec2u(1)) && all(id.xy <= vec2u(uniforms.N)))) {
        return;
      }
      
      let h = vec2f(1) / vec2f(f32(uniforms.N));

      let gx = 
        textureLoad(presReadTex, vec2u(id.x + 1, id.y), 0).x - 
        textureLoad(presReadTex, vec2u(id.x - 1, id.y), 0).x;
      let gy = 
        textureLoad(presReadTex, vec2u(id.x, id.y + 1), 0).x - 
        textureLoad(presReadTex, vec2u(id.x, id.y - 1), 0).x;

      var velocity = textureLoad(velReadTex, id.xy, 0).xy;

      velocity -= 0.5 * vec2f(gx, gy) / h;

      textureStore(velWriteTex, id.xy, vec4f(velocity, 0, 0));
    }
  `;
}
