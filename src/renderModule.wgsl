struct Uniforms {
  N: i32,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> densityBuffer: array<f32>;

@vertex fn vs(
  @builtin(vertex_index) vertexIndex : u32
) -> VertexOutput {
  // full-screen quad
  let pos = array(
    vec2f(-1, -1),
    vec2f(1, -1),
    vec2f(-1, 1),

    vec2f(-1, 1),
    vec2f(1, -1),
    vec2f(1, 1)
  );

  let uv = array(
    vec2f(0, 1),
    vec2f(1, 1),
    vec2f(0, 0),

    vec2f(0, 0),
    vec2f(1, 1),
    vec2f(1, 0)
  );
  
  var output: VertexOutput;
  output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
  output.uv = uv[vertexIndex];
  return output;
}

@fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
  let x = i32(floor(input.uv.x * f32(uniforms.N)));
  let y = i32(floor(input.uv.y * f32(uniforms.N)));
  let index = y * uniforms.N + x;
  
  return vec4(densityBuffer[index], 0, 0, 1);
}
