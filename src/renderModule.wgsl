struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var densitySampler: sampler;
@group(0) @binding(1) var densityTexture: texture_2d<f32>;

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
  return textureSample(densityTexture, densitySampler, input.uv);
}
