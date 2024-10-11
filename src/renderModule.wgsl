struct Uniforms {
  N: u32,
  time: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var fieldTexture: texture_2d<f32>;
@group(0) @binding(2) var fieldSampler: sampler;

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
  _ = uniforms;
  let fieldRaw = textureSample(fieldTexture, fieldSampler, input.uv).rgb;
  var color = vec3f(0,0,0);
  if (any(fieldRaw != vec3f(0))) {
    color = abs(fieldRaw).rgb;
  }
  // density = max(0, density);
  // density = density / (density + 1);
  // let color = vec3(pow(density, 1.1), pow(density, 1.7), pow(density, 0.5));
  return vec4(color, 1);
}
