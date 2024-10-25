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
  var value = max(0, fieldRaw.r);
  value /= (value + 1);
  // let color = vec3(pow(value, 1.1), pow(value, 1.7), pow(value, 0.5));
  // let color = vec3(pow(value, 0.512), pow(value, 0.361), pow(value, 0.133));
  // let color = vec3(pow(value, 12), pow(value, 61), pow(value, 33));
  let color = hsl_to_rgb(value * 360.0, 0.6, clamp(value * 5.0, 0, 0.5));
  // let color = vec3(abs(fieldRaw));
  return vec4(color, 1);
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    if (s == 0.0) {
        return vec3<f32>(l, l, l);
    }

    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0));
    let m = l - c / 2.0;

    var r: f32;
    var g: f32;
    var b: f32;

    if (h < 60.0) {
        r = c; g = x; b = 0.0;
    } else if (h < 120.0) {
        r = x; g = c; b = 0.0;
    } else if (h < 180.0) {
        r = 0.0; g = c; b = x;
    } else if (h < 240.0) {
        r = 0.0; g = x; b = c;
    } else if (h < 300.0) {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }

    return vec3<f32>(r + m, g + m, b + m);
}