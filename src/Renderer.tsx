import { useCallback, useMemo } from "react";
import { useAnimationFrame } from "./useAnimationFrame";
import renderModuleCode from "./renderModule.wgsl?raw";
import computeModuleCode from "./computeModule.wgsl?raw";
import { f32, u32, struct } from "typegpu/data";
import tgpu from "typegpu";

const N = 256;
const workgroupDim = 16;

export function Renderer({
  context,
  device,
}: {
  context: GPUCanvasContext;
  adapter: GPUAdapter;
  device: GPUDevice;
}) {
  const format = useMemo(() => {
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
    });
    return format;
  }, [context, device]);

  // the plan
  /*
  1. full screen quad
  2. uvs
  3. make a storage buffer & fill it with junk
  4. draw from the storage buffer
  */

  const Uniforms = useMemo(
    () =>
      struct({
        N: u32,
        time: f32,
      }),
    []
  );

  const uniformsBuffer = useMemo(
    () =>
      tgpu
        .createBuffer(Uniforms, {
          N,
          time: 0,
        })
        .$device(device)
        .$usage(tgpu.Uniform),
    [Uniforms, device]
  );

  const densityTexture = useMemo(() => {
    const texture = device.createTexture({
      label: "density storage texture",
      format: "r32float",
      size: [N, N],
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });

    const data = new Float32Array(N * N);
    for (let x = 0; x < N; ++x) {
      for (let y = 0; y < N; ++y) {
        const dx = x - N / 2;
        const dy = y - N / 2;
        const dist = Math.hypot(dx, dy);
        if (dist < 20) {
          data[y * N + x] = 1;
        }
      }
    }

    device.queue.writeTexture(
      { texture },
      data,
      {
        bytesPerRow: data.BYTES_PER_ELEMENT * N,
        rowsPerImage: N,
      },
      [N, N]
    );

    return texture;
  }, [device]);

  const densitySampler = useMemo(
    () =>
      device.createSampler({
        minFilter: "linear",
        magFilter: "linear",
      }),
    [device]
  );

  const computeModule = useMemo(
    () =>
      device.createShaderModule({
        label: "Compute module",
        code: computeModuleCode,
      }),
    [device]
  );

  const junkPipeline = useMemo(
    () =>
      device.createComputePipeline({
        label: "junk pipeline",
        layout: "auto",
        compute: {
          module: computeModule,
          entryPoint: "fillWithJunk",
        },
      }),
    [computeModule, device]
  );

  const junkBindGroup = useMemo(
    () =>
      device.createBindGroup({
        layout: junkPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: uniformsBuffer },
          { binding: 1, resource: densityTexture.createView() },
        ],
      }),
    [densityTexture, device, junkPipeline, uniformsBuffer]
  );

  const renderModule = useMemo(
    () =>
      device.createShaderModule({
        label: "Field display renderer",
        code: renderModuleCode,
      }),
    [device]
  );

  const renderPipeline = useMemo(
    () =>
      device.createRenderPipeline({
        label: "Field display pipeline",
        layout: "auto",
        vertex: {
          module: renderModule,
        },
        fragment: {
          module: renderModule,
          targets: [{ format }],
        },
      }),
    [device, format, renderModule]
  );

  const renderPassDescriptor = useMemo(
    () =>
      ({
        label: "Field display render pass",
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: [0.3, 0.3, 0.3, 1],
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      } satisfies GPURenderPassDescriptor),
    [context]
  );

  const renderBindGroup = useMemo(
    () =>
      device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: uniformsBuffer },
          { binding: 1, resource: densityTexture.createView() },
          { binding: 2, resource: densitySampler },
        ],
      }),
    [densitySampler, densityTexture, device, renderPipeline, uniformsBuffer]
  );

  useAnimationFrame(
    useCallback(() => {
      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const encoder = device.createCommandEncoder({
        label: "Field display encoder",
      });

      uniformsBuffer.write({ N, time: performance.now() });

      const junkPass = encoder.beginComputePass();
      junkPass.setPipeline(junkPipeline);
      junkPass.setBindGroup(0, junkBindGroup);
      junkPass.dispatchWorkgroups(
        Math.ceil(N / workgroupDim),
        Math.ceil(N / workgroupDim)
      );
      junkPass.end();

      const renderPass = encoder.beginRenderPass(renderPassDescriptor);
      renderPass.setPipeline(renderPipeline);
      renderPass.setBindGroup(0, renderBindGroup);
      renderPass.draw(6);
      renderPass.end();

      device.queue.submit([encoder.finish()]);
    }, [
      context,
      device,
      junkBindGroup,
      junkPipeline,
      renderBindGroup,
      renderPassDescriptor,
      renderPipeline,
      uniformsBuffer,
    ])
  );

  return null;
}
