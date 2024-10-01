import { useCallback, useMemo } from "react";
import { useAnimationFrame } from "./useAnimationFrame";
import renderModuleCode from "./renderModule.wgsl?raw";

const N = 256;

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
  4. copy buffer to a texture
  5. sample the texture in shader
  */

  const densityBuffer = useMemo(() => {
    const buffer = device.createBuffer({
      label: "Density buffer",
      usage:
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.STORAGE,
      size: 4 * N * N,
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
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }, [device]);

  const densityTexture = useMemo(
    () =>
      device.createTexture({
        label: "Density display texture",
        format: "r32float",

        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        size: [N, N],
      }),
    [device]
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

  const densitySampler = useMemo(() => device.createSampler(), [device]);

  const bindGroup = useMemo(
    () =>
      device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: densitySampler },
          { binding: 1, resource: densityTexture.createView() },
        ],
      }),
    [densitySampler, densityTexture, device, renderPipeline]
  );

  useAnimationFrame(
    useCallback(() => {
      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const encoder = device.createCommandEncoder({
        label: "Field display encoder",
      });

      encoder.copyBufferToTexture(
        { buffer: densityBuffer, bytesPerRow: 4 * N },
        { texture: densityTexture },
        [N, N]
      );

      const renderPass = encoder.beginRenderPass(renderPassDescriptor);
      renderPass.setPipeline(renderPipeline);
      renderPass.setBindGroup(0, bindGroup);
      renderPass.draw(6);
      renderPass.end();

      device.queue.submit([encoder.finish()]);
    }, [
      bindGroup,
      context,
      densityBuffer,
      densityTexture,
      device,
      renderPassDescriptor,
      renderPipeline,
    ])
  );

  return null;
}
