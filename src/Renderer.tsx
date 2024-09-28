import { useCallback, useMemo } from "react";
import redTriShaderCode from "./redTri.wgsl?raw";
import { useAnimationFrame } from "./useAnimationFrame";

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

  const redTriShader = useMemo(
    () =>
      device.createShaderModule({
        label: "our hardcoded red triangle shaders",
        code: redTriShaderCode,
      }),
    [device]
  );

  const pipeline = useMemo(
    () =>
      device.createRenderPipeline({
        label: "our hardcoded red triangle pipeline",
        layout: "auto",
        vertex: {
          entryPoint: "vs",
          module: redTriShader,
        },
        fragment: {
          entryPoint: "fs",
          module: redTriShader,
          targets: [{ format }],
        },
      }),
    [device, format, redTriShader]
  );

  const renderPassDescriptor = useMemo(
    () =>
      ({
        label: "basic render pass",
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: [0.5, 0.0, 0.5, 1.0],
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      } satisfies GPURenderPassDescriptor),
    [context]
  );

  const ubSize = 4 * (4 + 2 + 2);
  const ubColorOffset = 0;
  const ubScaleOffset = 4;
  const ubOffsetOffset = 6;

  const uniformValues = useMemo(() => {
    const uv = new Float32Array(ubSize / 4);
    uv.set([1, 1, 0, 1], ubColorOffset);
    uv.set([1, 1], ubScaleOffset);
    uv.set([-0.5, -0.25], ubOffsetOffset);
    return uv;
  }, [ubOffsetOffset, ubScaleOffset, ubSize]);

  const uniformBuffer = useMemo(() => {
    const ub = device.createBuffer({
      size: ubSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    return ub;
  }, [device, ubSize]);

  const bindGroup = useMemo(
    () =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
      }),
    [device, pipeline, uniformBuffer]
  );

  useAnimationFrame(
    useCallback(() => {
      const t = Date.now() / 1000;
      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const aspectRatio = context.canvas.width / context.canvas.height;
      uniformValues.set([0.5 / aspectRatio, 0.5], ubScaleOffset);
      uniformValues.set([Math.sin(t) * 0.5, Math.cos(t) * 0.5], ubOffsetOffset);

      device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

      const encoder = device.createCommandEncoder({
        label: "basic command encoder",
      });
      const pass = encoder.beginRenderPass(renderPassDescriptor);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();

      const commandBuffer = encoder.finish();
      device.queue.submit([commandBuffer]);
    }, [
      bindGroup,
      context,
      device,
      pipeline,
      renderPassDescriptor,
      ubScaleOffset,
      uniformBuffer,
      uniformValues,
    ])
  );

  return null;
}
