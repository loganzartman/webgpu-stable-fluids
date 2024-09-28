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

  useAnimationFrame(
    useCallback(() => {
      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const encoder = device.createCommandEncoder({
        label: "basic command encoder",
      });
      const pass = encoder.beginRenderPass(renderPassDescriptor);
      pass.setPipeline(pipeline);
      pass.draw(3);
      pass.end();

      const commandBuffer = encoder.finish();
      device.queue.submit([commandBuffer]);
    }, [context, device, pipeline, renderPassDescriptor])
  );

  return null;
}
