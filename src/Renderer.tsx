import { useCallback, useMemo } from "react";
import { useAnimationFrame } from "./useAnimationFrame";
import renderModuleCode from "./renderModule.wgsl?raw";

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
  */

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

  useAnimationFrame(
    useCallback(() => {
      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const encoder = device.createCommandEncoder({
        label: "Field display encoder",
      });

      const renderPass = encoder.beginRenderPass(renderPassDescriptor);
      renderPass.setPipeline(renderPipeline);
      renderPass.draw(6);
      renderPass.end();

      device.queue.submit([encoder.finish()]);
    }, [context, device, renderPassDescriptor, renderPipeline])
  );

  return null;
}
