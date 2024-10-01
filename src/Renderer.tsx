import { useCallback, useMemo } from "react";
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

  useAnimationFrame(useCallback(() => {}, []));

  return null;
}
