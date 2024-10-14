import { useEffect, useState } from "react";
import "./App.css";
import Canvas from "./Canvas";
import { Renderer } from "./Renderer";
import { ErrorBoundary } from "./ErrorBoundary";

function App() {
  const [error, setError] = useState<Error | null>(null);
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  const [context, setContext] = useState<GPUCanvasContext | null>(null);
  const [adapter, setAdapter] = useState<GPUAdapter | null>(null);
  const [device, setDevice] = useState<GPUDevice | null>(null);

  useEffect(() => {
    if (!canvas) return;
    (async () => {
      if (!navigator.gpu) {
        throw new Error("navigator.gpu not available");
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error("requestAdapter failed");
      }
      const device = await adapter.requestDevice({
        requiredFeatures: ["float32-filterable"],
      });
      if (!device) {
        throw new Error("requestDevice failed");
      }
      const context = canvas.getContext("webgpu");
      if (!context) {
        throw new Error("failed to get context");
      }
      setAdapter(adapter);
      setDevice(device);
      setContext(context);
    })().catch((e) => {
      setError(e);
      console.error(e);
    });
  }, [canvas]);

  return (
    <div style={{ height: "100%" }}>
      {error && (
        <div className="error-container">
          <div className="error-header">Error occurred:</div>
          <div>{String(error)}</div>
        </div>
      )}
      <ErrorBoundary onError={(e) => setError(e)}>
        <Canvas ref={setCanvas} />
        {context && adapter && device && (
          <Renderer context={context} adapter={adapter} device={device} />
        )}
      </ErrorBoundary>
    </div>
  );
}

export default App;
