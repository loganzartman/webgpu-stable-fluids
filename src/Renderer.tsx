import { useCallback, useMemo } from "react";
import { useAnimationFrame } from "./useAnimationFrame";
import renderModuleCode from "./renderModule.wgsl?raw";
import diffuseModuleCode from "./diffuseModule.wgsl?raw";
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

  let [densityReadTex, densityWriteTex, densityPrevTex] = useMemo(() => {
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

    return Array.from({ length: 3 }).map((_, i) => {
      const texture = device.createTexture({
        label: `density texture ${i}`,
        format: "r32float",
        size: [N, N],
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });

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
    });
  }, [device]);

  const densitySampler = useMemo(
    () =>
      device.createSampler({
        minFilter: "linear",
        magFilter: "linear",
      }),
    [device]
  );

  const diffuseModule = useMemo(
    () =>
      device.createShaderModule({
        label: "diffuse module",
        code: diffuseModuleCode,
      }),
    [device]
  );

  const diffuseUniformsBuffer = useMemo(
    () =>
      tgpu
        .createBuffer(
          struct({
            N: u32,
            diff: f32,
            dt: f32,
          }),
          {
            N,
            diff: 0,
            dt: 0,
          }
        )
        .$device(device)
        .$usage(tgpu.Uniform),
    [device]
  );

  const diffuseStepPipeline = useMemo(
    () =>
      device.createComputePipeline({
        label: "diffuseStep pipeline",
        layout: "auto",
        compute: {
          module: diffuseModule,
          entryPoint: "diffuseStep",
        },
      }),
    [device, diffuseModule]
  );

  const diffuse = useCallback(
    ({ encoder, iters }: { encoder: GPUCommandEncoder; iters: number }) => {
      diffuseUniformsBuffer.write({
        N,
        diff: 0.01,
        dt: 0.01,
      });
      for (let i = 0; i < iters; ++i) {
        const bindGroup = device.createBindGroup({
          layout: diffuseStepPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: diffuseUniformsBuffer },
            { binding: 1, resource: densityReadTex.createView() },
            { binding: 2, resource: densityWriteTex.createView() },
            { binding: 3, resource: densityPrevTex.createView() },
          ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(diffuseStepPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
          Math.ceil(N / workgroupDim),
          Math.ceil(N / workgroupDim)
        );
        pass.end();
        [densityReadTex, densityWriteTex] = [densityWriteTex, densityReadTex];
      }
      [densityPrevTex, densityWriteTex] = [densityWriteTex, densityPrevTex];
    },
    [
      densityPrevTex,
      densityReadTex,
      densityWriteTex,
      device,
      diffuseStepPipeline,
      diffuseUniformsBuffer,
    ]
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

  useAnimationFrame(
    useCallback(() => {
      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const encoder = device.createCommandEncoder({
        label: "Field display encoder",
      });

      diffuse({ encoder, iters: 10 });

      const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: diffuseUniformsBuffer },
          { binding: 1, resource: densityReadTex.createView() },
          { binding: 2, resource: densitySampler },
        ],
      });

      const renderPass = encoder.beginRenderPass(renderPassDescriptor);
      renderPass.setPipeline(renderPipeline);
      renderPass.setBindGroup(0, renderBindGroup);
      renderPass.draw(6);
      renderPass.end();

      device.queue.submit([encoder.finish()]);
    }, [
      context,
      densityReadTex,
      densitySampler,
      device,
      diffuse,
      diffuseUniformsBuffer,
      renderPassDescriptor,
      renderPipeline,
    ])
  );

  return null;
}
