import { useCallback, useMemo } from "react";
import { useAnimationFrame } from "./useAnimationFrame";
import renderModuleCode from "./renderModule.wgsl?raw";
import diffuseModuleCode from "./diffuseModule.wgsl?raw";
import { advectModuleCode } from "./advectModule";
import { f32, u32, struct } from "typegpu/data";
import tgpu from "typegpu";
import { ReadWritePrevTex } from "./ReadWritePrevTex";

const N = 256;
const workgroupDim = 8;

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

  const densityRwp = useMemo(() => {
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

    return new ReadWritePrevTex({
      device,
      descriptor: {
        label: `density texture`,
        format: "r32float",
        size: [N, N],
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      },
      writeInitialData(texture) {
        device.queue.writeTexture(
          { texture },
          data,
          {
            bytesPerRow: data.BYTES_PER_ELEMENT * N,
            rowsPerImage: N,
          },
          [N, N]
        );
      },
    });
  }, [device]);

  const velocityRwp = useMemo(() => {
    const data = new Float32Array(N * N * 2);
    for (let x = 0; x < N; ++x) {
      for (let y = 0; y < N; ++y) {
        data[(y * N + x) * 2] = 1;
        data[(y * N + x) * 2 + 1] = 0.1;
      }
    }

    return new ReadWritePrevTex({
      device,
      descriptor: {
        label: "velocity texture",
        format: "rg32float",
        size: [N, N],
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      },
      writeInitialData(texture) {
        device.queue.writeTexture(
          { texture },
          data,
          {
            bytesPerRow: data.BYTES_PER_ELEMENT * N * 2,
            rowsPerImage: N,
          },
          [N, N]
        );
      },
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

  const advectModule = useMemo(
    () =>
      device.createShaderModule({
        label: "advect module",
        code: advectModuleCode({ texFormat: "r32float", workgroupDim }),
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

  const advectUniformsBuffer = useMemo(
    () =>
      tgpu
        .createBuffer(
          struct({
            N: u32,
            dt: f32,
          }),
          {
            N,
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

  const advectPipeline = useMemo(
    () =>
      device.createComputePipeline({
        label: "advect pipeline",
        layout: "auto",
        compute: {
          module: advectModule,
          entryPoint: "advect",
        },
      }),
    [advectModule, device]
  );

  const diffuse = useCallback(
    ({
      encoder,
      target,
      diff,
      dt,
      iters,
    }: {
      encoder: GPUCommandEncoder;
      target: ReadWritePrevTex;
      diff: number;
      dt: number;
      iters: number;
    }) => {
      diffuseUniformsBuffer.write({
        N,
        diff,
        dt,
      });

      for (let i = 0; i < iters; ++i) {
        const bindGroup = device.createBindGroup({
          layout: diffuseStepPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: diffuseUniformsBuffer },
            { binding: 1, resource: target.readTex.createView() },
            { binding: 2, resource: target.writeTex.createView() },
            { binding: 3, resource: target.prevTex.createView() },
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

        target.swap();
      }
    },
    [device, diffuseStepPipeline, diffuseUniformsBuffer]
  );

  const advect = useCallback(
    ({
      encoder,
      target,
      targetSampler,
      dt,
    }: {
      encoder: GPUCommandEncoder;
      target: ReadWritePrevTex;
      targetSampler: GPUSampler;
      dt: number;
    }) => {
      advectUniformsBuffer.write({
        N,
        dt,
      });

      const bindGroup = device.createBindGroup({
        layout: advectPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: advectUniformsBuffer },
          { binding: 1, resource: target.readTex.createView() },
          { binding: 2, resource: targetSampler },
          { binding: 3, resource: target.writeTex.createView() },
          { binding: 4, resource: velocityRwp.readTex.createView() },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(advectPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(
        Math.ceil(N / workgroupDim),
        Math.ceil(N / workgroupDim)
      );
      pass.end();

      target.swap();
    },
    [advectPipeline, advectUniformsBuffer, device, velocityRwp.readTex]
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
      const dt = 0.005;
      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const encoder = device.createCommandEncoder({
        label: "Field display encoder",
      });

      diffuse({
        encoder,
        target: densityRwp,
        diff: 0.01,
        dt,
        iters: 10,
      });

      advect({
        encoder,
        target: densityRwp,
        targetSampler: densitySampler,
        dt,
      });

      densityRwp.commit();
      velocityRwp.commit();

      const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: diffuseUniformsBuffer },
          { binding: 1, resource: densityRwp.readTex.createView() },
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
      advect,
      context,
      densityRwp,
      densitySampler,
      device,
      diffuse,
      diffuseUniformsBuffer,
      renderPassDescriptor,
      renderPipeline,
      velocityRwp,
    ])
  );

  return null;
}
