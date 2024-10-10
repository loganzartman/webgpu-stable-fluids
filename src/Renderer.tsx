import { useCallback, useMemo } from "react";
import { useAnimationFrame } from "./useAnimationFrame";
import renderModuleCode from "./renderModule.wgsl?raw";
import { diffuseModuleCode } from "./diffuseModule";
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
    const data = new Float32Array((N + 2) * (N + 2));
    for (let x = 1; x <= N; ++x) {
      for (let y = 1; y <= N; ++y) {
        const dx = x - (N + 2) / 2;
        const dy = y - (N + 2) / 2;
        const dist = Math.hypot(dx, dy);
        if (dist < 20) {
          data[y * (N + 2) + x] = 1;
        }
      }
    }

    return new ReadWritePrevTex({
      device,
      descriptor: {
        label: `density texture`,
        format: "r32float",
        size: [N + 2, N + 2],
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
            bytesPerRow: data.BYTES_PER_ELEMENT * (N + 2),
            rowsPerImage: N + 2,
          },
          [N + 2, N + 2]
        );
      },
    });
  }, [device]);

  const velocityRwp = useMemo(() => {
    const data = new Float32Array((N + 2) * (N + 2) * 2);
    for (let x = 1; x <= N; ++x) {
      for (let y = 1; y <= N; ++y) {
        const dx = x - (N + 2) / 2;
        const dy = y - (N + 2) / 2;
        const dist = Math.hypot(dx, dy);
        if (dist < 20) {
          data[(y * (N + 2) + x) * 2 + 0] = 1;
          data[(y * (N + 2) + x) * 2 + 1] = 0;
        }
      }
    }

    return new ReadWritePrevTex({
      device,
      descriptor: {
        label: "velocity texture",
        format: "rg32float",
        size: [N + 2, N + 2],
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
            bytesPerRow: data.BYTES_PER_ELEMENT * (N + 2) * 2,
            rowsPerImage: N + 2,
          },
          [N + 2, N + 2]
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
        code: diffuseModuleCode({ workgroupDim }),
      }),
    [device]
  );

  const advectRModule = useMemo(
    () =>
      device.createShaderModule({
        label: "advect module",
        code: advectModuleCode({ texFormat: "r32float", workgroupDim }),
      }),
    [device]
  );

  const advectRGModule = useMemo(
    () =>
      device.createShaderModule({
        label: "advect module",
        code: advectModuleCode({ texFormat: "rg32float", workgroupDim }),
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

  const advectRPipeline = useMemo(
    () =>
      device.createComputePipeline({
        label: "advect r32f pipeline",
        layout: "auto",
        compute: {
          module: advectRModule,
          entryPoint: "advect",
        },
      }),
    [advectRModule, device]
  );

  const advectRGPipeline = useMemo(
    () =>
      device.createComputePipeline({
        label: "advect rg32f pipeline",
        layout: "auto",
        compute: {
          module: advectRGModule,
          entryPoint: "advect",
        },
      }),
    [advectRGModule, device]
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
          Math.ceil((N + 2) / workgroupDim),
          Math.ceil((N + 2) / workgroupDim)
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
      velocityTex,
      dt,
    }: {
      encoder: GPUCommandEncoder;
      target: ReadWritePrevTex;
      targetSampler: GPUSampler;
      velocityTex: GPUTexture;
      dt: number;
    }) => {
      advectUniformsBuffer.write({
        N,
        dt,
      });

      let pipeline;
      switch (target.readTex.format) {
        case "r32float":
          pipeline = advectRPipeline;
          break;
        case "rg32float":
          pipeline = advectRGPipeline;
          break;
        default:
          throw new Error("Invalid texture format for advect()");
      }

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: advectUniformsBuffer },
          { binding: 1, resource: target.readTex.createView() },
          { binding: 2, resource: targetSampler },
          { binding: 3, resource: target.writeTex.createView() },
          { binding: 4, resource: velocityTex.createView() },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(
        Math.ceil((N + 2) / workgroupDim),
        Math.ceil((N + 2) / workgroupDim)
      );
      pass.end();

      target.swap();
    },
    [advectRGPipeline, advectRPipeline, advectUniformsBuffer, device]
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
      const dt = 0.001;
      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const encoder = device.createCommandEncoder({
        label: "Field display encoder",
      });

      // densityRwp.commit();

      // diffuse({
      //   encoder,
      //   target: densityRwp,
      //   diff: 0.01,
      //   dt,
      //   iters: 10,
      // });

      densityRwp.commit();

      advect({
        encoder,
        target: densityRwp,
        targetSampler: densitySampler,
        velocityTex: velocityRwp.readTex,
        dt,
      });

      velocityRwp.commit();

      advect({
        encoder,
        target: velocityRwp,
        targetSampler: densitySampler,
        velocityTex: velocityRwp.prevTex,
        dt,
      });

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
      densityRwp.readTex,
      densitySampler,
      device,
      diffuseUniformsBuffer,
      renderPassDescriptor,
      renderPipeline,
      velocityRwp,
    ])
  );

  return null;
}
