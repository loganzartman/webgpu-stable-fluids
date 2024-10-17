import { useCallback, useEffect, useMemo, useRef } from "react";
import { useAnimationFrame } from "./useAnimationFrame";
import renderModuleCode from "./renderModule.wgsl?raw";
import { diffuseModuleCode } from "./diffuseModule";
import { advectModuleCode } from "./advectModule";
import { f32, u32, struct, vec2f } from "typegpu/data";
import tgpu from "typegpu";
import { ReadWritePrevTex } from "./ReadWritePrevTex";
import { projectModuleCode } from "./projectModule";
import { splatModuleCode } from "./splatModule";

const N = 1024;
const workgroupDim = 8;

export function Renderer({
  context,
  device,
}: {
  context: GPUCanvasContext;
  adapter: GPUAdapter;
  device: GPUDevice;
}) {
  const { current: pointer } = useRef({
    x: 0,
    y: 0,
    px: 0,
    py: 0,
    down: false,
  });

  useEffect(() => {
    function onMove(e: PointerEvent) {
      const rect = document.body.getBoundingClientRect();
      pointer.x = (e.clientX / rect.width) * N;
      pointer.y = (e.clientY / rect.height) * N;
    }
    function onDown(e: PointerEvent) {
      onMove(e);
      pointer.down = true;
    }
    function onUp() {
      pointer.down = false;
    }

    window.addEventListener("pointerdown", onDown, false);
    window.addEventListener("pointerup", onUp, false);
    window.addEventListener("pointermove", onMove, false);
    return () => {
      window.removeEventListener("pointerdown", onDown, false);
      window.removeEventListener("pointerup", onUp, false);
      window.removeEventListener("pointermove", onMove, false);
    };
  }, [pointer]);

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

  const divergenceRwp = useMemo(() => {
    return new ReadWritePrevTex({
      device,
      descriptor: {
        label: "divergence texture",
        format: "r32float",
        size: [N + 2, N + 2],
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      },
    });
  }, [device]);

  const pressure1Rwp = useMemo(() => {
    return new ReadWritePrevTex({
      device,
      descriptor: {
        label: "pressure1 texture",
        format: "r32float",
        size: [N + 2, N + 2],
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      },
    });
  }, [device]);

  const pressure2Rwp = useMemo(() => {
    return new ReadWritePrevTex({
      device,
      descriptor: {
        label: "pressure2 texture",
        format: "r32float",
        size: [N + 2, N + 2],
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      },
    });
  }, [device]);

  const linearSampler = useMemo(
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

  const projectModule = useMemo(
    () =>
      device.createShaderModule({
        label: "project module",
        code: projectModuleCode({ workgroupDim }),
      }),
    [device]
  );

  const splatModule = useMemo(
    () =>
      device.createShaderModule({
        label: "splat module",
        code: splatModuleCode({ workgroupDim }),
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

  const projectUniformsBuffer = useMemo(
    () =>
      tgpu
        .createBuffer(
          struct({
            N: u32,
            dt: f32,
            i: u32,
          }),
          {
            N,
            dt: 0,
            i: 0,
          }
        )
        .$device(device)
        .$usage(tgpu.Uniform),
    [device]
  );

  const splatUniformsBuffer = useMemo(
    () =>
      tgpu
        .createBuffer(
          struct({
            N: u32,
            dt: f32,
            position: vec2f,
            velocity: vec2f,
            radius: f32,
            amount: f32,
          })
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

  const projectInitPipeline = useMemo(
    () =>
      device.createComputePipeline({
        label: "projectInit pipeline",
        layout: "auto",
        compute: {
          module: projectModule,
          entryPoint: "projectInit",
        },
      }),
    [device, projectModule]
  );

  const projectSolvePipeline = useMemo(
    () =>
      device.createComputePipeline({
        label: "projectSolve pipeline",
        layout: "auto",
        compute: {
          module: projectModule,
          entryPoint: "projectSolve",
        },
      }),
    [device, projectModule]
  );

  const projectApplyPipeline = useMemo(
    () =>
      device.createComputePipeline({
        label: "projectApply pipeline",
        layout: "auto",
        compute: {
          module: projectModule,
          entryPoint: "projectApply",
        },
      }),
    [device, projectModule]
  );

  const splatPipeline = useMemo(
    () =>
      device.createComputePipeline({
        label: "splat pipeline",
        layout: "auto",
        compute: {
          module: splatModule,
          entryPoint: "splat",
        },
      }),
    [device, splatModule]
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

        target.flip();
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

      target.flip();
    },
    [advectRGPipeline, advectRPipeline, advectUniformsBuffer, device]
  );

  const project = useCallback(
    ({
      encoder,
      dt,
      iters,
      pressureTarget,
    }: {
      encoder: GPUCommandEncoder;
      dt: number;
      iters: number;
      pressureTarget: ReadWritePrevTex;
    }) => {
      projectUniformsBuffer.write({
        N,
        dt,
        i: 0,
      });

      // init
      const initBindGroup = device.createBindGroup({
        layout: projectInitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: projectUniformsBuffer },
          { binding: 1, resource: velocityRwp.readTex.createView() },
          { binding: 2, resource: velocityRwp.writeTex.createView() },
          { binding: 3, resource: divergenceRwp.readTex.createView() },
          { binding: 4, resource: divergenceRwp.writeTex.createView() },
          { binding: 5, resource: pressureTarget.readTex.createView() },
          { binding: 6, resource: pressureTarget.writeTex.createView() },
        ],
      });

      const initPass = encoder.beginComputePass();
      initPass.setPipeline(projectInitPipeline);
      initPass.setBindGroup(0, initBindGroup);
      initPass.dispatchWorkgroups(
        Math.ceil((N + 2) / workgroupDim),
        Math.ceil((N + 2) / workgroupDim)
      );
      initPass.end();
      divergenceRwp.flip();
      pressureTarget.flip();

      // solve
      for (let i = 0; i < iters; ++i) {
        projectUniformsBuffer.write({
          N,
          dt,
          i,
        });

        const solveBindGroup = device.createBindGroup({
          layout: projectSolvePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: projectUniformsBuffer },
            { binding: 1, resource: velocityRwp.readTex.createView() },
            { binding: 2, resource: velocityRwp.writeTex.createView() },
            { binding: 3, resource: divergenceRwp.readTex.createView() },
            { binding: 4, resource: divergenceRwp.writeTex.createView() },
            { binding: 5, resource: pressureTarget.readTex.createView() },
            { binding: 6, resource: pressureTarget.writeTex.createView() },
          ],
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(projectSolvePipeline);
        pass.setBindGroup(0, solveBindGroup);
        pass.dispatchWorkgroups(
          Math.ceil((N + 2) / workgroupDim),
          Math.ceil((N + 2) / workgroupDim)
        );
        pass.end();
        pressureTarget.flip();
      }

      // apply
      const applyBindGroup = device.createBindGroup({
        layout: projectApplyPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: projectUniformsBuffer },
          { binding: 1, resource: velocityRwp.readTex.createView() },
          { binding: 2, resource: velocityRwp.writeTex.createView() },
          { binding: 3, resource: divergenceRwp.readTex.createView() },
          { binding: 4, resource: divergenceRwp.writeTex.createView() },
          { binding: 5, resource: pressureTarget.readTex.createView() },
          { binding: 6, resource: pressureTarget.writeTex.createView() },
        ],
      });

      const applyPass = encoder.beginComputePass();
      applyPass.setPipeline(projectApplyPipeline);
      applyPass.setBindGroup(0, applyBindGroup);
      applyPass.dispatchWorkgroups(
        Math.ceil((N + 2) / workgroupDim),
        Math.ceil((N + 2) / workgroupDim)
      );
      applyPass.end();
      velocityRwp.flip();
    },
    [
      device,
      divergenceRwp,
      projectApplyPipeline,
      projectInitPipeline,
      projectSolvePipeline,
      projectUniformsBuffer,
      velocityRwp,
    ]
  );

  const splat = useCallback(
    ({
      encoder,
      dt,
      x,
      y,
      vx,
      vy,
      radius,
      amount,
    }: {
      encoder: GPUCommandEncoder;
      dt: number;
      x: number;
      y: number;
      vx: number;
      vy: number;
      radius: number;
      amount: number;
    }) => {
      splatUniformsBuffer.write({
        N,
        dt,
        position: vec2f(x, y),
        velocity: vec2f(vx, vy),
        radius,
        amount,
      });

      const bindGroup = device.createBindGroup({
        layout: splatPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: splatUniformsBuffer },
          { binding: 1, resource: densityRwp.readTex.createView() },
          { binding: 2, resource: densityRwp.writeTex.createView() },
          { binding: 3, resource: densityRwp.prevTex.createView() },
          { binding: 4, resource: velocityRwp.readTex.createView() },
          { binding: 5, resource: velocityRwp.writeTex.createView() },
          { binding: 6, resource: velocityRwp.prevTex.createView() },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(splatPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(
        Math.ceil((N + 2) / workgroupDim),
        Math.ceil((N + 2) / workgroupDim)
      );
      pass.end();

      densityRwp.flip();
      velocityRwp.flip();
    },
    [densityRwp, device, splatPipeline, splatUniformsBuffer, velocityRwp]
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
      const dt = 0.01;
      renderPassDescriptor.colorAttachments[0].view = context
        .getCurrentTexture()
        .createView();

      const encoder = device.createCommandEncoder({
        label: "Field display encoder",
      });

      if (pointer.down) {
        splat({
          encoder,
          dt,
          x: pointer.x,
          y: pointer.y,
          vx: (pointer.x - pointer.px) * 0.1,
          vy: (pointer.y - pointer.py) * 0.1,
          radius: N / 50,
          amount: 1,
        });
        densityRwp.swap();
      } else {
        // TODO: why???
        densityRwp.flip();
        velocityRwp.flip();
      }

      // velocity step
      {
        // todo swap velocity
        // todo diffuse velocity

        project({
          encoder,
          dt,
          iters: 100,
          pressureTarget: pressure1Rwp,
        });

        velocityRwp.swap();

        advect({
          encoder,
          dt,
          target: velocityRwp,
          targetSampler: linearSampler,
          velocityTex: velocityRwp.prevTex,
        });

        project({
          encoder,
          dt,
          iters: 100,
          pressureTarget: pressure2Rwp,
        });
      }

      // density step
      {
        densityRwp.swap();

        diffuse({
          encoder,
          dt,
          diff: 0,
          iters: 20,
          target: densityRwp,
        });

        densityRwp.swap();

        advect({
          encoder,
          dt,
          target: densityRwp,
          targetSampler: linearSampler,
          velocityTex: velocityRwp.readTex,
        });
      }

      const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: diffuseUniformsBuffer },
          { binding: 1, resource: densityRwp.readTex.createView() },
          { binding: 2, resource: linearSampler },
        ],
      });

      const renderPass = encoder.beginRenderPass(renderPassDescriptor);
      renderPass.setPipeline(renderPipeline);
      renderPass.setBindGroup(0, renderBindGroup);
      renderPass.draw(6);
      renderPass.end();

      device.queue.submit([encoder.finish()]);

      pointer.px = pointer.x;
      pointer.py = pointer.y;
    }, [
      renderPassDescriptor,
      context,
      device,
      pointer,
      renderPipeline,
      diffuseUniformsBuffer,
      densityRwp,
      linearSampler,
      splat,
      velocityRwp,
      project,
      pressure1Rwp,
      advect,
      pressure2Rwp,
      diffuse,
    ])
  );

  return null;
}
