export class ReadWritePrevTex {
  readTex: GPUTexture;
  writeTex: GPUTexture;
  prevTex: GPUTexture;

  constructor({
    device,
    descriptor,
    writeInitialData,
  }: {
    device: GPUDevice;
    descriptor: GPUTextureDescriptor;
    writeInitialData?: (t: GPUTexture) => void;
  }) {
    const oldLabel = descriptor.label ?? "<unnamed>";
    const textures = Array.from({ length: 3 }).map((_, i) => {
      const label = `${oldLabel} RWP #${i}`;
      const texture = device.createTexture({ ...descriptor, label });
      writeInitialData?.(texture);
      return texture;
    });
    this.readTex = textures[0];
    this.writeTex = textures[1];
    this.prevTex = textures[2];
  }

  swap() {
    [this.readTex, this.writeTex] = [this.writeTex, this.readTex];
  }

  commit() {
    [this.prevTex, this.writeTex] = [this.writeTex, this.prevTex];
  }
}
