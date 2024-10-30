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
      const name = ["read", "write", "prev"][i];
      const label = `${oldLabel} ${name}`;
      const texture = device.createTexture({ ...descriptor, label });
      writeInitialData?.(texture);
      return texture;
    });
    this.readTex = textures[0];
    this.writeTex = textures[1];
    this.prevTex = textures[2];
  }

  flip(encoder: GPUCommandEncoder) {
    encoder.copyTextureToTexture(
      { texture: this.writeTex },
      { texture: this.readTex },
      [this.readTex.width, this.readTex.height]
    );
  }

  swap(encoder: GPUCommandEncoder) {
    encoder.copyTextureToTexture(
      { texture: this.writeTex },
      { texture: this.prevTex },
      [this.readTex.width, this.readTex.height]
    );
  }
}
