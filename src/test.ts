import { MNISTData } from "./loadImage";
import '@tensorflow/tfjs-node';

async function main() {
  console.log("Loading data...");
  const data = new MNISTData();
  await data.load();
  console.log("Making image");
  const image: number[][] = [];
  for (let i = 0; i < 784; i++) {
    let y = Math.floor(i / 28);
    let x = i % 28;

    if (!image[y]) image[y] = [];
    image[y][x] = data.imageBuffer[i];
  }
  console.log("Emitting image!");
  console.log(image);
  console.log("Getting first batch...");
  const batch1 = data.getNextBatch(64);
  console.log(batch1.labels);
  console.log(Array.from(batch1.labels.dataSync()).slice(0, 10));
  data.writeNumber(batch1.xs, batch1.labels);
}

main();
