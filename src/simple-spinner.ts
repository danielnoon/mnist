declare module "simple-spinner" {
  export function start(interval?: number): void; 
  export function stop(): void; 
  export function changeSequence(chars: string[]): void; 
}